"""Model-agnostic inference loop for the YAM bimanual robot (+ HG-DAgger).

Provides:

- ``ModelBridge`` — abstract interface for model-specific logic.
- ``RaidenInferenceLoop`` — camera/motor infrastructure + control loop, plus the
  ``--intervene`` HG-DAgger interactive-correction state machine.

The split: **raiden** owns hardware (cameras, motors, 14-D joint commands); the
**model repo** owns everything model-specific (action space, preprocessing,
inference, action chunking) behind the :class:`ModelBridge` seam.

Usage from a model repo::

    # your_model/deployment/yam_bridge.py
    from raiden.inference import ModelBridge, RaidenInferenceLoop

    class MyBridge(ModelBridge):
        def load(self, ckpt_path, **kwargs): ...
        def predict(self, obs):             ...  # returns (14,)

Or via CLI::

    rd infer --bridge your_model.deployment.yam_bridge:MyBridge \\
             --ckpt-path /path/to/checkpoint.pt

HG-DAgger (interactive correction)::

    rd infer --bridge ... --ckpt-path ... --intervene

    # Leader TOP button  : toggle policy <-> teleop (take over / hand back)
    # Leader BOTTOM button: end the episode (looped collection only)

Thread layout (inherited from RaidenPolicyServer)::

    camera-<name>       : grabs frames from ZED at ~30 Hz (per camera)
    proprio-<name>      : reads joint state at ~100 Hz
    dagger-input        : owns ALL leader CAN I/O (reads + shadow commands)
    main thread         : inference loop (gated by bridge.predict() speed)

Arm-ordering convention (TRI)
-----------------------------
The 14-D joint action is **LEFT-then-RIGHT**::

    action[0:7]   = left arm  (6 joints + gripper)   → follower_l
    action[7:14]  = right arm (6 joints + gripper)   → follower_r

This matches the TRI server's ``_smooth_command``, ``_check_joint_delta``, and
``_ee_pose_to_joint_cmd`` (which returns ``concatenate([left, right])``), so
joint mode and ee_pose mode command the same arms.  (The YAM source packed
RIGHT-then-LEFT; this port adopts TRI's order end to end.)
"""

import importlib
import shutil
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from chiral.types import Observation

from raiden.robot.controller import joint_delta_takeover, smooth_move_joints
from raiden.server import RaidenPolicyServer

DOF = 7  # joints per arm (6 revolute + 1 gripper)


# ---------------------------------------------------------------------------
# Abstract bridge interface
# ---------------------------------------------------------------------------


class ModelBridge(ABC):
    """Abstract interface between a policy model and the YAM robot.

    Implement this in your model's repo to handle all model-specific logic:

    - Model loading (checkpoint format, framework imports)
    - Observation preprocessing (image resize, pointmaps, state assembly,
      normalization — whatever your model's action space requires)
    - Inference (forward pass)
    - Action postprocessing (denormalization, chunking, format conversion)

    ``predict()`` receives a raw :class:`chiral.types.Observation` and must
    return a ``(14,)`` float32 array of motor commands in **LEFT-then-RIGHT**
    joint order (TRI convention)::

        [0:6]   left arm joint angles (rad)
        [6]     left gripper (0=closed, 1=open)
        [7:13]  right arm joint angles (rad)
        [13]    right gripper (0=closed, 1=open)

    This is the same layout the TRI server's ``_ee_pose_to_joint_cmd`` produces
    for ``action_type="ee_pose"``, so both action types command the same arms.

    Action chunking, if used, must be managed internally — ``predict()`` is
    called once per control step and returns exactly one action.
    """

    @abstractmethod
    def load(self, ckpt_path: str, **kwargs) -> None:
        """Load model from checkpoint.

        Args:
            ckpt_path: Path to the model checkpoint.
            **kwargs (Any): Model-specific arguments (e.g. ``chunk_size``,
                ``device``).
        """

    @abstractmethod
    def predict(self, obs: Observation) -> np.ndarray:
        """Given a raw observation, return an action for the loop's action type.

        The return shape depends on the ``--action-type`` the loop runs with:

        - ``joint`` (default): ``(14,)`` float32 joint commands
          ``[left_joints(6), left_grip(1), right_joints(6), right_grip(1)]``,
          sent to the motors as-is.
        - ``ee_pose``: ``(20,)`` float32 end-effector pose
          ``[l_xyz(3), r_xyz(3), l_rot6d(6), r_rot6d(6), l_grip(1), r_grip(1)]``,
          which the loop converts to a ``(14,)`` joint command via the server's
          ``_ee_pose_to_joint_cmd`` (IK). Left-before-right throughout.

        Args:
            obs: Raw observation with ``obs.cameras`` (list of ``CameraInfo``),
                ``obs.proprios`` (dict of ``(N,)`` float32 arrays), and
                ``obs.timestamp`` (float).
        """

    def reset(self) -> None:
        """Called when the robot is homed / control is handed back.

        Override to reset internal state (e.g. clear the action-chunk buffer so
        the next ``predict()`` re-infers on the current observation).  This is
        the whole HG-DAgger handback seam — a fresh inference on the exact pose
        the human left, giving a zero-jump resume.
        """


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


class RaidenInferenceLoop(RaidenPolicyServer):
    """Model-agnostic inference loop for the YAM robot (+ HG-DAgger).

    Reuses :class:`~raiden.server.RaidenPolicyServer`'s camera capture, depth
    computation, proprioception reading, and motor control, delegating all
    model-specific logic to a :class:`ModelBridge`.

    Args:
        bridge: a ``ModelBridge`` implementation from the model repo.
        ckpt_path: passed to ``bridge.load()``.
        action_hz: control-loop frequency in Hz (default 30, should match the
            training data frame rate).
        bridge_kwargs: extra keyword arguments forwarded to ``bridge.load()``.
        intervene: bring up the passive leader arms + the policy<->teleop button
            state machine (implies ``record``).
        record: record the rollout with a per-frame ``control_source`` flag.
        dagger_task: task / round name the rollout is saved under (skips the
            interactive task prompt).
        dagger_instruction: language instruction stored with the episode.
        reset_pose: optional path to a ``robot_data.npz`` whose first frame the
            followers are moved to before the rollout.
        leader_track: shadow tracking — in policy mode the leaders continuously
            mirror the followers so takeover is instant + seam-free (default on).
        data_dir: root data dir; recordings go to ``<data_dir>/raw/<task>/``.
        **kwargs (Any): forwarded to ``RaidenPolicyServer`` (camera_config_file,
            calibration_file, stereo_method, action_type, max_joint_delta, ...).
    """

    def __init__(
        self,
        bridge: ModelBridge,
        ckpt_path: str,
        action_hz: float = 30.0,
        bridge_kwargs: Optional[dict] = None,
        intervene: bool = False,
        record: bool = False,
        dagger_task: Optional[str] = None,
        dagger_instruction: Optional[str] = None,
        reset_pose: Optional[str] = None,
        leader_track: bool = True,
        data_dir: str = "data",
        **kwargs,
    ):
        self._bridge = bridge
        self._action_hz = action_hz
        self._data_dir = data_dir

        # HG-DAgger config.  ``intervene`` brings up the (passive) leader arms
        # and a button-driven policy<->teleop state machine; it implies recording.
        self._intervene = intervene
        self._record = record or intervene
        self._dagger_task = dagger_task
        self._dagger_instruction = dagger_instruction
        self._reset_pose = reset_pose
        # Shadow tracking: during policy mode, continuously drive the leaders onto
        # the followers so takeover is instant + seam-free (vs the ~1.5 s blocking
        # sync used when this is off).  See _dagger_input_loop.
        self._leader_track = leader_track

        # State-machine state (only used when intervening).
        self._mode = "policy"  # "policy" | "teleop"
        self._mode_lock = threading.Lock()
        self._leader_state: dict = {"q7_l": None, "q7_r": None}
        # Latest measured follower arm pose, published by the main loop for the
        # input thread to shadow the leaders onto (no extra follower CAN reads).
        self._shadow_target: dict = {"l": None, "r": None}
        self._dagger_stop = threading.Event()
        self._dagger_thread: Optional[threading.Thread] = None
        # Per-physical-leader button edge state (either leader toggles).
        self._last_btn = {"r": 0.0, "l": 0.0}
        self._last_toggle_t = 0.0
        # End-episode signal: a leader BOTTOM-button press sets this so the looped
        # collector (run_session) finishes the current episode cleanly instead of
        # Ctrl+C.  Ignored by single-rollout run() (end_on_button=False).
        self._end_episode_evt = threading.Event()
        self._last_btn_bottom = {"r": 0.0, "l": 0.0}
        self._last_end_toggle_t = 0.0
        self._episode_steps = 0
        # Resolved once per session (DB task or fzf picker) by _resolve_dagger_task.
        self._dagger_task_name: Optional[str] = None
        self._dagger_instruction_resolved: str = ""

        # Load model FIRST — this can take seconds (downloading weights, building
        # the network) and doesn't need hardware.  Loading after robot init
        # causes CAN timeouts on idle motor chains.
        print(f"\nLoading model from {ckpt_path}...")
        self._bridge.load(ckpt_path, **(bridge_kwargs or {}))
        print("Model loaded.\n")

        # Now initialize cameras, proprio threads, and robots (inherited).
        # _wants_leaders() (overridden below) tells RaidenPolicyServer to also
        # bring up the leader arms when intervening.  NOTE: with leaders enabled,
        # the inherited __init__ home move also drives the leaders to home — keep
        # hands OFF the leaders during startup.
        super().__init__(**kwargs)

        # rd serve wires the footpedal as a hard e-stop (attach_footpedal in the
        # base __init__).  The infer loop must NOT inherit that: its only estops
        # are _check_joint_delta (policy mode) + Ctrl+C, and a footpedal press
        # must never kill a handback.  Detach it.
        try:
            if self._robot._footpedal is not None:
                self._robot._footpedal.close()
                self._robot._footpedal = None
        except Exception:
            pass

        # Home the arms at startup.  The base server does NOT home in __init__
        # (only its async reset(), which the local loop never calls), so drive
        # the followers — and the leaders, when present — to home here so the
        # rollout starts from a known pose.  With leaders enabled this also puts
        # them at home so shadow mode has no startup jump.  Keep hands OFF.
        print("Homing arms... (keep hands clear)")
        self._robot.move_to_home_positions(simultaneous=True)

        # Put leaders into their HG-DAgger startup mode AFTER the home move.
        if self._intervene:
            if self._leader_track:
                # Shadow mode: keep the leaders PD-held (the inherited home move
                # left non-zero gains in the command path).  The input thread will
                # immediately drive them onto the followers — both start at home,
                # so there is no startup jump.  Do NOT zero_torque here or the
                # leaders would go limp until the first shadow command.
                print(
                    "HG-DAgger: leaders SHADOW the followers (under power). Press a "
                    "leader trigger to take over (instant); press again to hand "
                    "back. Keep hands clear until you take over.\n"
                )
            else:
                # Static mode: leaders truly passive until takeover, then synced.
                # Must use zero_torque_mode (not bare update_kp_kd): the home move
                # latched non-zero gains into the command path and nothing commands
                # the leader again in this loop, so update_kp_kd(0, 0) alone would
                # never reach the motors (leader stays PD-locked at home).
                if self._robot.leader_l is not None:
                    self._robot.leader_l.zero_torque_mode()
                if self._robot.leader_r is not None:
                    self._robot.leader_r.zero_torque_mode()
                print(
                    "HG-DAgger: leaders passive. Press a leader trigger to toggle "
                    "operator takeover (syncs leaders ~1.5 s); press again to hand "
                    "back to the policy.\n"
                )

    # ------------------------------------------------------------------
    # HG-DAgger hooks
    # ------------------------------------------------------------------

    def _wants_leaders(self) -> bool:
        return self._intervene

    def _dagger_input_loop(self) -> None:
        """Daemon thread: owns ALL leader CAN I/O + button toggle detection.

        ``YAMLeaderRobot.get_info()`` does a direct CAN read plus a ~10 ms sleep,
        so it must never run in the 30 Hz main loop.  This is the SOLE owner of
        the leader bus (reads AND shadow ``command_joint_pos`` writes) — the main
        loop only ever reads a snapshot under the lock.  Concurrent read/write on
        the leader bus from two threads throws mid-operation, which is why the
        teardown joins this thread before the home move.
        """
        while not self._dagger_stop.is_set():
            q7_l = q7_r = None
            btn_l = btn_r = 0.0
            btm_l = btm_r = 0.0
            if self._robot.leader_l is not None:
                try:
                    q7_l, io_l = self._robot.leader_l.get_info()
                    btn_l = float(io_l[0]) if len(io_l) > 0 else 0.0
                    btm_l = float(io_l[1]) if len(io_l) > 1 else 0.0
                except Exception:
                    pass
            if self._robot.leader_r is not None:
                try:
                    q7_r, io_r = self._robot.leader_r.get_info()
                    btn_r = float(io_r[0]) if len(io_r) > 0 else 0.0
                    btm_r = float(io_r[1]) if len(io_r) > 1 else 0.0
                except Exception:
                    pass

            # Rising edge on either TOP trigger toggles the mode (0.3 s cooldown).
            rising = (btn_r > 0.5 and self._last_btn["r"] < 0.5) or (
                btn_l > 0.5 and self._last_btn["l"] < 0.5
            )
            self._last_btn["r"] = btn_r
            self._last_btn["l"] = btn_l

            now = time.monotonic()

            # Rising edge on either BOTTOM button ends the current episode
            # (consumed by run_session; ignored by single-rollout run()).
            end_rising = (btm_r > 0.5 and self._last_btn_bottom["r"] < 0.5) or (
                btm_l > 0.5 and self._last_btn_bottom["l"] < 0.5
            )
            self._last_btn_bottom["r"] = btm_r
            self._last_btn_bottom["l"] = btm_l
            if end_rising and (now - self._last_end_toggle_t) > 0.3:
                self._last_end_toggle_t = now
                self._end_episode_evt.set()
                print("\n  [HG-DAgger] end-episode requested")

            with self._mode_lock:
                if q7_l is not None:
                    self._leader_state["q7_l"] = q7_l
                if q7_r is not None:
                    self._leader_state["q7_r"] = q7_r
                if rising and (now - self._last_toggle_t) > 0.3:
                    self._last_toggle_t = now
                    self._mode = "teleop" if self._mode == "policy" else "policy"
                    print(f"\n  [HG-DAgger] -> {self._mode.upper()} mode")
                mode_now = self._mode
                tgt_l = self._shadow_target["l"]
                tgt_r = self._shadow_target["r"]

            # Shadow tracking: in POLICY mode, drive each leader onto the latest
            # MEASURED follower pose so a later takeover is instant + seam-free.
            # Commanding here (not the 30 Hz main loop) keeps all leader CAN I/O
            # on this one thread.  Gated on policy: in teleop the leaders are
            # zero_torque so the operator can move them.
            if self._leader_track and mode_now == "policy":
                if tgt_l is not None and self._robot.leader_l is not None:
                    self._robot.leader_l.command_joint_pos(
                        np.asarray(tgt_l[:6], dtype=np.float64)
                    )
                if tgt_r is not None and self._robot.leader_r is not None:
                    self._robot.leader_r.command_joint_pos(
                        np.asarray(tgt_r[:6], dtype=np.float64)
                    )

            time.sleep(0.005)

    def _sync_leaders_to_followers(self) -> tuple:
        """Drive each leader to its follower's arm pose, then re-passivate.

        Static-mode takeover: the motorized leader is driven to the follower's
        actual pose (instead of leaving it pinned at home) so the operator gets
        full mechanical range around the follower configuration and the
        subsequent absolute mirror is seam-free.  The leaders move UNDER POWER —
        the operator must keep hands off during the sync.  Re-passivation goes
        through ``zero_torque_mode`` (command path) so the leader ends up
        genuinely free.

        Returns ``(f0_l, f0_r)`` — the follower poses used as the sync targets,
        which also become the follower baseline for the joint-delta takeover.
        """
        f0_l = (
            self._robot.follower_l.get_joint_pos()
            if self._robot.follower_l is not None
            else None
        )
        f0_r = (
            self._robot.follower_r.get_joint_pos()
            if self._robot.follower_r is not None
            else None
        )

        def _sync(leader, name: str, target: np.ndarray) -> None:
            # Restore PD so the leader can be driven, interpolate to the follower
            # arm pose, then go truly passive again via the command path.  NB:
            # TRI's smooth_move_joints takes keyword args (its 3rd positional is
            # start_joint_positions, not time_interval_s) — pass by keyword.
            leader.update_kp_kd(
                kp=self._robot.kp_gains[name], kd=self._robot.kd_gains[name]
            )
            smooth_move_joints(
                leader._robot,
                np.asarray(target[:6], dtype=np.float64),
                time_interval_s=1.5,
                steps=150,
            )
            leader.zero_torque_mode()

        threads = []
        if self._robot.leader_l is not None and f0_l is not None:
            threads.append(
                threading.Thread(
                    target=_sync, args=(self._robot.leader_l, "leader_l", f0_l)
                )
            )
        if self._robot.leader_r is not None and f0_r is not None:
            threads.append(
                threading.Thread(
                    target=_sync, args=(self._robot.leader_r, "leader_r", f0_r)
                )
            )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Let the input thread take at least one fresh post-sync leader reading
        # before the caller snapshots leader_q0, so leader_q0 is in the same frame
        # as subsequent live leader_now reads (otherwise the first step jumps).
        time.sleep(0.1)

        return (f0_l, f0_r)

    def _rearm_leader_shadow(self) -> None:
        """Restore leader PD gains so shadow tracking can drive them again.

        Called on handback (teleop->policy) when ``leader_track`` is on: the
        leaders were left in ``zero_torque_mode`` for the operator, so their gains
        must be restored before the input thread's ``command_joint_pos`` can exert
        torque (``update_kp_kd`` only sets ``_kp``; the next command stamps it into
        the command path).

        FIRST refresh ``_shadow_target`` to the followers' CURRENT measured pose,
        THEN restore the gains.  The shadow target goes stale during teleop (only
        published in policy mode); without this refresh the first shadow command
        after re-arming would yank the leader toward the pre-intervention pose — a
        visible jerk.  The followers are at the pose the human just left (≈ where
        the leaders are), so targeting that makes the first shadow command a no-op.
        """
        with self._mode_lock:
            if self._robot.follower_l is not None:
                self._shadow_target["l"] = self._robot.follower_l.get_joint_pos()[:6]
            if self._robot.follower_r is not None:
                self._shadow_target["r"] = self._robot.follower_r.get_joint_pos()[:6]
        if self._robot.leader_l is not None and "leader_l" in self._robot.kp_gains:
            self._robot.leader_l.update_kp_kd(
                kp=self._robot.kp_gains["leader_l"],
                kd=self._robot.kd_gains["leader_l"],
            )
        if self._robot.leader_r is not None and "leader_r" in self._robot.kp_gains:
            self._robot.leader_r.update_kp_kd(
                kp=self._robot.kp_gains["leader_r"],
                kd=self._robot.kd_gains["leader_r"],
            )

    def _make_dagger_recorder(self):
        """Build a DaggerRecorder writing into the standard ``rd record`` layout."""
        from raiden.recorder import DaggerRecorder, _next_recording_dir

        if self._dagger_task_name is None:
            self._resolve_dagger_task()
        task_name = self._dagger_task_name
        instruction = self._dagger_instruction_resolved

        task_dir = Path(self._data_dir) / "raw" / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        rec_dir = _next_recording_dir(task_dir)
        print(f"\n  Recording rollout -> {rec_dir}")
        return DaggerRecorder(
            cam_handles=self._cam_handles,
            robot_controller=self._robot,
            recording_dir=rec_dir,
            task_name=task_name,
            task_instruction=instruction,
            camera_fps=30,
        )

    def _resolve_dagger_task(self) -> None:
        """Resolve the dagger task name + instruction once (DB task or fzf picker)."""
        from raiden.recorder import select_task

        if self._dagger_task:
            task_name = self._dagger_task
            instruction = self._dagger_instruction or ""
            if not instruction:
                try:
                    from raiden.db import get_db

                    t = get_db().get_task_by_name(task_name)
                    if t:
                        instruction = t.get("instruction", "")
                except Exception:
                    pass
        else:
            task_name, instruction = select_task()
        self._dagger_task_name = task_name
        self._dagger_instruction_resolved = instruction

    def _register_dagger_demo(self, rec_dir: Path, verdict: Optional[str]) -> None:
        """Best-effort DB registration of a dagger rollout (tagged episode_kind).

        Silently no-ops when the task is absent from the DB — dagger tasks
        commonly have no DB rows by design.
        """
        try:
            from raiden.db import get_db

            db = get_db()
            task = db.get_task_by_name(getattr(self, "_dagger_task_name", "") or "")
            teachers = db.get_teachers()
            if task is None or not teachers:
                return
            demo_id = db.add_demonstration(
                teacher_id=teachers[-1]["id"],
                task_id=task["id"],
                raw_data_path=str(rec_dir),
                camera_config_id=0,
                calibration_result_id=None,
            )
            db.update_demonstration(
                demo_id,
                status=verdict or "pending",
                converted=False,
                episode_kind="dagger",
            )
        except Exception as exc:
            print(f"  (DB registration skipped: {exc})")

    # ------------------------------------------------------------------
    # Single rollout
    # ------------------------------------------------------------------

    def _would_estop(self, action: np.ndarray) -> bool:
        """Whether ``action`` would trip the base joint-delta safety check.

        Mirrors :meth:`RaidenPolicyServer._check_joint_delta`'s threshold (arm
        joints only, gripper excluded) so the leader-owning input thread can be
        quiesced BEFORE the base's ``emergency_stop()`` drives the leaders on the
        main thread.  A false positive at worst ends the session one step early.
        """
        pairs = []
        if self._robot.follower_l:
            q_l = self._read_proprio("follower_l_joint_pos")
            if q_l is not None:
                pairs.append((q_l, action[:DOF]))
        if self._robot.follower_r:
            q_r = self._read_proprio("follower_r_joint_pos")
            if q_r is not None:
                pairs.append((q_r, action[DOF : DOF * 2]))
        for current, commanded in pairs:
            if float(np.abs(commanded[:6] - current[:6]).max()) > self._max_joint_delta:
                return True
        return False

    def _stop_input_thread(self) -> None:
        """Stop + join the leader-CAN input thread so nothing else touches the
        leader bus.  Idempotent (a no-op once already stopped)."""
        self._dagger_stop.set()
        t = getattr(self, "_dagger_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=1.0)

    def run(self) -> None:
        """Run the closed-loop inference loop.  Blocks until Ctrl+C.

        In HG-DAgger mode (``intervene=True``) the policy runs autonomously until
        the operator presses a leader trigger to take over, then presses again to
        hand control back (a fresh ``bridge.reset()`` re-infer).  With shadow
        tracking on (the default) the leaders continuously mirror the followers
        during policy mode, so takeover is INSTANT and seam-free; with it off the
        leaders sync to the followers (~1.5 s) at takeover.  Either way the
        operator then drives via a joint-delta off the takeover baseline.  The
        whole rollout is recorded with a per-frame ``control_source`` flag
        (0=policy, 1=human) so ``rd convert`` / ``rd shardify`` can weight it.
        """
        dt = 1.0 / self._action_hz

        self._maybe_reset_pose()
        self._bridge.reset()

        recorder = self._make_dagger_recorder() if self._record else None
        if recorder is not None:
            # Quiesce the ZED grab loops around enable_recording: a concurrent
            # grab() + enable_recording() deadlocks the ZED SDK.  super().__init__
            # has already started the grab threads, so start() MUST be guarded
            # here just like run_session does.
            self._pause_grabbing()
            try:
                recorder.start()
            finally:
                self._resume_grabbing()
        if self._intervene:
            self._dagger_thread = threading.Thread(
                target=self._dagger_input_loop, name="dagger-input", daemon=True
            )
            self._dagger_thread.start()

        print(f"Starting inference at {self._action_hz} Hz")
        print("Press Ctrl+C to stop.\n")

        verdict: Optional[str] = None
        try:
            self._episode_loop(recorder, dt, end_on_button=False)
        finally:
            self._teardown_run(recorder, verdict)

    def _episode_loop(self, recorder, dt: float, end_on_button: bool = False) -> str:
        """Run the per-step control loop for ONE episode.

        Returns ``"ended"`` when the operator presses the end-episode (BOTTOM)
        button (only when ``end_on_button``) or ``"interrupt"`` on Ctrl+C.  Shared
        by :meth:`run` and :meth:`run_session` so autonomous + takeover behavior
        is identical in both.
        """
        import os

        step = 0
        prev_mode = "policy"
        # (follower_q0_l, follower_q0_r, leader_q0_l, leader_q0_r) at takeover edge.
        takeover_q0: tuple = (None, None, None, None)
        grip_engaged = {"l": False, "r": False}  # clutch latch, reset each takeover
        prof_path = os.environ.get("RAIDEN_PROFILE")  # path or "1" → per-step timing
        prof_rows = [] if prof_path else None
        try:
            while True:
                if end_on_button and self._end_episode_evt.is_set():
                    self._episode_steps = step
                    return "ended"
                t_step = time.perf_counter()

                # Build observation from live sensors (synthetic step clock).
                obs = self._make_obs(timestamp=step * dt)
                # Expose the last COMMANDED joints so an eef bridge can rebase the
                # next action chunk on the commanded (open-loop) pose rather than
                # the measured (lagging) one — avoids the gravity-droop ratchet.
                if self._last_joint_cmd is not None:
                    _ljc = np.asarray(self._last_joint_cmd, dtype=np.float32)
                    if _ljc.shape[0] >= DOF * 2:
                        obs.proprios["follower_l_joint_cmd"] = _ljc[:DOF]
                        obs.proprios["follower_r_joint_cmd"] = _ljc[DOF : DOF * 2]
                _t_obs = time.perf_counter()
                _t_pred0 = _t_pred1 = None

                # Snapshot control mode + leader poses (filled by the input thread).
                if self._intervene:
                    with self._mode_lock:
                        mode = self._mode
                        leader_now = dict(self._leader_state)
                else:
                    mode, leader_now = "policy", {}

                # Handle mode transitions on the edge.
                if mode != prev_mode:
                    if mode == "teleop":
                        if self._leader_track:
                            # Shadow tracking already kept the leaders ON the
                            # followers, so takeover is INSTANT — just go passive
                            # so the operator can move them, and capture the
                            # baseline from the current (already-synced) read.
                            if self._robot.leader_l is not None:
                                self._robot.leader_l.zero_torque_mode()
                            if self._robot.leader_r is not None:
                                self._robot.leader_r.zero_torque_mode()
                            f0_l = (
                                self._robot.follower_l.get_joint_pos()
                                if self._robot.follower_l
                                else None
                            )
                            f0_r = (
                                self._robot.follower_r.get_joint_pos()
                                if self._robot.follower_r
                                else None
                            )
                            print(
                                "\n  [HG-DAgger] you have control (instant takeover).\n"
                            )
                        else:
                            # Static mode: sync the motorized leaders to the
                            # followers (blocking ~1.5 s). Hands OFF — leaders
                            # move under power during the sync.
                            print(
                                "\n  [HG-DAgger] syncing leaders to followers — "
                                "hands OFF the leaders..."
                            )
                            f0_l, f0_r = self._sync_leaders_to_followers()
                            print("  [HG-DAgger] leaders synced — you have control.\n")
                        # Capture the takeover baseline from the SAME read used for
                        # this step's leader_now so the first joint-delta is exactly
                        # zero (seam-free by construction).
                        with self._mode_lock:
                            leader_now = dict(self._leader_state)
                        takeover_q0 = (
                            f0_l,
                            f0_r,
                            leader_now.get("q7_l"),
                            leader_now.get("q7_r"),
                        )
                        # Fresh gripper clutch for this takeover.
                        grip_engaged = {"l": False, "r": False}
                    else:
                        # Handback: the policy resumes from the EXACT pose the
                        # human left.  bridge.reset() discards the pre-intervention
                        # chunk so the next predict() re-infers on the CURRENT obs.
                        self._bridge.reset()
                        if self._leader_track:
                            # Re-arm shadow: restore leader PD so the input thread
                            # can drive them again.  Followers are at the pose the
                            # human left (≈ where the leaders are), so the first
                            # shadow command is a tiny move, not a snap.
                            self._rearm_leader_shadow()
                    prev_mode = mode

                if mode == "teleop":
                    # Operator in control — joint-delta off the synced baseline
                    # (arm becomes absolute mirroring; gripper uses a bumpless
                    # clutch).  _check_joint_delta is SKIPPED in teleop: the human
                    # is the safety authority, and running it would
                    # emergency_stop()->os._exit on a large deliberate delta.
                    action_14d = joint_delta_takeover(
                        leader_now, takeover_q0, dof=DOF, grip_engaged=grip_engaged
                    )
                else:
                    # Policy in control — observe and act from the current state.
                    _t_pred0 = time.perf_counter()
                    action = self._bridge.predict(obs)
                    if self._action_type == "ee_pose":
                        action_14d = self._ee_pose_to_joint_cmd(
                            action, self._last_joint_cmd
                        )
                    else:
                        action_14d = np.asarray(action, dtype=np.float32)
                    # A policy-runaway breach here calls emergency_stop(), which
                    # drives the LEADER bus from this (main) thread.  In intervene
                    # mode the dagger input thread is the sole leader-bus owner, so
                    # join it FIRST (only when an estop is actually imminent) or the
                    # two race on the bus during the safety home move.
                    if self._intervene and self._would_estop(action_14d):
                        self._stop_input_thread()
                    self._check_joint_delta(action_14d)
                    _t_pred1 = time.perf_counter()

                self._last_joint_cmd = action_14d

                # Snapshot measured joints BEFORE commanding (the gap the PD must
                # close), for delta logging + shadow publish.
                meas_l = (
                    self._robot.follower_l.get_joint_pos()
                    if self._robot.follower_l
                    else None
                )
                meas_r = (
                    self._robot.follower_r.get_joint_pos()
                    if self._robot.follower_r
                    else None
                )

                # Publish the follower's MEASURED pose for the input thread to
                # shadow the leaders onto (policy mode only).  We track measured —
                # not the raw policy command — because the follower's damped PD has
                # already smoothed the command; shadowing the raw command makes the
                # differently-geared leader jitter while the follower stays smooth.
                if self._intervene and self._leader_track and mode == "policy":
                    with self._mode_lock:
                        if meas_l is not None:
                            self._shadow_target["l"] = meas_l[:6]
                        if meas_r is not None:
                            self._shadow_target["r"] = meas_r[:6]

                # Command motors: action[:7] → left follower, action[7:14] → right.
                _t_cmd0 = time.perf_counter()
                if self._robot.follower_l:
                    self._robot.follower_l.command_joint_pos(action_14d[:DOF])
                if self._robot.follower_r:
                    self._robot.follower_r.command_joint_pos(action_14d[DOF : DOF * 2])
                _t_cmd1 = time.perf_counter()

                # Record the rollout frame (observed obs + control_source flag +
                # commanded joints).  Camera SDK clock so robot_ts aligns with the
                # SVO2 frame timestamps.
                if recorder is not None:
                    recorder.add_frame(
                        self._get_zed_current_time_ns(),
                        self._robot.get_all_observations(),
                        1 if mode == "teleop" else 0,
                        action_14d,
                    )

                step += 1

                # commanded-vs-measured delta per arm (6 arm joints only). A
                # growing/plateauing delta near the torque ceiling is a
                # contact / no-progress signal.
                delta_l = (
                    np.abs(action_14d[:6] - meas_l[:6]) if meas_l is not None else None
                )
                delta_r = (
                    np.abs(action_14d[DOF : DOF + 6] - meas_r[:6])
                    if meas_r is not None
                    else None
                )

                elapsed = time.perf_counter() - t_step
                hz = 1.0 / max(elapsed, 1e-6)
                tag = "HUMAN" if mode == "teleop" else "policy"
                l_act = np.array2string(
                    action_14d[:DOF], precision=3, suppress_small=True
                )
                r_act = np.array2string(
                    action_14d[DOF : DOF * 2], precision=3, suppress_small=True
                )
                if step % 5 == 1:
                    dl = f"max={delta_l.max():.3f}" if delta_l is not None else "n/a"
                    dr = f"max={delta_r.max():.3f}" if delta_r is not None else "n/a"
                    print(
                        f"step={step:5d}  hz={hz:.1f}  [{tag}]  "
                        f"l_delta[{dl}]  r_delta[{dr}]"
                    )
                else:
                    print(f"step={step:5d}  hz={hz:.1f}  [{tag}]  l={l_act}  r={r_act}")

                # Sleep to maintain target Hz.
                elapsed = time.perf_counter() - t_step
                remaining = dt - elapsed
                if prof_rows is not None:
                    prof_rows.append(
                        {
                            "step": step,
                            "mode": mode,
                            "obs_ms": round((_t_obs - t_step) * 1e3, 3),
                            "predict_ms": (
                                round((_t_pred1 - _t_pred0) * 1e3, 3)
                                if _t_pred0 is not None and _t_pred1 is not None
                                else None
                            ),
                            "cmd_ms": round((_t_cmd1 - _t_cmd0) * 1e3, 3),
                            "work_ms": round(elapsed * 1e3, 3),
                            "cmd": [round(float(x), 4) for x in action_14d],
                        }
                    )
                if remaining > 0:
                    time.sleep(remaining)

        except KeyboardInterrupt:
            print("\n\nStopping inference...")
            self._episode_steps = step
            return "interrupt"
        finally:
            if prof_rows:
                import json

                _out = (
                    "policy_profile.jsonl"
                    if prof_path in ("1", "true", "True")
                    else prof_path
                )
                _n = getattr(self, "_prof_ep_idx", 0)
                self._prof_ep_idx = _n + 1
                if _n:
                    _root, _ext = os.path.splitext(_out)
                    _out = f"{_root}.ep{_n}{_ext}"
                try:
                    with open(_out, "w") as _f:
                        for _r in prof_rows:
                            _f.write(json.dumps(_r) + "\n")
                    print(f"\n[profile] wrote {len(prof_rows)} steps -> {_out}")
                except Exception as _exc:
                    print(f"[profile] dump failed: {_exc}")

    def _teardown_run(self, recorder, verdict) -> None:
        """Home + de-energize + save — teardown for the single-rollout run().

        ORDER IS LOAD-BEARING.  (1) Stop + JOIN the input thread first: it owns
        all leader CAN I/O and must be quiet before the home move drives the
        leaders (concurrent read/write on the leader bus throws mid-home).  (2)
        Drive home while CAN is healthy (restoring leader PD first so zero_torque
        leaders don't sag).  (3) close() to de-energize FROM home.  (4) ONLY THEN
        the heavy recorder.stop() (SVO2 flush + np.savez of thousands of frames)
        — deferred because it hogs the GIL for seconds and would starve the
        100/250 Hz motor threads past the ~400 ms DM watchdog and drop every CAN
        bus.  Doing it after close() means there are no motor threads left to
        starve.  (5) DB registration.  (6) bridge.reset().  (7) close cameras.
        """
        step = self._episode_steps

        # (1) quiet the leader bus before homing.  The join MUST complete: a live
        # input thread owns the leader bus and the home move below drives it too.
        # 2 s is ~200 input-loop iterations of margin (it re-checks the stop flag
        # every ~10 ms), so a surviving thread means a genuinely wedged CAN read.
        self._dagger_stop.set()
        if self._dagger_thread is not None:
            self._dagger_thread.join(timeout=2.0)
            if self._dagger_thread.is_alive():
                print(
                    "  Warning: leader input thread did not stop within 2 s "
                    "(wedged CAN read?). The home move below may contend with it "
                    "on the leader bus — keep clear of the arms."
                )

        # (2) home while CAN is healthy.
        if step > 0:
            print("Returning to home... (keep hands clear of the arms)")
            try:
                if self._intervene:
                    # Leaders may be in zero_torque_mode (static, or post-takeover)
                    # — restore PD first or move_to_home commands them with zero
                    # gains and they sag.  Guarded so a CAN error still falls
                    # through to close() rather than crashing.
                    self._robot.disable_gravity_compensation()
                self._robot.move_to_home_positions(simultaneous=True)
            except Exception as exc:
                print(f"  (home move error: {exc})")

        # (3) stop capture loops + de-energize FROM home.  Setting _running=False
        # also stops the ZED grab loops, so the subsequent recorder.stop()
        # disable_recording cannot deadlock against a live grab().
        self._running = False
        time.sleep(0.1)
        try:
            self._robot.close()
        except Exception as exc:
            print(f"  (robot close error: {exc})")

        # (4) arms are safe — persist the rollout (heavy) and clean up.
        try:
            if recorder is not None and recorder.is_recording:
                rec_dir = recorder.stop(complete=True, verdict=verdict)
                if self._intervene:
                    self._register_dagger_demo(rec_dir, verdict)
        except Exception as exc:
            print(f"  (recording shutdown error: {exc})")
        # (6) bridge reset.
        try:
            self._bridge.reset()
        except Exception:
            pass
        # (7) close cameras.
        for handle in self._cam_handles.values():
            try:
                if handle.get("type") == "zed":
                    handle["camera"].close()
                elif "pipeline" in handle:
                    handle["pipeline"].stop()
            except Exception:
                pass
        print("Done.")

    # ------------------------------------------------------------------
    # Looped HG-DAgger collection (reuses model/cameras/robot)
    # ------------------------------------------------------------------

    def run_session(self) -> None:
        """Looped HG-DAgger collection — episode after episode in one process.

        The model, cameras, and robot load ONCE; every episode reuses them.  Per
        episode: a start gate (reset the scene) -> the policy runs autonomously
        (leader TOP button to take over, again to hand back) -> leader BOTTOM
        button to end -> keyboard ``s``/``d`` to save or discard -> next episode.
        ``q`` at the gate, or Ctrl+C mid-episode, ends the session.
        """
        dt = 1.0 / self._action_hz
        self._maybe_reset_pose()
        self._resolve_dagger_task()

        print(f"\n=== HG-DAgger session — task '{self._dagger_task_name}' ===")
        episode_idx = 0
        try:
            while True:
                if not self._wait_for_episode_start(episode_idx):
                    break

                self._bridge.reset()
                self._last_joint_cmd = None
                self._mode = "policy"
                self._end_episode_evt.clear()
                recorder = self._make_dagger_recorder()
                # Quiesce the grab threads around the SVO2 enable toggle (they run
                # continuously between episodes; concurrent grab()+enable_recording
                # deadlocks the ZED SDK).
                self._pause_grabbing()
                try:
                    recorder.start()
                finally:
                    self._resume_grabbing()

                # The leader-input thread runs ONLY during the episode.
                if self._intervene:
                    self._dagger_stop.clear()
                    self._dagger_thread = threading.Thread(
                        target=self._dagger_input_loop, name="dagger-input", daemon=True
                    )
                    self._dagger_thread.start()

                print(
                    f"\nEpisode running at {self._action_hz} Hz   "
                    "[TOP = take over / hand back   BOTTOM = end episode]\n"
                )
                reason = self._episode_loop(recorder, dt, end_on_button=True)

                # Quiet the leader bus (stop shadow-driving) before homing so the
                # input thread doesn't fight the home move on the same bus.
                self._dagger_stop.set()
                if self._dagger_thread is not None:
                    self._dagger_thread.join(timeout=1.0)
                    self._dagger_thread = None

                # Home + DISCONNECT all four CAN buses BEFORE finalizing the
                # recording — mirrors DemonstrationRecorder.stop_recording, which
                # closes motors THEN saves.  With CAN already closed the heavy
                # SVO2 flush + np.savez can't starve the motor threads past the DM
                # watchdog and drop the bus.
                self._robot.return_to_home_and_disconnect()

                # Now safe to stop SVO2 + write robot_data.npz.  Pause grabbing
                # around stop() for the grab()-vs-disable_recording deadlock.
                self._pause_grabbing()
                try:
                    rec_dir = recorder.stop(complete=True, verdict=None)
                finally:
                    self._resume_grabbing()

                if reason == "interrupt":
                    # Ctrl+C ends the whole session; keep the in-progress episode
                    # and skip the re-init.
                    self._register_dagger_demo(rec_dir, None)
                    print(f"  Saved -> {rec_dir}")
                    break
                if self._wait_for_save_or_discard():
                    self._register_dagger_demo(rec_dir, None)
                    print(f"  Saved -> {rec_dir}")
                else:
                    shutil.rmtree(rec_dir, ignore_errors=True)
                    print("  Discarded.")

                # Re-open CAN + re-home + restore the leader mode for the next
                # episode (mirrors rd record between episodes).
                self._reinit_robot_for_next_episode()
                episode_idx += 1
        finally:
            self._shutdown_session()

    def _reinit_robot_for_next_episode(self) -> None:
        """Re-open CAN, re-home, and restore the dagger leader mode for next episode.

        ``return_to_home_and_disconnect()`` closed every CAN bus and nulled the
        arm handles.  ``initialize_robots()`` recreates each arm from scratch and
        re-opens its bus; drive them home, then put the leaders back into shadow
        (leader_track) or passive (static) mode — the state the constructor left.
        """
        print("\nRe-initializing arms for the next episode... (hands clear)")
        self._robot.check_can_interfaces()
        self._robot.initialize_robots()
        self._robot.move_to_home_positions(simultaneous=True)
        if self._intervene:
            if self._leader_track:
                self._rearm_leader_shadow()
            else:
                if self._robot.leader_l is not None:
                    self._robot.leader_l.zero_torque_mode()
                if self._robot.leader_r is not None:
                    self._robot.leader_r.zero_torque_mode()

    def _pause_grabbing(self) -> None:
        """Quiesce the camera grab threads before toggling SVO2 recording."""
        evt = getattr(self, "_grab_paused", None)
        if evt is not None:
            evt.set()
            time.sleep(0.15)  # let any in-flight grab() (~33 ms at 30 fps) finish

    def _resume_grabbing(self) -> None:
        evt = getattr(self, "_grab_paused", None)
        if evt is not None:
            evt.clear()

    def _shutdown_session(self) -> None:
        """Stop the input thread, de-energize the arms, close cameras."""
        self._dagger_stop.set()
        if self._dagger_thread is not None:
            self._dagger_thread.join(timeout=1.0)
        self._running = False
        time.sleep(0.1)
        try:
            self._robot.close()
        except Exception as exc:
            print(f"  (robot close error: {exc})")
        try:
            self._bridge.reset()
        except Exception:
            pass
        for handle in self._cam_handles.values():
            try:
                if handle.get("type") == "zed":
                    handle["camera"].close()
                elif "pipeline" in handle:
                    handle["pipeline"].stop()
            except Exception:
                pass
        print("Session done.")

    def _wait_for_episode_start(self, episode_idx: int) -> bool:
        """Gate before each episode.  Enter -> start, q -> quit the session."""
        import select
        import sys
        import termios
        import tty

        print("\n" + "=" * 60)
        print(
            f"  Ready for episode #{episode_idx}.  Reset the scene, then:\n"
            "    Enter -> start    q -> quit session"
        )
        print("=" * 60)
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch.lower() == "q":
                        return False
                    if ch in ("\r", "\n"):
                        return True
                time.sleep(0.05)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _wait_for_save_or_discard(self) -> bool:
        """After an episode ends: s -> save, d -> discard (keyboard only)."""
        import select
        import sys
        import termios
        import tty

        print("\n" + "-" * 60)
        print("  Keep this episode?   s -> save    d -> discard")
        print("-" * 60)
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1).lower()
                    if ch == "s":
                        return True
                    if ch == "d":
                        return False
                time.sleep(0.05)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _maybe_reset_pose(self) -> None:
        """Optionally move the followers to a saved pose before the rollout.

        Opt-in via ``--reset-pose``; moves to the first frame of a saved
        ``robot_data.npz`` (per ABC, resetting is discouraged — prefer
        intervene-and-continue — hence off by default).
        """
        if not self._reset_pose:
            return
        try:
            from raiden.robot.replay import _load_raw_joints

            joints_l, joints_r, _ = _load_raw_joints(Path(self._reset_pose))
            print(f"\nResetting to first-frame pose from {self._reset_pose}...")
            threads = []
            if self._robot.follower_l is not None and joints_l is not None:
                threads.append(
                    threading.Thread(
                        target=smooth_move_joints,
                        args=(self._robot.follower_l, joints_l[0]),
                    )
                )
            if self._robot.follower_r is not None and joints_r is not None:
                threads.append(
                    threading.Thread(
                        target=smooth_move_joints,
                        args=(self._robot.follower_r, joints_r[0]),
                    )
                )
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        except Exception as exc:
            print(f"  (reset-pose skipped: {exc})")


# ---------------------------------------------------------------------------
# Dynamic bridge loading (used by ``rd infer``)
# ---------------------------------------------------------------------------


def load_bridge(bridge_spec: str, **kwargs) -> ModelBridge:
    """Import a ModelBridge class from a ``module.path:ClassName`` string.

    Extra *kwargs* are forwarded to the bridge constructor.

    Examples::

        load_bridge("your_model.deployment.yam_bridge:MyBridge")
        load_bridge("your_model.deployment.yam_bridge:MyBridge", chunk_size=4)
    """
    if ":" not in bridge_spec:
        raise ValueError(
            f"Bridge spec must be 'module.path:ClassName', got '{bridge_spec}'"
        )
    module_path, class_name = bridge_spec.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Cannot import bridge module '{module_path}'. "
            "Make sure it is installed or on PYTHONPATH."
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Module '{module_path}' has no attribute '{class_name}'")
    bridge = cls(**kwargs)
    if not isinstance(bridge, ModelBridge):
        raise TypeError(f"'{bridge_spec}' is not a ModelBridge subclass")
    return bridge
