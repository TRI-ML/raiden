"""Demonstration recorder – cameras at 30 fps, robot joints at ~100 Hz.

Thread layout during a recording session::

    teleop-right     : follower_r mirrors leader_r at 100 Hz  (background, per episode)
    teleop-left      : follower_l mirrors leader_l at 100 Hz  (background, per episode)
    camera-<name>    : camera.grab() loop per camera          (active while recording)
    robot-recorder   : reads all joint observations at 100 Hz (active while recording)

Cameras are opened once at session start and stay open across episodes.
Teleop threads are started/stopped per episode.
Camera and robot threads are started/stopped per recording episode.

Output layout::

    data/raw/<task_name>/<episode_idx>/
        metadata.json
        robot_data.npz
        cameras/
            scene_camera.svo2
            left_wrist_camera.svo2
            right_wrist_camera.svo2
            ...

After conversion (``rd convert``) each camera directory is expanded into
PNG frames and depth maps.
"""

import json
import re
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import numpy as np

from raiden._config import CALIBRATION_FILE, CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.cameras import Camera
from raiden.control import TeleopInterface
from raiden.db.database import get_db
from raiden.robot.controller import RobotController
from raiden.utils import fzf_select

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class RecordingMetadata:
    task_name: str
    task_instruction: str
    timestamp: str
    duration_s: float
    robot_frames: int
    robot_hz: float
    cameras: List[str]
    camera_fps: int
    control: str = "leader"
    complete: bool = False
    converted: bool = False


# ---------------------------------------------------------------------------
# DemonstrationRecorder
# ---------------------------------------------------------------------------


class DemonstrationRecorder:
    """Manages one recording episode.

    Cameras are opened before this object is created and remain open across
    episodes (opening/closing ZED cameras is slow).  start_recording() /
    stop_recording() toggle the SVO2 writers and the robot data thread.

    stop_recording() does NOT close cameras or shut down robots — the caller
    (run_recording) is responsible for that at session end.
    """

    def __init__(
        self,
        cameras: List[Camera],
        robot_controller: RobotController,
        recording_dir: Path,
        task_name: str,
        task_instruction: str,
        interface: TeleopInterface,
    ):
        self.cameras = cameras
        self.robot_controller = robot_controller
        self.recording_dir = recording_dir
        self.task_name = task_name
        self.task_instruction = task_instruction
        self.interface = interface

        self.cameras_dir = recording_dir / "cameras"

        self.is_recording = False
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

        # Robot data accumulated during one episode
        self._robot_frames: List[Dict] = []
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        if self.is_recording:
            return

        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.cameras_dir.mkdir(parents=True, exist_ok=True)

        self.is_recording = True
        self._start_time = time.monotonic()
        self._robot_frames = []
        self._stop_event.clear()
        self._threads = []
        self._camera_start_times_ns: Dict[str, int] = {}

        # Start all cameras in parallel so they begin recording simultaneously.
        # Sequential starts would introduce a per-camera startup offset (e.g.
        # ~50 frames for RealSense due to pipeline restart) that misaligns ZED
        # and bag frames.
        def _start_one(camera) -> None:
            path = self.cameras_dir / f"{camera.name}.{camera.recording_extension}"
            t0 = time.time_ns()
            camera.start_recording(path)
            self._camera_start_times_ns[camera.name] = t0

        start_threads = [
            threading.Thread(target=_start_one, args=(cam,), daemon=True)
            for cam in self.cameras
        ]
        for t in start_threads:
            t.start()
        for t in start_threads:
            t.join()

        # One camera grab thread per camera (naturally rate-limited by SDK)
        for camera in self.cameras:
            t = threading.Thread(
                target=self._camera_loop,
                args=(camera, self._stop_event),
                name=f"camera-{camera.name}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        # Robot data recording thread — uses the first camera as a shared clock
        # reference so that robot timestamps are on the same clock as camera
        # frame timestamps (enabling direct interpolation at conversion time).
        t = threading.Thread(
            target=self._robot_loop,
            args=(self._stop_event, self.cameras[0]),
            name="robot-recorder",
            daemon=True,
        )
        t.start()
        self._threads.append(t)

        print("\n" + "!" * 60)
        print("  RECORDING STARTED")
        print("!" * 60)
        print("  Press the button again to stop recording\n")

    def stop_recording(self, complete: bool = True) -> Path:
        """Stop the current recording episode and persist data.

        Shuts down the robot controller (returns home + closes motor connections)
        and stops camera recording, but does NOT close cameras — they stay open
        so the next episode can start without re-initialising the camera SDK.
        The caller is responsible for calling camera.close() at session end.

        Args:
            complete: Set to True when the episode was cleanly stopped by the
                      user.  Set to False on crash / Ctrl-C so the directory
                      can be detected as incomplete and overridden next run.
        """
        if not self.is_recording:
            return self.recording_dir

        self.is_recording = False
        self._stop_event.set()
        duration = time.monotonic() - self._start_time

        # Shut down robot (return home + close motor connections).
        # Motors must be closed between episodes — keeping them alive idle
        # causes DM4310 CAN watchdog errors due to lack of regular commands.
        self.robot_controller.shutdown()

        # Wait for threads (camera grab threads will exit within one frame period)
        for t in self._threads:
            t.join(timeout=3.0)

        # Finalise SVO2 / bag files but keep cameras open — run in parallel so
        # all cameras stop at the same time (avoids trailing frames on one camera).
        stop_threads = [
            threading.Thread(target=camera.stop_recording, daemon=True)
            for camera in self.cameras
        ]
        for t in stop_threads:
            t.start()
        for t in stop_threads:
            t.join()

        print("\n" + "!" * 60)
        print("  RECORDING STOPPED")
        print(f"  Duration : {duration:.2f} s")
        print(f"  Robot frames : {len(self._robot_frames)}")
        print("!" * 60 + "\n")

        # Persist robot data
        self._save_robot_data()
        self._save_metadata(duration, complete=complete)

        return self.recording_dir

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _camera_loop(self, camera: Camera, stop_event: threading.Event) -> None:
        """Grab loop – camera SDK rate-limits to its own FPS (30 Hz)."""
        while not stop_event.is_set():
            camera.grab()

    def _robot_loop(self, stop_event: threading.Event, ref_camera) -> None:
        """Read joint observations at ~100 Hz and buffer them.

        Timestamps are recorded via ``ref_camera.get_current_timestamp_ns()``.

        - ZED cameras override this to return the ZED SDK hardware clock, which
          is on the same clock as the frame timestamps in the SVO2 file.
          Direct interpolation at conversion time requires no correction.
        - RealSense cameras fall back to ``time.time_ns()`` (system wall clock),
          while frame timestamps use the RealSense hardware clock.  The offset
          between the two clocks is measured once per session and stored in
          ``metadata.json`` as ``realsense_clock_offsets``, then applied at
          conversion time.
        """
        target_dt = 0.01  # 100 Hz
        while not stop_event.is_set():
            loop_start = time.monotonic()

            try:
                ts_ns = ref_camera.get_current_timestamp_ns()
                obs = self.robot_controller.get_all_observations()
                cmd = self.robot_controller.get_last_commanded_positions()
                self._robot_frames.append({"t": ts_ns, "obs": obs, "cmd": cmd})
            except Exception as e:
                print(f"  Warning: robot observation failed: {e}")

            elapsed = time.monotonic() - loop_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_robot_data(self) -> None:
        """Flatten the robot frame list into per-key numpy arrays and save .npz."""
        output_file = self.recording_dir / "robot_data.npz"
        n = len(self._robot_frames)

        if n == 0:
            print("  Warning: no robot frames recorded")
            return

        data: Dict[str, np.ndarray] = {}
        data["timestamps"] = np.array(
            [f["t"] for f in self._robot_frames], dtype=np.int64
        )

        # Collect all robot names present in the first frame
        robot_names = list(self._robot_frames[0]["obs"].keys())

        for robot_name in robot_names:
            obs_keys = list(self._robot_frames[0]["obs"][robot_name].keys())
            for key in obs_keys:
                arr = np.stack([f["obs"][robot_name][key] for f in self._robot_frames])
                data[f"{robot_name}_{key}"] = arr

        # Build combined 7-DOF arrays for follower arms.
        # follower_<arm>_joint_pos_7d  — executed: arm joints (6) + gripper (1)
        # follower_<arm>_joint_cmd     — commanded: 7-DOF sent to command_joint_pos
        for arm_key in ("follower_r", "follower_l"):
            jp_key = f"{arm_key}_joint_pos"
            gp_key = f"{arm_key}_gripper_pos"
            if jp_key in data and gp_key in data:
                grip = data[gp_key].reshape(n, 1).astype(np.float32)
                data[f"{arm_key}_joint_pos_7d"] = np.concatenate(
                    [data[jp_key].astype(np.float32), grip], axis=1
                )

            # Build cmd array, falling back to actual joint pos for frames
            # where no command has been issued yet (e.g. the first few frames
            # before the first teleop command is processed).
            pos7d_key = f"{arm_key}_joint_pos_7d"
            raw_cmds = [f["cmd"].get(arm_key) for f in self._robot_frames]
            if jp_key in data:
                # This arm was active — cmd data is required.
                if not any(c is not None for c in raw_cmds):
                    raise RuntimeError(
                        f"No commanded positions recorded for {arm_key}. "
                        "The teleop loop never issued a command to this arm."
                    )
                fallback = None
                filled: list = []
                for i, c in enumerate(raw_cmds):
                    if c is not None:
                        fallback = c
                    if fallback is not None:
                        filled.append(fallback)
                    elif pos7d_key in data:
                        filled.append(data[pos7d_key][i])
                if len(filled) != n:
                    raise RuntimeError(
                        f"Could not build complete cmd array for {arm_key}: "
                        f"got {len(filled)}/{n} frames."
                    )
                data[f"{arm_key}_joint_cmd"] = np.stack(filled).astype(np.float32)

        np.savez_compressed(output_file, **data)
        print(f"  ✓ Robot data saved  ({n} frames) → {output_file}")

    def _save_metadata(self, duration: float, complete: bool = True) -> None:
        output_file = self.recording_dir / "metadata.json"
        n = len(self._robot_frames)
        hz = n / duration if duration > 0 else 0.0

        meta = RecordingMetadata(
            task_name=self.task_name,
            task_instruction=self.task_instruction,
            timestamp=datetime.now().isoformat(),
            duration_s=round(duration, 3),
            robot_frames=n,
            robot_hz=round(hz, 1),
            cameras=[c.name for c in self.cameras],
            camera_fps=30,
            control=self.interface.name,
            complete=complete,
            converted=False,
        )

        meta_dict = asdict(meta)

        if self._camera_start_times_ns:
            meta_dict["camera_start_times_ns"] = self._camera_start_times_ns

        rs_offsets = {
            cam.name: cam._clock_offset_ns
            for cam in self.cameras
            if getattr(cam, "_clock_offset_ns", None) is not None
        }
        if rs_offsets:
            meta_dict["realsense_clock_offsets"] = rs_offsets

        with open(output_file, "w") as f:
            json.dump(meta_dict, f, indent=2)

        print(f"  ✓ Metadata saved → {output_file}")


# ---------------------------------------------------------------------------
# Camera factory
# ---------------------------------------------------------------------------


def load_cameras_from_config(
    config_file: str = CAMERA_CONFIG,
) -> List[Camera]:
    """Create Camera instances for every entry in camera.json."""
    cfg = CameraConfig(config_file)
    names = cfg.list_camera_names()

    results: dict = {}
    errors: dict = {}

    def _open(name: str) -> None:
        try:
            cam = cfg.create_camera(name)
            cam.open()
            cam_type = cfg.get_camera_type(name)
            serial = cfg.get_serial_by_name(name)
            print(f"  ✓ Camera '{name}' opened ({cam_type}, serial: {serial})")
            results[name] = cam
        except Exception as exc:
            errors[name] = exc

    threads = [
        threading.Thread(target=_open, args=(name,), daemon=True) for name in names
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise RuntimeError(f"Camera initialization failed: {errors}")

    # Return cameras in config order.
    return [results[name] for name in names]


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


_TASK_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def validate_task_name(name: str) -> str | None:
    """Return an error message if *name* is invalid, otherwise None.

    Valid names contain only letters, digits, and underscores — no spaces.
    Examples: ``PickUpApple``, ``pick_up_apple``, ``task01``.
    """
    if not name:
        return "Task name cannot be empty."
    if not _TASK_NAME_RE.match(name):
        return "Task name must contain only letters, digits, and underscores (no spaces). E.g. PickUpApple or pick_up_apple."
    return None


def select_task() -> tuple[str, str]:
    """Use fzf to choose (or create) a task for recording."""
    db = get_db()
    tasks = db.get_tasks()

    _NEW = "<< Add new task >>"
    labels = {f"{t['name']}  ({t['instruction']})": t for t in reversed(tasks)}
    chosen = fzf_select(list(labels) + [_NEW], prompt="Record task> ")[0]

    if chosen == _NEW:
        name = input("  New task name: ").strip()
        err = validate_task_name(name)
        if err:
            print(f"Error: {err}")
            sys.exit(1)
        instruction = input("  Task instruction: ").strip()
        if not instruction:
            print("Error: task instruction cannot be empty")
            sys.exit(1)
        task = db.get_task_by_name(name)
        if task is None:
            db.add_task(name, instruction)
            task = db.get_task_by_name(name)
        return task["name"], task["instruction"]

    task = labels[chosen]
    return task["name"], task["instruction"]


def select_teacher() -> int:
    """Use fzf to choose (or create) a teacher; returns the DB teacher id."""
    db = get_db()
    teachers = db.get_teachers()

    _NEW = "<< Add new teacher >>"
    labels = [t["name"] for t in reversed(teachers)] + [_NEW]
    chosen = fzf_select(labels, prompt="Select teacher> ")[0]

    if chosen == _NEW:
        name = input("  New teacher name: ").strip()
        if not name:
            print("Error: teacher name cannot be empty")
            sys.exit(1)
        existing = db.get_teacher_by_name(name)
        if existing:
            return existing["id"]
        return db.add_teacher(name)

    existing = db.get_teacher_by_name(chosen)
    if existing:
        return existing["id"]
    # Shouldn't happen, but recover gracefully
    return db.add_teacher(chosen)


# ---------------------------------------------------------------------------
# Episode directory management
# ---------------------------------------------------------------------------


def _next_recording_dir(task_dir: Path) -> Path:
    """Return the directory to record the next episode into.

    If the last episode directory has no metadata or ``complete=False``, it is
    treated as an incomplete recording, wiped, and reused.  Otherwise a new
    numbered directory is created.
    """
    existing = sorted(d for d in task_dir.iterdir() if d.is_dir() and d.name.isdigit())

    if existing:
        last_dir = existing[-1]
        meta_file = last_dir / "metadata.json"
        is_incomplete = True
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            is_incomplete = not meta.get("complete", False)

        if is_incomplete:
            print(f"  Overriding incomplete recording: {last_dir}")
            shutil.rmtree(last_dir)
            last_dir.mkdir()
            return last_dir

    episode_idx = f"{len(existing):04d}"
    new_dir = task_dir / episode_idx
    new_dir.mkdir()
    return new_dir


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------


def upload_to_s3(recording_dir: Path, bucket: str, prefix: str) -> None:
    s3 = boto3.client("s3")
    for file_path in sorted(recording_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(recording_dir.parent)
        key = f"{prefix}/{rel}"
        print(f"  Uploading {file_path.name} → s3://{bucket}/{key}")
        s3.upload_file(str(file_path), bucket, key)

    print(f"\n✓ Upload complete → s3://{bucket}/{prefix}/")


# ---------------------------------------------------------------------------
# Keyboard helper
# ---------------------------------------------------------------------------


def _wait_for_enter_or_quit(
    robot_controller: RobotController,
    interface: TeleopInterface,
) -> bool:
    """Wait for Enter/Space (proceed) or 'q' (quit session).

    Returns:
        True  — 'q' pressed or footpedal e-stop; caller should end the session.
        False — Enter/Space or footpedal left fired; caller should proceed.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            if robot_controller.session_estop_requested:
                return True
            if interface.poll(robot_controller):
                return False
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == "q":
                    return True
                if ch in ("\r", "\n", " "):
                    return False
            time.sleep(0.05)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _wait_for_verdict(
    robot_controller: RobotController,
    interface: TeleopInterface,
) -> Optional[str]:
    """After recording stops, wait for the user to mark success or failure.

    Inputs accepted:
      - Footpedal middle button → "success"
      - Footpedal right button  → "failure"
      - Leader arm top button   → "success"  (non-spacemouse)
      - Leader arm bottom button→ "failure"  (non-spacemouse)
      - Enter key               → "success"
      - 'f' key                 → "failure"
      - Any other key / timeout → None (leaves status as "pending")

    Returns:
        "success", "failure", or None.
    """
    print("\n" + "-" * 60)
    print("  Mark this demonstration:")
    lines = []
    lines += ["    Middle pedal → success", "    Right pedal  → failure"]
    if interface.supports_verdict_button:
        lines += ["    Top button    → success", "    Bottom button → failure"]
    lines += ["    Enter → success   f → failure   other key → skip"]
    print("\n".join(lines))
    print("-" * 60 + "\n")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        deadline = time.monotonic() + 30.0  # 30 s timeout
        while time.monotonic() < deadline:
            if interface.poll_success(robot_controller):
                return "success"
            if interface.poll_failure(robot_controller):
                return "failure"
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    return "success"
                if ch.lower() == "f":
                    return "failure"
                return None
            if interface.supports_verdict_button:
                verdict = robot_controller.check_verdict_button()
                if verdict is not None:
                    return verdict
            time.sleep(0.05)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return None  # timed out


def _wait_for_start_or_quit(
    robot_controller: RobotController,
    interface: TeleopInterface,
) -> bool:
    """Wait for a leader button press (start) or the 'q' key (quit session).

    Returns:
        True  — 'q' pressed or footpedal e-stop; caller should end the session.
        False — leader button / footpedal left pressed; caller should start recording.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            if robot_controller.session_estop_requested:
                return True
            if interface.poll(robot_controller):
                return False
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == "q":
                    return True
            time.sleep(0.05)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_recording(
    s3_bucket: Optional[str],
    s3_prefix: str,
    interface: TeleopInterface,
    camera_config_file: str = CAMERA_CONFIG,
    calibration_file: str = CALIBRATION_FILE,
    arms: str = "bimanual",
    data_dir: str = "data",
) -> None:
    """Run teleoperation with continuous demonstration recording.

    Cameras and robots are fully instantiated and torn down each episode so
    every demonstration starts from a clean state.

    - Press the leader button to START each episode; press again to STOP and save.
    - After each stop the system re-initializes and shows a READY prompt.
      Press the leader button to start the next episode, or 'q' to end the session.
    - Incomplete recordings (estop / Ctrl-C) are detected via the ``complete``
      flag in metadata.json and overridden on the next run.
    """
    print("\n" + "=" * 60)
    print("  DEMONSTRATION RECORDING")
    print("=" * 60 + "\n")

    # ── task and teacher selection (once per session) ────────────────────
    task_name, task_instruction = select_task()
    task_dir = Path(data_dir) / "raw" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    teacher_id = select_teacher()

    print(f"\n  Task       : {task_name}")
    print(f"  Instruction: {task_instruction}\n")

    # ── snapshot camera config and look up latest calibration result ─────
    db = get_db()
    try:
        with open(camera_config_file) as _f:
            _cam_cfg_data = json.load(_f)
        camera_config_id = db.snapshot_camera_config(_cam_cfg_data, camera_config_file)
    except Exception:
        camera_config_id = db.snapshot_camera_config({}, camera_config_file)

    task_record = db.get_task_by_name(task_name)
    task_id = task_record["id"] if task_record else None

    latest_calib = db.get_latest_calibration_result()
    calibration_result_id = latest_calib["id"] if latest_calib else None

    def _copy_calibration(dest_dir: Path) -> None:
        calib_src = Path(calibration_file)
        if calib_src.exists():
            shutil.copy(calib_src, dest_dir / calib_src.name)
            print(f"✓ Calibration copied → {dest_dir / calib_src.name}")
        else:
            print(f"  Warning: calibration file not found at {calib_src}")

    # ── optional footpedal (once per session) ────────────────────────────
    interface.open()

    # ── one-time camera + robot initialisation ───────────────────────────
    print("Initializing cameras...")
    try:
        cameras = load_cameras_from_config(camera_config_file)
    except Exception as e:
        print(f"Error initialising cameras: {e}")
        interface.close()
        return
    print(f"✓ {len(cameras)} camera(s) ready\n")

    # Signal handler updated each episode to point at the current controller.
    _active_ctrl: List[Optional[RobotController]] = [None]

    def emergency_stop(signum, frame):
        if _active_ctrl[0] is not None:
            _active_ctrl[0].emergency_stop()

    signal.signal(signal.SIGTERM, emergency_stop)
    signal.signal(signal.SIGINT, emergency_stop)

    # ── continuous episode loop ──────────────────────────────────────────
    # Cameras stay open for the whole session (SDK init is slow).
    # Robot controller is reinited each episode — keeping motor CAN threads
    # alive while idle causes DM4310 watchdog / loss-communication errors.
    last_saved_dir: Optional[Path] = None
    recorder: Optional[DemonstrationRecorder] = None
    robot_controller: Optional[RobotController] = None

    use_right = arms == "bimanual"
    use_left = True

    try:
        while True:
            recorder = None

            # ── per-episode: init robots ──────────────────────────────────
            robot_controller = RobotController(
                use_right_leader=interface.uses_leaders and use_right,
                use_left_leader=interface.uses_leaders and use_left,
                use_right_follower=use_right,
                use_left_follower=use_left,
            )
            _active_ctrl[0] = robot_controller

            try:
                robot_controller.setup_for_teleop_recording()
            except Exception as e:
                print(f"Error initialising robots: {e}")
                break

            # Setup interface (warmup IK, attach devices, etc.)
            interface.setup(robot_controller)

            # Flush stdout so any SDK log output has time to drain before
            # printing the READY banner.
            sys.stdout.flush()
            time.sleep(0.1)

            # ── wait to start ────────────────────────────────────────────
            print("\n" + "=" * 60)
            print("  READY")
            print("=" * 60)
            print(f"\n  Data dir   : {task_dir}")
            if interface.waits_for_button_start:
                print("\n  Press button on any leader arm or left pedal to START.")
            else:
                print("\n  Press Enter or left pedal to START recording.")
            print("  Press 'q' to end session.\n")
            print("=" * 60 + "\n")

            if interface.waits_for_button_start:
                quit_session = _wait_for_start_or_quit(robot_controller, interface)
            else:
                quit_session = _wait_for_enter_or_quit(robot_controller, interface)
            if quit_session:
                print("\nEnding session.\n")
                robot_controller.shutdown()
                _active_ctrl[0] = None
                robot_controller = None
                break

            # ── start episode ────────────────────────────────────────────
            recording_dir = _next_recording_dir(task_dir)
            print(f"\n  Output: {recording_dir}")

            recorder = DemonstrationRecorder(
                cameras=cameras,
                robot_controller=robot_controller,
                recording_dir=recording_dir,
                task_name=task_name,
                task_instruction=task_instruction,
                interface=interface,
            )
            interface.start(robot_controller)
            recorder.start_recording()
            robot_controller.enable_estop()
            interface.set_active_recording(robot_controller)

            # ── wait for stop ────────────────────────────────────────────
            forced_failure = False
            forced_success = False

            needs_stdin = not interface.waits_for_button_start
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd) if needs_stdin else None
            if needs_stdin:
                tty.setcbreak(fd)
            try:
                while True:
                    if robot_controller.session_estop_requested:
                        break
                    if interface.poll_success(robot_controller):
                        forced_success = True
                        break
                    if interface.poll_failure(robot_controller):
                        forced_failure = True
                        break
                    if needs_stdin:
                        if select.select([sys.stdin], [], [], 0)[0]:
                            ch = sys.stdin.read(1)
                            if ch in ("\r", "\n", " "):
                                break
                    else:
                        if interface.poll(robot_controller):
                            break
                    time.sleep(0.05)
            finally:
                if old_settings is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            interface.set_active_recording(None)
            estop = robot_controller.session_estop_requested

            # ── verdict phase (arms still active, before shutdown) ────────
            if estop or forced_failure:
                verdict: Optional[str] = "failure"
            elif forced_success:
                verdict = "success"
            else:
                verdict = _wait_for_verdict(robot_controller, interface)

            # ── stop episode (shuts down robots, keeps cameras open) ──────
            saved_dir = recorder.stop_recording(complete=not estop)
            recorder = None
            _active_ctrl[0] = None
            robot_controller = None  # shut down inside stop_recording

            _copy_calibration(saved_dir)

            if estop:
                if task_id is not None:
                    demo_id = db.add_demonstration(
                        teacher_id=teacher_id,
                        task_id=task_id,
                        raw_data_path=str(saved_dir),
                        camera_config_id=camera_config_id,
                        calibration_result_id=calibration_result_id,
                    )
                    db.update_demonstration(demo_id, status="failure", converted=False)
                print("\nRecording aborted — marked as failure.\n")
                break

            last_saved_dir = saved_dir

            # ── record demonstration in DB ───────────────────────────────
            if task_id is not None:
                demo_id = db.add_demonstration(
                    teacher_id=teacher_id,
                    task_id=task_id,
                    raw_data_path=str(saved_dir),
                    camera_config_id=camera_config_id,
                    calibration_result_id=calibration_result_id,
                )
                status = verdict if verdict is not None else "pending"
                db.update_demonstration(demo_id, status=status, converted=False)
                if verdict:
                    print(f"  Demonstration marked as: {verdict}")

            print(f"✓ Recording saved to: {saved_dir}\n")
            # Loop back — cameras stay open, robots reinited next iteration.

    except KeyboardInterrupt:
        print("\nCancelled by user.")
        if recorder is not None and recorder.is_recording:
            saved_dir = recorder.stop_recording(complete=False)
            _copy_calibration(saved_dir)
            robot_controller = None
        elif robot_controller is not None:
            robot_controller.shutdown()
            robot_controller = None
    except Exception as e:
        print(f"\nError during recording: {e}")
        if recorder is not None and recorder.is_recording:
            saved_dir = recorder.stop_recording(complete=False)
            _copy_calibration(saved_dir)
            robot_controller = None
        elif robot_controller is not None:
            robot_controller.shutdown()
            robot_controller = None
    finally:
        # ── session teardown ──────────────────────────────────────────────
        # Robots are shut down per-episode; only cameras and interface remain.
        if robot_controller is not None:
            robot_controller.shutdown()
        for cam in cameras:
            cam.close()
        interface.close()

    # ── optional S3 upload ───────────────────────────────────────────────
    if s3_bucket and last_saved_dir:
        print("\nUploading to S3...")
        upload_to_s3(last_saved_dir, s3_bucket, s3_prefix)

    if last_saved_dir:
        print(f"\n✓ Done.  Last recording at: {last_saved_dir}")
        print("  Run 'rd convert' to extract PNG frames from camera files.\n")
    else:
        print("\n✓ Done (no recordings saved).\n")
