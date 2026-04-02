"""Policy server for the YAM bimanual robot — serves live observations via chiral.

Bridges the raiden robot stack (ZED/RealSense cameras + YAM follower arms) to
any remote policy client that speaks the chiral WebSocket protocol.

Usage::

    rd serve

Or with custom options::

    rd serve --port 8765 --stereo-method ffs

Thread layout::

    camera-<name>       : grabs frames from ZED or RealSense at ~30 Hz (per camera)
    proprio-<name>      : reads joint state from follower arms at ~100 Hz (per stream)
    asyncio event loop  : handles WebSocket connections (chiral protocol)

Extrinsics convention
---------------------
All extrinsics are expressed in the left-arm base frame (matching the dataset
convention used by the converter).

*Wrist cameras* have moving extrinsics that are recomputed on every
``_make_obs()`` call::

    T_left_base→cam  =  [T_left_base_from_right_base @]  FK(q[:6])  @  T_cam→ee

*Scene camera* extrinsics are static and loaded once from the calibration file.

Image orientation
-----------------
Cameras in ``_FLIP_CAMERAS`` (e.g. ``right_wrist_camera``) are physically
mounted upside-down.  The capture loop rotates their images by 180° to match
the right-side-up orientation used by the training dataset.  The principal
point in the intrinsics and the ``T_cam→ee`` rotation are corrected accordingly.
"""

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import chiral
import cv2
import numpy as np
from chiral.types import CameraInfo, Observation

from raiden.camera_config import CameraConfig as RaidenCameraConfig
from raiden.robot.controller import RobotController, smooth_move_joints

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cameras mounted upside-down: images are rotated 180° and extrinsics/
# intrinsics are corrected to match the right-side-up frame.
_FLIP_CAMERAS = {"right_wrist_camera"}

# Maps wrist camera name → which follower arm drives its extrinsics.
_WRIST_CAMERA_ARM: dict[str, str] = {
    "left_wrist_camera": "left",
    "right_wrist_camera": "right",
}

# 4×4 homogeneous 180° rotation around the Z (optical) axis.
# Right-multiplying T_cam→ee by this converts from the physical (upside-down)
# camera frame used during calibration to the standard (right-side-up) frame
# that matches the rotated images served by the policy server.
_R_FLIP_180 = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DOF = 7  # joints per arm (6 revolute + 1 gripper)
BIMANUAL_DOF = DOF * 2  # right arm then left arm

# EE pose action layout — matches vla_foundry action_fields concatenation order:
#   [l_xyz(3), r_xyz(3), l_rot6d(6), r_rot6d(6), l_grip(1), r_grip(1)]  (20-D)
EE_POSE_ARM_DOF = 10  # per arm: xyz(3) + rot_6d(6) + grip(1)
BIMANUAL_EE_POSE_DOF = EE_POSE_ARM_DOF * 2

# Robot name used in proprioception key names — must match the shardify convention.
_ROBOT = "yam"

# Number of proprio samples kept in the interpolation ring buffer (~0.64 s at 100 Hz).
_PROPRIO_HISTORY_SIZE = 64

# Default maximum allowed joint position delta per policy step.
# At a 20 Hz control rate, 0.2 rad/step ≈ 4 rad/s — large enough to catch
# abrupt policy jumps while allowing normal motion.
_DEFAULT_MAX_JOINT_DELTA = 0.2  # radians

_CONTROL_HZ = 1.0

# Lazily-loaded MuJoCo kinematics instance (shared across calls).
_kinematics: Any = None


def _get_kinematics() -> Any:
    global _kinematics
    if _kinematics is None:
        from i2rt.robots.kinematics import Kinematics

        from raiden._xml_paths import get_yam_4310_linear_xml_path

        _kinematics = Kinematics(get_yam_4310_linear_xml_path(), "grasp_site")
    return _kinematics


def _rot6d_to_mat(v: np.ndarray) -> np.ndarray:
    """Convert a 6D rotation representation to a 3×3 rotation matrix.

    Uses the Gram-Schmidt orthonormalisation of the first two rows, matching
    vla_foundry's ``rot_6d_to_matrix`` convention.

    Args:
        v: (6,) array — first two rows of a rotation matrix: [R[0,:], R[1,:]].

    Returns:
        (3, 3) float64 rotation matrix.
    """
    a1, a2 = v[:3].astype(np.float64), v[3:6].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)  # rows


def _fk_padded(kin: Any, q6: np.ndarray) -> np.ndarray:
    """Call FK, padding q6 with zeros to the model's nq (combined XML has nq=8)."""
    nq = kin._configuration.model.nq
    q = np.zeros(nq, dtype=np.float64)
    q[: len(q6)] = q6
    return kin.fk(q)


def _fk_to_xyz_rot6d(kin: Any, q6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run FK and return ``(xyz, rot_6d)`` for the end-effector site.

    Args:
        kin: Kinematics instance.
        q6: (6,) arm joint angles (no gripper).

    Returns:
        xyz: (3,) float32 position.
        rot_6d: (6,) float32 — first two columns of the rotation matrix.
    """
    T = _fk_padded(kin, q6)
    xyz = T[:3, 3].astype(np.float32)
    rot_6d = (
        T[:2, :3].flatten().astype(np.float32)
    )  # row0 + row1 → [R00,R01,R02,R10,R11,R12]
    return xyz, rot_6d


def _pose_from_xyz_rot6d(xyz: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
    """Build a 4×4 TCP pose from a position and 6D rotation vector."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = xyz.astype(np.float64)
    T[:3, :3] = _rot6d_to_mat(rot6d)
    return T


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class RaidenPolicyServer(chiral.PolicyServer):
    """Policy server for the YAM bimanual robot system.

    Streams live camera images, depth maps, and joint-state observations to a
    remote policy over WebSocket using the chiral protocol.

    Args:
        camera_config_file: Path to ``camera.json``.
        calibration_file: Path to ``calibration_results.json``.  Camera
            intrinsics are read from the SDK at the actual serving resolution;
            the calibration file supplies extrinsics and ``T_cam→ee``.
        host: WebSocket host to bind to.
        port: WebSocket port to listen on.
    """

    def __init__(
        self,
        camera_config_file: str = "./config/camera.json",
        calibration_file: str = "./config/calibration_results.json",
        host: str = "0.0.0.0",
        port: int = 8765,
        stereo_method: str = "zed",
        ffs_scale: float = 1.0,
        ffs_iters: int = 8,
        tri_stereo_variant: str = "c64",
        max_joint_delta: float = _DEFAULT_MAX_JOINT_DELTA,
        action_type: str = "ee_pose",
        no_depth: bool = False,
    ):
        self._no_depth = no_depth
        if action_type not in ("joint", "ee_pose"):
            raise ValueError(
                f"action_type must be 'joint' or 'ee_pose', got {action_type!r}"
            )
        self._action_type = action_type
        self._raiden_cam_cfg = RaidenCameraConfig(camera_config_file)
        self._calibration = self._load_calibration(calibration_file)
        self._stereo_method = stereo_method

        # Lazily-loaded learned-stereo predictor (shared across all ZED cameras).
        # Both FFS and TRI Stereo share the same predict(left, right, fx, baseline) API.
        self._ffs_predictor = None
        if stereo_method == "ffs":
            from raiden.depth.ffs import (
                FFSDepthPredictor,
                FFSOnnxDepthPredictor,
                FFSTrtDepthPredictor,
            )

            if FFSTrtDepthPredictor.engines_available():
                self._ffs_predictor = FFSTrtDepthPredictor()
            elif FFSOnnxDepthPredictor.models_available():
                self._ffs_predictor = FFSOnnxDepthPredictor()
            else:
                self._ffs_predictor = FFSDepthPredictor(
                    scale=ffs_scale, iters=ffs_iters
                )
                print(
                    "[FFS] Using Fast Foundation Stereo (PyTorch)"
                    + (f" scale={ffs_scale}" if ffs_scale != 1.0 else "")
                )
        elif stereo_method == "tri_stereo":
            from raiden.depth.tri_stereo import (
                TRIStereoOnnxDepthPredictor,
                TRIStereoTrtDepthPredictor,
            )

            pred = None
            if TRIStereoTrtDepthPredictor.engine_available(variant=tri_stereo_variant):
                try:
                    trt = TRIStereoTrtDepthPredictor(variant=tri_stereo_variant)
                    trt._ensure_loaded()
                    pred = trt
                except RuntimeError as e:
                    print(
                        f"[TRIStereo] TRT engine unusable ({e}), falling back to ONNX"
                    )
            else:
                print(
                    f"[TRIStereo] No TRT engine found for variant '{tri_stereo_variant}', "
                    "using ONNX. Compile an engine with trtexec for faster inference."
                )
            if pred is None:
                if TRIStereoOnnxDepthPredictor.model_available(
                    variant=tri_stereo_variant
                ):
                    pred = TRIStereoOnnxDepthPredictor(variant=tri_stereo_variant)
                    pred._ensure_loaded()
                else:
                    raise RuntimeError(
                        f"No TRI Stereo model found for variant '{tri_stereo_variant}'. "
                        "Run: git lfs pull"
                    )
            self._ffs_predictor = pred

        # Per-camera stereo calibration (fx, baseline) populated by _open_zed.
        self._stereo_calib: dict[str, tuple[float, float]] = {}

        # Per-camera data populated across _open_cameras() and
        # _prepare_camera_transforms():
        #   _cam_handles     : hardware handles keyed by camera name
        #   _cam_intrinsics  : 3×3 float64 camera matrix (from SDK, flip-corrected)
        #   _cam_extrinsics  : 4×4 float64 static extrinsics (scene cameras only)
        #   _T_cam2ee        : 4×4 float64 T_cam→ee (wrist cameras, flip-corrected)
        self._cam_handles: dict = {}
        self._cam_intrinsics: dict[str, np.ndarray] = {}
        self._cam_extrinsics: dict[str, np.ndarray] = {}
        self._T_cam2ee: dict[str, np.ndarray] = {}

        # T_left_base_from_right_base = inv(T_right_base_to_left_base)
        # Brings right-arm FK results into the left-arm base frame.
        self._T_left_base_from_right_base: Optional[np.ndarray] = None

        # Open cameras first so we can query actual H, W, and SDK intrinsics
        # before super().__init__() calls camera_configs().
        print("\nOpening cameras...")
        self._open_cameras()
        self._prepare_camera_transforms()

        # Per-camera frame arrival timestamps (monotonic_ns) updated by capture
        # loops, plus one-time phase offsets computed at startup so all cameras
        # share a common time axis.  Initialised before threads start.
        self._cam_arrival_ts_ns: dict[str, int] = {
            name: 0 for name in self._cam_handles
        }
        self._cam_ts_locks: dict[str, threading.Lock] = {
            name: threading.Lock() for name in self._cam_handles
        }
        self._cam_phase_offset_ns: dict[str, int] = {
            name: 0 for name in self._cam_handles
        }

        # Initialize follower robots only (leaders not needed for inference).
        self._robot = RobotController(
            use_right_leader=False,
            use_left_leader=False,
            use_right_follower=True,
            use_left_follower=True,
        )
        self._robot.initialize_robots()

        # super().__init__() calls camera_configs() and proprio_configs() to
        # pre-allocate self.images, self.depths, and self.proprios.
        super().__init__(host=host, port=port)

        # Proprio interpolation ring buffers — keyed by proprio name, each holds
        # (monotonic_ns, value) pairs.  Initialised after super().__init__() since
        # proprio names come from proprio_configs().
        self._proprio_history: dict[str, deque] = {
            name: deque(maxlen=_PROPRIO_HISTORY_SIZE) for name in self.proprios
        }
        self._proprio_history_locks: dict[str, threading.Lock] = {
            name: threading.Lock() for name in self.proprios
        }

        self._max_joint_delta = max_joint_delta
        self._step_count = 0
        self._t_sum = 0.0
        self._running = True

        for name in self._raiden_cam_cfg.list_camera_names():
            threading.Thread(
                target=self._camera_loop, args=(name,), daemon=True
            ).start()

        for name in self.proprios:
            threading.Thread(
                target=self._proprio_loop, args=(name,), daemon=True
            ).start()

        # Single dedicated thread for learned stereo depth inference.
        # Processes all cameras sequentially so the GPU is never contested.
        if (
            self._stereo_method in ("ffs", "tri_stereo")
            and self._ffs_predictor is not None
        ):
            threading.Thread(target=self._depth_inference_loop, daemon=True).start()

        # Wait for the first frame from every camera, then compute phase offsets.
        print("\nComputing camera phase offsets...")
        self._compute_camera_offsets()

        # Attach footpedal soft e-stop (optional — warns and continues if absent).
        print("Initializing footpedal...")
        self._robot.attach_footpedal()

        # Background thread: shut down when footpedal e-stop fires.
        threading.Thread(target=self._estop_monitor, daemon=True).start()

        print("\nRaiden policy server ready.")

    # -------------------------------------------------------------------------
    # Calibration helpers
    # -------------------------------------------------------------------------

    def _load_calibration(self, path: str) -> dict:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        print(
            f"Note: calibration file not found at '{path}'; "
            "using default intrinsics / extrinsics."
        )
        return {}

    def _prepare_camera_transforms(self) -> None:
        """Load extrinsics and T_cam→ee from the calibration file.

        Intrinsics come from the SDK (already stored in _cam_handles by
        _open_cameras()); the calibration file is only used for extrinsics.
        """
        # The calibration file nests per-camera data under a top-level "cameras" key.
        calib_cameras: dict = self._calibration.get("cameras", {})

        # "right_base_to_left_base" maps left_arm_base → right_arm_base (despite the name).
        # Invert to bring right_arm_base points into left_arm_base.
        bimanual = self._calibration.get("bimanual_transform", {})
        mat = bimanual.get("right_base_to_left_base")
        if mat is not None:
            self._T_left_base_from_right_base = np.linalg.inv(
                np.array(mat, dtype=np.float64)
            )
        else:
            print(
                "Note: bimanual_transform not found in calibration; "
                "right wrist extrinsics will be in right-arm base frame."
            )

        for name in self._raiden_cam_cfg.list_camera_names():
            handle = self._cam_handles.get(name)
            flip = name in _FLIP_CAMERAS

            # ── intrinsics (from SDK, already at serving resolution) ──────
            if handle is not None:
                K = handle["intrinsics"].copy()
                if flip:
                    # Reflect the principal point for the 180°-rotated image.
                    w, h = handle["w"], handle["h"]
                    K[0, 2] = (w - 1) - K[0, 2]
                    K[1, 2] = (h - 1) - K[1, 2]
                self._cam_intrinsics[name] = K
            else:
                # Camera failed to open — use a placeholder.
                self._cam_intrinsics[name] = np.eye(3, dtype=np.float64)

            # ── extrinsics ────────────────────────────────────────────────
            cam_data: dict = calib_cameras.get(name, {})

            if name in _WRIST_CAMERA_ARM:
                # Wrist cameras: load T_cam→ee; extrinsics are dynamic (FK).
                he: dict = cam_data.get("hand_eye_calibration", {})
                if he.get("success") and "rotation_matrix" in he:
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = np.array(he["rotation_matrix"])
                    T[:3, 3] = np.array(he["translation_vector"]).flatten()
                    if flip:
                        # Calibration was done with raw (upside-down) images.
                        # Right-multiplying by _R_FLIP_180 converts T_cam→ee
                        # to the right-side-up camera frame used for serving.
                        T = T @ _R_FLIP_180
                    self._T_cam2ee[name] = T
                else:
                    print(
                        f"Note: hand-eye calibration missing for '{name}'; "
                        "wrist extrinsics will be identity."
                    )
                # Placeholder; overwritten per-step in _make_obs().
                self._cam_extrinsics[name] = np.eye(4, dtype=np.float64)
            else:
                # Scene camera: static extrinsics.
                ext: dict = cam_data.get("extrinsics", {})
                if ext.get("success") and "rotation_matrix" in ext:
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = np.array(ext["rotation_matrix"])
                    T[:3, 3] = np.array(ext["translation_vector"]).flatten()
                    self._cam_extrinsics[name] = T
                else:
                    self._cam_extrinsics[name] = np.eye(4, dtype=np.float64)

    # -------------------------------------------------------------------------
    # chiral.PolicyServer interface
    # -------------------------------------------------------------------------

    def camera_configs(self) -> list[chiral.CameraConfig]:
        configs = []
        for name in self._raiden_cam_cfg.list_camera_names():
            handle = self._cam_handles.get(name, {})
            h = handle.get("h", 0)
            w = handle.get("w", 0)
            is_zed = handle.get("type") == "zed"
            has_depth = not (self._no_depth and is_zed)
            configs.append(
                chiral.CameraConfig(
                    name=name,
                    height=h,
                    width=w,
                    channels=3,
                    has_depth=has_depth,
                    intrinsics=self._cam_intrinsics[name],
                    extrinsics=self._cam_extrinsics[name],
                )
            )
        return configs

    def proprio_configs(self) -> list[chiral.ProprioConfig]:
        streams = []
        if self._robot.follower_r:
            streams.append(chiral.ProprioConfig(name="follower_r_joint_pos", size=DOF))
            streams.append(chiral.ProprioConfig(name="follower_r_joint_vel", size=DOF))
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__poses__right::{_ROBOT}__xyz", size=3
                )
            )
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__poses__right::{_ROBOT}__rot_6d", size=6
                )
            )
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__grippers__right::{_ROBOT}_hand", size=1
                )
            )
        if self._robot.follower_l:
            streams.append(chiral.ProprioConfig(name="follower_l_joint_pos", size=DOF))
            streams.append(chiral.ProprioConfig(name="follower_l_joint_vel", size=DOF))
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__poses__left::{_ROBOT}__xyz", size=3
                )
            )
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__poses__left::{_ROBOT}__rot_6d", size=6
                )
            )
            streams.append(
                chiral.ProprioConfig(
                    name=f"robot__actual__grippers__left::{_ROBOT}_hand", size=1
                )
            )
        return streams

    async def get_metadata(self) -> dict:
        action_shape = (
            [1, BIMANUAL_EE_POSE_DOF]
            if self._action_type == "ee_pose"
            else [1, BIMANUAL_DOF]
        )
        return {
            "cameras": self._raiden_cam_cfg.list_camera_names(),
            "action_type": self._action_type,
            "action_shape": action_shape,
            "action_layout": (
                "left_xyz(3)+right_xyz(3)+left_rot6d(6)+right_rot6d(6)+left_grip(1)+right_grip(1)"
                if self._action_type == "ee_pose"
                else "right_joints(7)+left_joints(7)"
            ),
            "proprio_names": list(self.proprios.keys()),
        }

    async def reset(self) -> tuple[Observation, dict]:
        self._step_count = 0
        self._t_sum = 0.0
        self._robot.move_to_home_positions(simultaneous=True)
        return self._make_obs(), {}

    async def step(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, bool, dict]:
        t0 = time.perf_counter()

        # Accept (D,), (1, D), or (N, D).
        # When N > 1, execute each step sequentially at _CONTROL_HZ.
        action = np.asarray(action)
        if action.ndim == 1:
            action = action[None]  # (1, D)

        for i, step_action in enumerate(action):
            step_action = step_action.reshape(-1)
            print("step_action", step_action)

            if self._action_type == "ee_pose":
                joint_cmd = self._ee_pose_to_joint_cmd(step_action)
            else:
                joint_cmd = step_action

            print("joint_cmd", joint_cmd)
            # Safety check: abort if any joint delta exceeds the threshold.
            self._check_joint_delta(joint_cmd)

            # Smoothly interpolate to the target over one control period.
            # _smooth_command blocks for 1/_CONTROL_HZ seconds, so no extra sleep needed.
            # self._smooth_command(joint_cmd)

        # Use the final joint_cmd for the safety check reference already done above.
        # Reuse joint_cmd from the last iteration.

        obs = self._make_obs(timestamp=self._step_count * 0.05)
        step_ms = (time.perf_counter() - t0) * 1e3
        self._step_count += 1
        self._t_sum += step_ms

        if self._step_count % 10 == 0:
            print(
                f"step={self._step_count:4d}  "
                f"step_ms={step_ms:5.2f}  "
                f"avg={self._t_sum / self._step_count:5.2f}ms"
            )

        return obs, 0.0, False, False, {}

    # -------------------------------------------------------------------------
    # Camera temporal alignment
    # -------------------------------------------------------------------------

    def _get_reference_camera(self) -> str:
        """Return the reference camera name for temporal alignment.
        Prefers 'scene_camera'; falls back to the first camera in the config.
        """
        names = list(self._cam_handles.keys())
        if "scene_camera" in names:
            return "scene_camera"
        return names[0] if names else ""

    def _wait_for_first_frames(self, timeout_s: float = 5.0) -> None:
        """Block until every camera loop has delivered at least one frame."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if all(self._cam_arrival_ts_ns.get(n, 0) > 0 for n in self._cam_handles):
                return
            time.sleep(0.05)
        missing = [
            n for n in self._cam_handles if self._cam_arrival_ts_ns.get(n, 0) == 0
        ]
        print(
            f"  Warning: cameras did not produce a first frame within {timeout_s:.0f}s: {missing}"
        )

    def _compute_camera_offsets(self) -> None:
        """Compute per-camera phase offsets relative to the reference camera.

        Captures ``time.monotonic_ns()`` arrival timestamps from the first frame
        of each camera.  Since FPS is assumed consistent, this one-time offset
        corrects the fixed phase difference between capture loops so that::

            arrival_ts_ns + phase_offset_ns  →  unified time axis
        """
        self._wait_for_first_frames()
        names = list(self._cam_handles.keys())
        if not names:
            return

        arrival_ns: dict[str, int] = {}
        for name in names:
            with self._cam_ts_locks[name]:
                arrival_ns[name] = self._cam_arrival_ts_ns[name]

        ref_name = self._get_reference_camera()
        ref_ts = arrival_ns.get(ref_name, 0)

        self._cam_phase_offset_ns = {
            name: ref_ts - arrival_ns.get(name, ref_ts) for name in names
        }

        print(f"  Reference camera: '{ref_name}'")
        for name in names:
            offset_ms = self._cam_phase_offset_ns[name] / 1e6
            print(f"    {name}: {offset_ms:+.1f} ms")

    # -------------------------------------------------------------------------
    # Proprio interpolation
    # -------------------------------------------------------------------------

    def _interpolate_proprio(
        self, name: str, target_ts_ns: int
    ) -> Optional[np.ndarray]:
        """Interpolate the proprio ring buffer to *target_ts_ns* (monotonic_ns).

        Falls back to the latest buffered value when the history is too short
        (e.g. at startup).  Returns None if no data is available at all.
        """
        if name not in self._proprio_history:
            return self._read_proprio(name)

        with self._proprio_history_locks[name]:
            history = list(self._proprio_history[name])

        if len(history) < 2:
            return self._read_proprio(name)

        ts_arr = np.array([h[0] for h in history], dtype=np.float64)
        val_arr = np.stack([h[1] for h in history], axis=0)

        t = float(target_ts_ns)
        # np.interp clamps at boundaries, so out-of-range values return the
        # nearest endpoint rather than extrapolating.
        return np.stack(
            [np.interp(t, ts_arr, val_arr[:, d]) for d in range(val_arr.shape[1])],
            axis=0,
        ).astype(np.float32)

    # -------------------------------------------------------------------------
    # EE pose → joint IK
    # -------------------------------------------------------------------------

    def _ee_pose_to_joint_cmd(self, action: np.ndarray) -> np.ndarray:
        """Convert a 20-D EE pose action to a 14-D joint command via IK.

        Action layout (matching vla_foundry action_fields concatenation order)::

            [l_xyz(3), r_xyz(3), l_rot6d(6), r_rot6d(6), l_grip(1), r_grip(1)]

        Poses are in each arm's own base frame (left arm in left-base frame,
        right arm in right-base frame) — same convention as the stored action
        in ``lowdim.npz``.

        Returns:
            (14,) float32 joint command — left arm first, then right arm,
            matching the joint action convention expected by ``step()``.
        """
        kin = _get_kinematics()

        # action layout: [l_xyz(3), r_xyz(3), l_rot6d(6), r_rot6d(6), l_grip(1), r_grip(1)]
        T_l = _pose_from_xyz_rot6d(action[0:3], action[6:12])
        l_grip = float(action[18])
        T_r = (
            _pose_from_xyz_rot6d(action[3:6], action[12:18])
            if len(action) >= 20
            else None
        )
        r_grip = float(action[19]) if len(action) >= 20 else 0.0

        # Seed IK from the current measured joint positions, padded to model nq.
        nq = kin._configuration.model.nq
        q_l_prop = self._read_proprio("follower_l_joint_pos")
        q_r_prop = self._read_proprio("follower_r_joint_pos")

        def _pad_init(q7: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if q7 is None:
                return None
            q = np.zeros(nq, dtype=np.float64)
            q[:6] = q7[:6]
            return q

        _, q_l_full = kin.ik(T_l, "grasp_site", init_q=_pad_init(q_l_prop))
        cmd_l = np.append(q_l_full[:6], l_grip).astype(np.float32)

        if T_r is not None:
            _, q_r_full = kin.ik(T_r, "grasp_site", init_q=_pad_init(q_r_prop))
            cmd_r = np.append(q_r_full[:6], r_grip).astype(np.float32)
        else:
            cmd_r = np.zeros(DOF, dtype=np.float32)

        # IK returns full nq — take arm joints only, then append gripper.
        # Output layout: left arm first, then right arm (matches joint action convention).
        return np.concatenate([cmd_l, cmd_r])

    # -------------------------------------------------------------------------
    # Smooth joint command
    # -------------------------------------------------------------------------

    def _smooth_command(self, joint_cmd: np.ndarray) -> None:
        """Send a joint command smoothly over one control period (1 / _CONTROL_HZ).

        Both arms are interpolated in parallel threads so neither blocks the other.

        Args:
            joint_cmd: (14,) float32 — left arm (7) then right arm (7),
                where each 7-D slice is [arm_joints(6), gripper(1)].
        """
        threads = []
        if self._robot.follower_l:
            t = threading.Thread(
                target=smooth_move_joints,
                args=(self._robot.follower_l, joint_cmd[:DOF]),
                kwargs={"time_interval_s": 1.0 / _CONTROL_HZ, "steps": 200},
                daemon=True,
            )
            threads.append(t)
        if self._robot.follower_r:
            t = threading.Thread(
                target=smooth_move_joints,
                args=(self._robot.follower_r, joint_cmd[DOF : DOF * 2]),
                kwargs={"time_interval_s": 1.0 / _CONTROL_HZ, "steps": 200},
                daemon=True,
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # -------------------------------------------------------------------------
    # Joint-delta safety check
    # -------------------------------------------------------------------------

    def _check_joint_delta(self, action: np.ndarray) -> None:
        """Abort the server if the commanded action jumps too far from current positions.

        Compares each arm's commanded joint positions against the latest buffered
        proprioception values.  If any joint delta exceeds ``_max_joint_delta``
        (radians), prints an error and calls ``os._exit(1)`` — the command is
        never sent to the robot.

        Args:
            action: Flat array of length ≥ ``BIMANUAL_DOF`` (left arm then right arm).
        """
        pairs = []
        if self._robot.follower_l:
            q_l = self._read_proprio("follower_l_joint_pos")
            if q_l is not None:
                pairs.append(("left", q_l, action[:DOF]))
        if self._robot.follower_r:
            q_r = self._read_proprio("follower_r_joint_pos")
            if q_r is not None:
                pairs.append(("right", q_r, action[DOF : DOF * 2]))

        for arm, current, commanded in pairs:
            delta = np.abs(commanded - current)
            print("delta", arm, commanded, current)
            max_delta = float(delta.max())
            if max_delta > self._max_joint_delta:
                joint_idx = int(delta.argmax())
                print(
                    f"\n[SAFETY] Dangerously large joint delta on {arm} arm — "
                    f"joint {joint_idx}: {max_delta:.4f} rad "
                    f"(limit={self._max_joint_delta:.4f} rad). "
                    "Stopping policy server."
                )
                self.close()
                os._exit(1)

    # -------------------------------------------------------------------------
    # Observation construction (overrides base class to add dynamic extrinsics)
    # -------------------------------------------------------------------------

    def _make_obs(self, timestamp: float = 0.0) -> Observation:
        """Snapshot all buffers and compute per-step wrist camera extrinsics.

        All joint states are interpolated to the reference camera's latest frame
        arrival time so the observation is temporally coherent across cameras and
        the proprioception stream.
        """
        # Reference timestamp: the reference camera's arrival time shifted onto
        # the unified time axis (arrival_ts + phase_offset).
        ref_cam = self._get_reference_camera()
        if ref_cam and self._cam_arrival_ts_ns.get(ref_cam, 0) > 0:
            with self._cam_ts_locks[ref_cam]:
                ref_ts_ns = self._cam_arrival_ts_ns[
                    ref_cam
                ] + self._cam_phase_offset_ns.get(ref_cam, 0)
        else:
            ref_ts_ns = time.monotonic_ns()

        # Interpolate joint positions to reference timestamp for FK.
        q_r = self._interpolate_proprio("follower_r_joint_pos", ref_ts_ns)
        q_l = self._interpolate_proprio("follower_l_joint_pos", ref_ts_ns)

        cameras: list[CameraInfo] = []
        for c in self._configs:
            with self._locks[c.name]:
                image = self.images[c.name].copy()
                depth = self.depths[c.name].copy() if c.name in self.depths else None

            extrinsics = self._compute_extrinsics(c.name, q_r, q_l)
            cameras.append(
                CameraInfo(
                    name=c.name,
                    intrinsics=c.intrinsics,
                    extrinsics=extrinsics,
                    image=image,
                    depth=depth,
                )
            )

        # EE pose proprio names are derived from FK — skip ring-buffer lookup for them.
        _ee_pose_keys = {
            f"robot__actual__poses__left::{_ROBOT}__xyz",
            f"robot__actual__poses__left::{_ROBOT}__rot_6d",
            f"robot__actual__grippers__left::{_ROBOT}_hand",
            f"robot__actual__poses__right::{_ROBOT}__xyz",
            f"robot__actual__poses__right::{_ROBOT}__rot_6d",
            f"robot__actual__grippers__right::{_ROBOT}_hand",
        }

        proprios: dict[str, np.ndarray] = {}
        for p in self._proprio_configs:
            if p.name in _ee_pose_keys:
                continue  # filled below via FK
            val = self._interpolate_proprio(p.name, ref_ts_ns)
            if val is not None:
                proprios[p.name] = val
            else:
                with self._proprio_locks[p.name]:
                    proprios[p.name] = self.proprios[p.name].copy()

        # Compute actual EE poses via FK and inject as proprio.
        kin = _get_kinematics()
        if q_l is not None:
            xyz_l, rot6d_l = _fk_to_xyz_rot6d(kin, q_l[:6])
            proprios[f"robot__actual__poses__left::{_ROBOT}__xyz"] = xyz_l
            proprios[f"robot__actual__poses__left::{_ROBOT}__rot_6d"] = rot6d_l
            proprios[f"robot__actual__grippers__left::{_ROBOT}_hand"] = q_l[6:7].astype(
                np.float32
            )
        if q_r is not None:
            xyz_r, rot6d_r = _fk_to_xyz_rot6d(kin, q_r[:6])
            proprios[f"robot__actual__poses__right::{_ROBOT}__xyz"] = xyz_r
            proprios[f"robot__actual__poses__right::{_ROBOT}__rot_6d"] = rot6d_r
            proprios[f"robot__actual__grippers__right::{_ROBOT}_hand"] = q_r[
                6:7
            ].astype(np.float32)

        return Observation(cameras=cameras, proprios=proprios, timestamp=timestamp)

    def _read_proprio(self, name: str) -> Optional[np.ndarray]:
        """Return a snapshot of a proprio buffer, or None if not available."""
        if name not in self.proprios:
            return None
        with self._proprio_locks[name]:
            return self.proprios[name].copy()

    def _compute_extrinsics(
        self,
        camera_name: str,
        q_r: Optional[np.ndarray],
        q_l: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return the current 4×4 extrinsic matrix for *camera_name*.

        Scene cameras return their static matrix.  Wrist cameras compute::

            T_left_base→cam = [T_left_base_from_right_base @] FK(q[:6]) @ T_cam→ee
        """
        arm = _WRIST_CAMERA_ARM.get(camera_name)
        if arm is None or camera_name not in self._T_cam2ee:
            return self._cam_extrinsics[camera_name]

        T_cam2ee = self._T_cam2ee[camera_name]
        q = q_r if arm == "right" else q_l

        if q is None:
            return np.eye(4, dtype=np.float64)

        try:
            kin = _get_kinematics()
            T_base_to_ee = _fk_padded(kin, q[:6])
            T = T_base_to_ee @ T_cam2ee

            if arm == "right" and self._T_left_base_from_right_base is not None:
                T = self._T_left_base_from_right_base @ T

            return T
        except Exception as e:
            print(f"FK error for '{camera_name}': {e}")
            return np.eye(4, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Camera management
    # -------------------------------------------------------------------------

    def _open_cameras(self) -> None:
        for name in self._raiden_cam_cfg.list_camera_names():
            cam_type = self._raiden_cam_cfg.get_camera_type(name) or "zed"
            serial = self._raiden_cam_cfg.get_serial_by_name(name)
            try:
                if cam_type == "zed":
                    handle = self._open_zed(int(serial))
                else:
                    handle = self._open_realsense(str(serial))
                self._cam_handles[name] = handle
                if cam_type == "zed" and self._stereo_method in ("ffs", "tri_stereo"):
                    self._stereo_calib[name] = (handle["fx"], handle["baseline"])
                print(
                    f"  ✓ Opened {cam_type} camera '{name}' "
                    f"(serial={serial}, {handle['w']}×{handle['h']})"
                )
            except Exception as e:
                print(f"  ✗ Failed to open camera '{name}': {e}")

    def _open_zed(self, serial: int) -> dict:
        import pyzed.sl as sl

        cam = sl.Camera()
        params = sl.InitParameters()
        params.set_from_serial_number(serial)
        params.camera_resolution = sl.RESOLUTION.HD720
        params.camera_fps = 30
        params.depth_mode = (
            sl.DEPTH_MODE.NONE
            if self._no_depth or self._stereo_method in ("ffs", "tri_stereo")
            else sl.DEPTH_MODE.NEURAL_LIGHT
        )
        params.coordinate_units = sl.UNIT.METER
        params.depth_minimum_distance = 0.1
        status = cam.open(params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed (serial={serial}): {status}")

        info = cam.get_camera_information()
        res = info.camera_configuration.resolution
        cal_params = info.camera_configuration.calibration_parameters
        cal = cal_params.left_cam
        K = np.array(
            [[cal.fx, 0.0, cal.cx], [0.0, cal.fy, cal.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        handle = {
            "type": "zed",
            "camera": cam,
            "image_mat": sl.Mat(),
            "depth_mat": sl.Mat(),
            "right_mat": sl.Mat(),
            "h": res.height,
            "w": res.width,
            "intrinsics": K,
            # Stereo inference fields (used when stereo_method is ffs or tri_stereo).
            "stereo_lock": threading.Lock(),
            "latest_left": None,
            "latest_right": None,
            "stereo_seq": 0,
            "last_depth_seq": -1,
        }
        if self._stereo_method in ("ffs", "tri_stereo"):
            handle["fx"] = float(cal.fx)
            handle["baseline"] = float(abs(cal_params.get_camera_baseline()))
        return handle

    def _open_realsense(self, serial: str) -> dict:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        profile = pipeline.start(cfg)

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        K = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        align = rs.align(rs.stream.color)
        return {
            "type": "realsense",
            "pipeline": pipeline,
            "align": align,
            "depth_scale": depth_scale,
            "h": intr.height,
            "w": intr.width,
            "intrinsics": K,
        }

    def _camera_loop(self, name: str) -> None:
        handle = self._cam_handles.get(name)
        if handle is None:
            return
        flip = name in _FLIP_CAMERAS
        if handle["type"] == "zed":
            self._zed_capture_loop(name, handle, flip)
        else:
            self._realsense_capture_loop(name, handle, flip)

    def _zed_capture_loop(self, name: str, handle: dict, flip: bool) -> None:
        import pyzed.sl as sl

        cam = handle["camera"]
        image_mat = handle["image_mat"]
        depth_mat = handle["depth_mat"]
        right_mat = handle["right_mat"]
        use_learned_stereo = (
            self._stereo_method in ("ffs", "tri_stereo")
            and self._ffs_predictor is not None
        )
        runtime = sl.RuntimeParameters(
            confidence_threshold=99, texture_confidence_threshold=100
        )
        while self._running:
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(image_mat, sl.VIEW.LEFT)
                # ZED returns BGRA; drop alpha channel → BGR.
                color_bgr = image_mat.get_data()[:, :, :3].copy()

                if use_learned_stereo:
                    cam.retrieve_image(right_mat, sl.VIEW.RIGHT)
                    right_bgr = right_mat.get_data()[:, :, :3].copy()
                    if flip:
                        color_bgr = cv2.rotate(color_bgr, cv2.ROTATE_180)
                        right_bgr = cv2.rotate(right_bgr, cv2.ROTATE_180)
                    # Stereo depth models use BGR (OpenCV convention).
                    with handle["stereo_lock"]:
                        handle["latest_left"] = color_bgr
                        handle["latest_right"] = right_bgr
                        handle["stereo_seq"] += 1
                elif not self._no_depth:
                    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                    raw = depth_mat.get_data().copy()
                    depth = np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)
                    if flip:
                        color_bgr = cv2.rotate(color_bgr, cv2.ROTATE_180)
                        depth = cv2.rotate(depth, cv2.ROTATE_180)
                    if name in self.depths:
                        self.update_depth(name, depth)
                else:
                    if flip:
                        color_bgr = cv2.rotate(color_bgr, cv2.ROTATE_180)

                # Serve RGB to the policy.
                self.update_image(name, color_bgr[..., ::-1].copy())
                with self._cam_ts_locks[name]:
                    self._cam_arrival_ts_ns[name] = time.monotonic_ns()

    def _realsense_capture_loop(self, name: str, handle: dict, flip: bool) -> None:
        pipeline = handle["pipeline"]
        align = handle["align"]
        depth_scale = handle["depth_scale"]
        while self._running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=500)
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color_bgr = np.asanyarray(color_frame.get_data())  # BGR uint8
                depth = (np.asanyarray(depth_frame.get_data()) * depth_scale).astype(
                    np.float32
                )
                if flip:
                    color_bgr = cv2.rotate(color_bgr, cv2.ROTATE_180)
                    depth = cv2.rotate(depth, cv2.ROTATE_180)
                # Serve RGB to the policy.
                self.update_image(name, color_bgr[..., ::-1].copy())
                if name in self.depths:
                    self.update_depth(name, depth)
                with self._cam_ts_locks[name]:
                    self._cam_arrival_ts_ns[name] = time.monotonic_ns()
            except RuntimeError:
                time.sleep(0.01)

    # -------------------------------------------------------------------------
    # Learned stereo depth inference (single GPU thread for all cameras)
    # -------------------------------------------------------------------------

    def _depth_inference_loop(self) -> None:
        """Run learned stereo depth inference for all ZED cameras in one thread.

        Camera loops store the latest left/right frame pair in the handle dict.
        This thread picks up new pairs (detected via stereo_seq), runs inference
        sequentially across cameras, and updates the depth buffers.  Running in
        a single thread avoids GPU contention and ensures the model is loaded
        exactly once.
        """
        while self._running:
            any_new = False
            for name, handle in self._cam_handles.items():
                if handle.get("type") != "zed":
                    continue
                with handle["stereo_lock"]:
                    left = handle["latest_left"]
                    right = handle["latest_right"]
                    seq = handle["stereo_seq"]
                    last_seq = handle["last_depth_seq"]
                if left is None or seq == last_seq:
                    continue
                any_new = True
                fx, baseline = self._stereo_calib.get(name, (0.0, 0.0))
                try:
                    depth = self._ffs_predictor.predict(left, right, fx, baseline)
                    if name in self.depths:
                        self.update_depth(name, depth)
                    with handle["stereo_lock"]:
                        handle["last_depth_seq"] = seq
                except Exception as e:
                    print(f"Depth inference error ({name}): {e}")
            if not any_new:
                time.sleep(0.005)

    # -------------------------------------------------------------------------
    # Proprioception
    # -------------------------------------------------------------------------

    def _proprio_loop(self, name: str) -> None:
        """Read robot joint state at ~100 Hz, push to the buffer, and record history."""
        while self._running:
            try:
                obs_all = self._robot.get_all_observations()
                ts = time.monotonic_ns()
                val: Optional[np.ndarray] = None

                if name == "follower_r_joint_pos" and "follower_r" in obs_all:
                    obs = obs_all["follower_r"]
                    val = np.concatenate(
                        [obs["joint_pos"], obs["gripper_pos"].reshape(1)]
                    )
                elif name == "follower_r_joint_vel" and "follower_r" in obs_all:
                    vel = obs_all["follower_r"]["joint_vel"]
                    val = np.concatenate([vel, [0.0]])  # pad gripper vel
                elif name == "follower_l_joint_pos" and "follower_l" in obs_all:
                    obs = obs_all["follower_l"]
                    val = np.concatenate(
                        [obs["joint_pos"], obs["gripper_pos"].reshape(1)]
                    )
                elif name == "follower_l_joint_vel" and "follower_l" in obs_all:
                    vel = obs_all["follower_l"]["joint_vel"]
                    val = np.concatenate([vel, [0.0]])  # pad gripper vel

                if val is not None:
                    self.update_proprio(name, val)
                    with self._proprio_history_locks[name]:
                        self._proprio_history[name].append((ts, val.copy()))

            except Exception as e:
                print(f"Proprio read error ({name}): {e}")
            time.sleep(1 / 100)

    # -------------------------------------------------------------------------
    # E-stop monitor
    # -------------------------------------------------------------------------

    def _estop_monitor(self) -> None:
        """Background thread: shut down the policy server when the footpedal fires.

        The footpedal soft_pause() holds all arms for 5 s then sets
        session_estop_requested.  This thread detects that flag and calls
        close() + os._exit(0) so the asyncio event loop is also terminated.
        """
        while self._running:
            if self._robot.session_estop_requested:
                print("\n[FootPedal] E-stop triggered — shutting down policy server.")
                self.close()
                os._exit(0)
            time.sleep(0.1)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Stop capture threads and release cameras and robot connections."""
        self._running = False
        time.sleep(0.1)
        self._robot.shutdown()
        for handle in self._cam_handles.values():
            try:
                if handle["type"] == "zed":
                    handle["camera"].close()
                else:
                    handle["pipeline"].stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_server(
    camera_config_file: str = "",
    calibration_file: str = "",
    host: str = "0.0.0.0",
    port: int = 8765,
    stereo_method: str = "zed",
    ffs_scale: float = 1.0,
    ffs_iters: int = 8,
    tri_stereo_variant: str = "c64",
    max_joint_delta: float = _DEFAULT_MAX_JOINT_DELTA,
    action_type: str = "ee_pose",
    no_depth: bool = False,
) -> None:
    """Start the Raiden chiral policy server."""
    from raiden._config import CALIBRATION_FILE, CAMERA_CONFIG

    server = RaidenPolicyServer(
        camera_config_file=camera_config_file or CAMERA_CONFIG,
        calibration_file=calibration_file or CALIBRATION_FILE,
        host=host,
        port=port,
        stereo_method=stereo_method,
        ffs_scale=ffs_scale,
        ffs_iters=ffs_iters,
        tri_stereo_variant=tri_stereo_variant,
        max_joint_delta=max_joint_delta,
        action_type=action_type,
        no_depth=no_depth,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.close()
