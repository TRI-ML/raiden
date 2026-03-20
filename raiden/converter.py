"""Convert raw camera recordings to UnifiedDataset format.

Input:  a recording directory produced by ``rd record``.
Output: UnifiedDataset layout inside the same directory.

Usage::

    rd convert ./data/test_2/test_task_2_20260223_162246

Output layout::

    <recording_dir>/
        split_all.json
        metadata_shared.json
        calibration_results.json      # copied from the first recording dir
        0000/
            metadata.json
            rgb/
                scene_camera/
                    0000000000.jpg      # JPEG quality >= 90
                    0000000001.jpg
                    ...
                left_wrist_camera/
                    ...
            depth/
                scene_camera/
                    0000000000.npz      # np.uint16, millimetres
                    ...
            lowdim/
                scene_camera/
                    0000000000.npz      # intrinsics, extrinsics, action, language
                    0000000001.npz
                    ...
                left_wrist_camera/
                    ...
                right_wrist_camera/
                    ...
        cameras/                      # raw SVO2/bag files kept
        metadata.json                 # original recording metadata (updated: converted=true)
        robot_data.npz
"""

import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

_SEQUENCE_NAME = "0000"
_JPG_QUALITY = 90

# Cameras whose images are physically mounted upside-down and need a 180° correction.
_FLIP_CAMERAS = {"right_wrist_camera"}

# Wrist cameras: map camera name → robot_data key for the corresponding follower arm.
_WRIST_CAMERA_JOINT_KEYS: Dict[str, str] = {
    "left_wrist_camera": "follower_l_joint_pos",
    "right_wrist_camera": "follower_r_joint_pos",
}

# Lazily-loaded kinematics instance (MuJoCo FK for YAM arm).
_kinematics: Any = None


def _get_kinematics() -> Any:
    global _kinematics
    if _kinematics is None:
        from i2rt.robots.kinematics import Kinematics

        from raiden._xml_paths import get_yam_4310_linear_xml_path

        _kinematics = Kinematics(get_yam_4310_linear_xml_path(), "grasp_site")
    return _kinematics


# 4x4 homogeneous matrix for 180° rotation around the optical (Z) axis.
# Corrects both extrinsics and principal point when a camera is mounted upside-down.
_R_FLIP_180 = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Frame extraction helpers
# ---------------------------------------------------------------------------


def _count_svo2_frames(svo_path: Path) -> int:
    """Return the total frame count of an SVO2 file without extracting any frames."""
    from raiden.cameras.zed import ZedCamera

    # Depth is never needed for counting; always use DEPTH_MODE.NONE to avoid
    # wasting GPU memory (especially important when FFS will run afterwards).
    camera = ZedCamera.from_svo(svo_path.stem, svo_path, compute_sdk_depth=False)
    n = camera.get_total_frames()
    camera.close()
    return n


def _extract_svo2_synchronized(
    svo_paths: List[Path],
    names: List[str],
    rgb_dirs: List[Path],
    depth_dirs: List[Path],
    flips: List[bool],
    max_frames: Optional[int] = None,
    sync_threshold_ns: int = 16_666_667,  # half a frame at 30 fps
    stereo_method: str = "zed",
    ffs_scale: float = 1.0,
    ffs_iters: int = 8,
) -> Dict[str, Tuple[np.ndarray, Optional[dict]]]:
    """Extract frames from multiple SVO2 files with cross-camera temporal alignment.

    On each output frame slot the algorithm selects the frame from every camera
    whose timestamp is closest to the latest camera in the group, skipping any
    frames that lag by more than *sync_threshold_ns*.  This guarantees that for
    any given output index N the timestamps across all cameras are within
    ``sync_threshold_ns`` of each other.

    Per-camera timestamps (int64, nanoseconds) are saved alongside images as
    ``rgb_dir/timestamps.npy`` for use during lowdim construction.

    Returns
    -------
    dict mapping camera name → (timestamps_ns np.ndarray, camera_info dict or None)
    """
    from raiden.cameras.zed import ZedCamera

    for d in rgb_dirs + depth_dirs:
        d.mkdir(parents=True, exist_ok=True)

    use_ffs = stereo_method == "ffs"

    # Lazily create a shared FFS predictor (one instance, GPU-loaded once).
    ffs_predictor = None
    if use_ffs:
        from raiden.depth.ffs import FFSDepthPredictor, FFSTrtDepthPredictor

        if FFSTrtDepthPredictor.engines_available():
            ffs_predictor = FFSTrtDepthPredictor()
        else:
            ffs_predictor = FFSDepthPredictor(scale=ffs_scale, iters=ffs_iters)

    # Open all cameras.
    cams: Dict[str, ZedCamera] = {
        name: ZedCamera.from_svo(name, svo_path, compute_sdk_depth=not use_ffs)
        for name, svo_path in zip(names, svo_paths)
    }

    # Cache per-camera stereo calibration for FFS.
    stereo_calib: Dict[str, Tuple[float, float]] = {}
    if use_ffs:
        for name, cam in cams.items():
            stereo_calib[name] = cam.get_stereo_calib()
    total_frames = {name: cam.get_total_frames() for name, cam in cams.items()}
    print(f"  Frames per camera: { {n: total_frames[n] for n in names} }")

    rgb_dir_map = dict(zip(names, rgb_dirs))
    depth_dir_map = dict(zip(names, depth_dirs))
    flip_map = dict(zip(names, flips))

    timestamps: Dict[str, List[int]] = {name: [] for name in names}
    frame_idx = 0

    # Initial grab.
    active = {name: cam.grab() for name, cam in cams.items()}

    # Each ZED camera has an independent hardware clock, so absolute timestamps
    # cannot be compared across cameras.  Record each camera's first-frame
    # timestamp and use elapsed time (relative to that origin) for alignment.
    first_ts: Dict[str, int] = {
        name: (cam.get_frame_timestamp_ns() if active[name] else 0)
        for name, cam in cams.items()
    }

    ref_total = max(total_frames.values()) if total_frames else 0
    pbar = tqdm(
        total=ref_total if max_frames is None else min(ref_total, max_frames),
        unit="frame",
        desc="  extracting",
        dynamic_ncols=True,
    )

    while all(active.values()):
        if max_frames is not None and frame_idx >= max_frames:
            break

        # Elapsed nanoseconds since each camera's first frame.
        ts = {
            name: cam.get_frame_timestamp_ns() - first_ts[name]
            for name, cam in cams.items()
        }
        ref_ts = max(ts.values())

        # Advance any camera whose current frame is too far behind the latest.
        advanced = False
        for name, cam in cams.items():
            while active[name] and ref_ts - ts[name] > sync_threshold_ns:
                active[name] = cam.grab()
                if active[name]:
                    ts[name] = cam.get_frame_timestamp_ns() - first_ts[name]
                advanced = True

        if not all(active.values()):
            break
        if advanced:
            # Re-evaluate after advancing (another camera may now be the latest).
            continue

        # All cameras are within threshold — save this synchronized frame.
        for name, cam in cams.items():
            frame = cam.get_frame()
            flip = flip_map[name]

            color = cv2.rotate(frame.color, cv2.ROTATE_180) if flip else frame.color
            cv2.imwrite(
                str(rgb_dir_map[name] / f"{frame_idx:010d}.jpg"),
                color,
                [cv2.IMWRITE_JPEG_QUALITY, _JPG_QUALITY],
            )

            if use_ffs:
                right = cam.get_right_color()
                fx, baseline = stereo_calib[name]
                depth_m = ffs_predictor.predict(frame.color, right, fx, baseline)
                if flip:
                    depth_m = cv2.rotate(depth_m, cv2.ROTATE_180)
                depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
            else:
                depth_mm = (frame.depth * 1000.0).clip(0, 65535).astype(np.uint16)
                if flip:
                    depth_mm = cv2.rotate(depth_mm, cv2.ROTATE_180)
            np.savez_compressed(
                str(depth_dir_map[name] / f"{frame_idx:010d}.npz"), depth=depth_mm
            )

            timestamps[name].append(frame.timestamp_ns)

        frame_idx += 1
        pbar.update(1)

        if use_ffs and frame_idx % 10 == 0 and ffs_predictor._n_calls > 0:
            avg_inf = ffs_predictor._t_inference / ffs_predictor._n_calls * 1000
            pbar.set_postfix(inf_ms=f"{avg_inf:.0f}", refresh=False)

        # Advance all cameras for the next slot.
        active = {name: cam.grab() for name, cam in cams.items()}

    pbar.close()
    print(f"    {frame_idx} synchronized frames extracted")
    if use_ffs and ffs_predictor._n_calls > 0:
        print(f"  FFS timing: {ffs_predictor.timing_summary()}")

    # Collect camera info and persist per-camera timestamps.
    results: Dict[str, Tuple[np.ndarray, Optional[dict]]] = {}
    for name, cam in cams.items():
        camera_info = None
        try:
            camera_info = cam.get_camera_info()
        except Exception:
            pass

        ts_arr = np.array(timestamps[name], dtype=np.int64)
        np.save(str(rgb_dir_map[name] / "timestamps.npy"), ts_arr)

        cam.close()
        results[name] = (ts_arr, camera_info)

    return results


def _extract_bag(
    bag_path: Path,
    rgb_dir: Path,
    depth_dir: Path,
    flip: bool = False,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[dict]]:
    """Extract color and depth frames from a RealSense .bag file.

    Returns (timestamps_ns, camera_info).  timestamps_ns is an int64 array of
    per-frame wall-clock timestamps (nanoseconds since Unix epoch) saved by the
    RealSense SDK when global_time_enabled is set.  The array is also written to
    ``rgb_dir/timestamps.npy`` for caching across re-conversions.
    """
    try:
        from raiden.cameras.realsense import RealSenseCamera
    except ImportError:
        print("  pyrealsense2 not installed – skipping .bag conversion")
        return np.array([], dtype=np.int64), None

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Opening {bag_path.name} ...")
    camera = RealSenseCamera.from_bag(bag_path.stem, bag_path)

    timestamps: List[int] = []
    idx = 0

    while camera.grab():
        if max_frames is not None and idx >= max_frames:
            break
        frame = camera.get_frame()

        color = cv2.rotate(frame.color, cv2.ROTATE_180) if flip else frame.color
        cv2.imwrite(
            str(rgb_dir / f"{idx:010d}.jpg"),
            color,
            [cv2.IMWRITE_JPEG_QUALITY, _JPG_QUALITY],
        )

        depth_mm = (frame.depth * 1000.0).clip(0, 65535).astype(np.uint16)
        if flip:
            depth_mm = cv2.rotate(depth_mm, cv2.ROTATE_180)
        np.savez_compressed(str(depth_dir / f"{idx:010d}.npz"), depth=depth_mm)

        timestamps.append(frame.timestamp_ns)
        idx += 1

        if idx % 100 == 0:
            print(f"    {idx} frames", end="\r", flush=True)

    print(f"    {idx} frames extracted          ")

    ts_arr = np.array(timestamps, dtype=np.int64)
    if len(ts_arr) > 1:
        duration_s = (int(ts_arr[-1]) - int(ts_arr[0])) / 1e9
        actual_fps = (len(ts_arr) - 1) / duration_s if duration_s > 0 else 0.0
        print(f"    Actual FPS: {actual_fps:.1f}  duration: {duration_s:.2f}s")
        if abs(actual_fps - 30.0) > 3.0:
            print(
                f"    WARNING: FPS {actual_fps:.1f} deviates from expected 30 fps — "
                "check camera stream configuration"
            )

    camera_info = None
    try:
        camera_info = camera.get_camera_info()
    except Exception:
        pass

    camera.close()
    np.save(str(rgb_dir / "timestamps.npy"), ts_arr)
    return ts_arr, camera_info


# ---------------------------------------------------------------------------
# Cross-camera temporal alignment
# ---------------------------------------------------------------------------


def _align_cameras_by_timestamp(
    seq_dir: Path,
    cam_timestamps: Dict[str, Optional[np.ndarray]],
    frame_counts: Dict[str, int],
    camera_start_times_ns: Optional[Dict[str, int]] = None,
    camera_fps: float = 30.0,
) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, int]]:
    """Trim per-camera frames to their overlapping recording window.

    Strategy (tried in order):

    1. **Wall-clock timestamp alignment** — if every camera's ``timestamps.npy``
       contains Unix-epoch nanoseconds (> year 2020), find the common time range
       and trim each camera to it.  This is the most accurate method.

    2. **Recording start-time alignment** — fall back to ``camera_start_times_ns``
       from ``metadata.json`` (wall-clock time recorded just before each
       ``camera.start_recording()`` call).  The per-camera start offset in
       frames is ``(t_start[cam] - t_start_min) * fps / 1e9``.  Less accurate
       than per-frame timestamps but works even when bag timestamps are not
       wall-clock (e.g. RealSense hardware clock).

    Renames on-disk jpg/npz files so that frame 0 of every camera corresponds
    to the same point in time.

    Returns updated (cam_timestamps, frame_counts).
    """
    # Unix timestamp for 2020-01-01 in nanoseconds — anything below this is
    # almost certainly a hardware-relative counter, not wall-clock.
    _WALL_CLOCK_MIN_NS = 1_577_836_800_000_000_000

    # --- Strategy 1: per-frame wall-clock timestamps ----------------------
    wall_ts = {
        name: ts
        for name, ts in cam_timestamps.items()
        if ts is not None and len(ts) > 0 and int(ts[0]) > _WALL_CLOCK_MIN_NS
    }

    if len(wall_ts) >= 2:
        # At least two cameras have wall-clock timestamps — align all of them.
        t_start = max(int(ts[0]) for ts in wall_ts.values())
        t_end = min(int(ts[-1]) for ts in wall_ts.values())

        if t_start < t_end:
            new_timestamps = dict(cam_timestamps)
            new_frame_counts = dict(frame_counts)

            for name, ts in wall_ts.items():
                start_idx = int(np.searchsorted(ts, t_start))
                end_idx = int(np.searchsorted(ts, t_end, side="right"))
                _apply_camera_trim(seq_dir, name, start_idx, end_idx, ts)
                new_timestamps[name] = ts[start_idx:end_idx]
                new_frame_counts[name] = end_idx - start_idx

            print(
                f"  Timestamp alignment: common window {(t_end - t_start) / 1e9:.2f}s"
            )
            return new_timestamps, new_frame_counts

    # --- Strategy 2: recording start-time alignment -----------------------
    if camera_start_times_ns and len(camera_start_times_ns) >= 2:
        t_min = min(camera_start_times_ns.values())
        offsets = {
            name: int(round((t - t_min) * camera_fps / 1e9))
            for name, t in camera_start_times_ns.items()
            if name in frame_counts
        }
        if any(off > 0 for off in offsets.values()):
            new_timestamps = dict(cam_timestamps)
            new_frame_counts = dict(frame_counts)

            for name, start_idx in offsets.items():
                if start_idx == 0:
                    continue
                n_total = frame_counts[name]
                end_idx = n_total  # keep all frames after the offset
                ts = cam_timestamps.get(name)
                _apply_camera_trim(seq_dir, name, start_idx, end_idx, ts)
                new_timestamps[name] = ts[start_idx:end_idx] if ts is not None else None
                new_frame_counts[name] = end_idx - start_idx

            offsets_str = ", ".join(
                f"{n}={off}fr" for n, off in offsets.items() if off > 0
            )
            print(f"  Start-time alignment: skipped {offsets_str}")
            return new_timestamps, new_frame_counts

    return cam_timestamps, frame_counts


def _apply_camera_trim(
    seq_dir: Path,
    name: str,
    start_idx: int,
    end_idx: int,
    ts: Optional[np.ndarray],
) -> None:
    """Rename frame files and update timestamps.npy for one camera."""
    n_new = end_idx - start_idx
    if start_idx > 0:
        rgb_dir = seq_dir / "rgb" / name
        depth_dir = seq_dir / "depth" / name
        # Rename in forward order: source indices (start_idx+i) are always
        # higher than destination indices (i), so no conflicts.
        for i in range(n_new):
            src = start_idx + i
            for d, ext in ((rgb_dir, ".jpg"), (depth_dir, ".npz")):
                src_f = d / f"{src:010d}{ext}"
                dst_f = d / f"{i:010d}{ext}"
                if src_f.exists():
                    src_f.rename(dst_f)
        print(
            f"  Aligned {name}: skipped {start_idx} leading frame(s) "
            f"(~{start_idx / 30:.2f}s)"
        )
    if ts is not None:
        np.save(
            str(seq_dir / "rgb" / name / "timestamps.npy"),
            ts[start_idx:end_idx],
        )


# ---------------------------------------------------------------------------
# Lowdim builder
# ---------------------------------------------------------------------------


def _build_lowdim(
    seq_dir: Path,
    cameras: List[str],
    n_frames: int,
    camera_infos: Dict[str, Optional[dict]],
    calib: Optional[dict],
    robot_data: Optional[Dict[str, np.ndarray]],
    rec_meta: dict,
    flip_cameras: set,
    right_base_to_left_base: Optional[np.ndarray],
    cam_timestamps: Dict[str, Optional[np.ndarray]],
) -> None:
    """Write seq_dir/lowdim.npz with all cameras' intrinsics/extrinsics plus joints, action, language.

    Output keys in lowdim.npz
    --------------------------
    Per-frame keys in each lowdim/<frame>.npz
    ------------------------------------------
    ``intrinsics``           dict[camera_name → (3, 3) float32]  camera matrix K per camera.
    ``extrinsics``           dict[camera_name → (4, 4) float32]  cam-to-left_arm_base.
                             Wrist cameras: computed per-frame via FK + hand-eye calibration.
                             Scene cameras: static calibrated extrinsics.
    ``joints``               (14,) float32  follower joint positions at this frame:
                             [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
    ``action``               (26,) float32  FK EE poses computed from commanded joint positions
                             (follower_*_joint_cmd), in the left_arm_base frame:
                             [l_pos(3), l_rot9(9), l_gripper(1), r_pos(3), r_rot9(9), r_gripper(1)].
    ``language_task``        str  task name.
    ``language_prompt``      str  task instruction.
    """
    _WALL_CLOCK_MIN_NS = 1_577_836_800_000_000_000

    # ── reference timestamp grid for interpolating robot data ─────────────
    # Prefer wall-clock timestamps; fall back to any available camera timestamps.
    ref_ts: Optional[np.ndarray] = None
    for name in cameras:
        ts = cam_timestamps.get(name)
        if ts is not None and len(ts) == n_frames and int(ts[0]) > _WALL_CLOCK_MIN_NS:
            ref_ts = ts.astype(np.float64)
            break
    if ref_ts is None:
        for name in cameras:
            ts = cam_timestamps.get(name)
            if ts is not None and len(ts) == n_frames:
                ref_ts = ts.astype(np.float64)
                break

    robot_ts: Optional[np.ndarray] = None
    if robot_data is not None and n_frames > 0:
        robot_ts_raw = robot_data.get("timestamps")
        if (
            ref_ts is not None
            and robot_ts_raw is not None
            and robot_ts_raw.dtype == np.int64
        ):
            robot_ts = robot_ts_raw.astype(np.float64)
        else:
            # Legacy: uniform linspace over the recording duration.
            duration = rec_meta.get("duration_s", 1.0)
            ref_ts = np.linspace(0.0, duration, n_frames, endpoint=False)
            if robot_ts_raw is not None:
                robot_ts = robot_ts_raw.astype(np.float64)

    def interp_to_cam(key: str) -> Optional[np.ndarray]:
        if robot_data is None or robot_ts is None or ref_ts is None:
            return None
        arr = robot_data.get(key)
        if arr is None:
            return None
        if arr.ndim == 1:
            arr = arr[:, None]
        return np.stack(
            [np.interp(ref_ts, robot_ts, arr[:, d]) for d in range(arr.shape[1])],
            axis=1,
        ).astype(np.float32)

    # ── intrinsics: intrinsics_<camera> → (4,) per camera ────────────────
    intrinsics: Dict[str, np.ndarray] = {}
    for name in cameras:
        info = camera_infos.get(name)
        if not info:
            continue
        flip = name in flip_cameras
        cx, cy = info["cx"], info["cy"]
        if flip:
            cx = (info["width"] - 1) - cx
            cy = (info["height"] - 1) - cy
        intrinsics[name] = np.array(
            [[info["fx"], 0.0, cx], [0.0, info["fy"], cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    # ── extrinsics: extrinsics_<camera> → (N, 4, 4) per camera ──────────
    kin = None  # lazily loaded
    extrinsics: Dict[str, np.ndarray] = {}
    for name in cameras:
        flip = name in flip_cameras
        static_ext = np.eye(4, dtype=np.float32)
        he = None
        if calib and "cameras" in calib and name in calib["cameras"]:
            cam_calib = calib["cameras"][name]
            if "hand_eye_calibration" in cam_calib:
                he = cam_calib["hand_eye_calibration"]
            elif "extrinsics" in cam_calib:
                ext = cam_calib["extrinsics"]
                if (
                    ext.get("success")
                    and "rotation_matrix" in ext
                    and "translation_vector" in ext
                ):
                    static_ext[:3, :3] = np.array(
                        ext["rotation_matrix"], dtype=np.float32
                    )
                    static_ext[:3, 3] = np.array(
                        ext["translation_vector"], dtype=np.float32
                    ).flatten()

        per_frame_ext: Optional[np.ndarray] = None
        joint_key = _WRIST_CAMERA_JOINT_KEYS.get(name)
        if (
            joint_key
            and he is not None
            and he.get("success")
            and robot_data is not None
            and robot_ts is not None
            and ref_ts is not None
        ):
            T_cam2ee = np.eye(4, dtype=np.float32)
            T_cam2ee[:3, :3] = np.array(he["rotation_matrix"], dtype=np.float32)
            T_cam2ee[:3, 3] = np.array(
                he["translation_vector"], dtype=np.float32
            ).flatten()
            if flip:
                T_cam2ee = T_cam2ee @ _R_FLIP_180

            arm_joints_raw = robot_data.get(joint_key)
            if arm_joints_raw is not None:
                arm_joints = np.stack(
                    [
                        np.interp(ref_ts, robot_ts, arm_joints_raw[:, d])
                        for d in range(arm_joints_raw.shape[1])
                    ],
                    axis=1,
                ).astype(np.float32)
                if kin is None:
                    kin = _get_kinematics()
                is_right = joint_key == "follower_r_joint_pos"
                per_frame_ext = np.stack(
                    [
                        (
                            right_base_to_left_base
                            @ kin.fk(arm_joints[i]).astype(np.float32)
                            @ T_cam2ee
                        )
                        if (is_right and right_base_to_left_base is not None)
                        else (kin.fk(arm_joints[i]).astype(np.float32) @ T_cam2ee)
                        for i in range(n_frames)
                    ]
                )  # (N, 4, 4)

        extrinsics[name] = (
            per_frame_ext
            if per_frame_ext is not None
            else np.tile(static_ext[None], (n_frames, 1, 1))
        )

    # ── joints: (N, 14) from follower_{l,r}_joint_pos_7d ─────────────────
    joints_parts = [
        p
        for p in (
            interp_to_cam("follower_l_joint_pos_7d"),
            interp_to_cam("follower_r_joint_pos_7d"),
        )
        if p is not None
    ]
    joints: Optional[np.ndarray] = (
        np.concatenate(joints_parts, axis=1) if joints_parts else None
    )

    # ── action: (N, 26) FK EE poses + gripper ────────────────────────────
    # Layout: [l_pos(3), l_rot9(9), l_grip(1), r_pos(3), r_rot9(9), r_grip(1)]
    action: Optional[np.ndarray] = None
    if robot_data is not None and robot_ts is not None and ref_ts is not None:
        if kin is None:
            kin = _get_kinematics()
        action_parts = []

        # follower_*_joint_cmd is 7-DOF: arm joints (6) + gripper (1)
        l_cmd = interp_to_cam("follower_l_joint_cmd")
        if l_cmd is not None:
            l_poses = np.stack(
                [
                    np.concatenate([T[:3, 3], T[:3, :3].flatten()]).astype(np.float32)
                    for T in (kin.fk(l_cmd[i, :6]) for i in range(n_frames))
                ]
            )  # (N, 12)
            action_parts.append(l_poses)
            action_parts.append(l_cmd[:, 6:7])  # gripper

        r_cmd = interp_to_cam("follower_r_joint_cmd")
        if r_cmd is not None:
            r_poses = np.stack(
                [
                    np.concatenate([T[:3, 3], T[:3, :3].flatten()]).astype(np.float32)
                    for T in (kin.fk(r_cmd[i, :6]) for i in range(n_frames))
                ]
            )  # (N, 12)  — in right-arm base frame
            action_parts.append(r_poses)
            action_parts.append(r_cmd[:, 6:7])  # gripper

        if action_parts:
            action = np.concatenate(action_parts, axis=1)

    # ── action_joints: (N, 14) commanded joint positions ─────────────────
    # Layout: [l_arm(6), l_gripper(1), r_arm(6), r_gripper(1)]
    action_joints_parts = [
        p
        for p in (
            interp_to_cam("follower_l_joint_cmd"),
            interp_to_cam("follower_r_joint_cmd"),
        )
        if p is not None
    ]
    action_joints: Optional[np.ndarray] = (
        np.concatenate(action_joints_parts, axis=1) if action_joints_parts else None
    )

    # ── language ──────────────────────────────────────────────────────────
    language_task = np.array(rec_meta.get("task_name", ""), dtype=object)
    language_prompt = np.array(rec_meta.get("task_instruction", ""), dtype=object)

    # ── write per-frame files into lowdim/ ────────────────────────────────
    lowdim_dir = seq_dir / "lowdim"
    lowdim_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        frame_data: Dict[str, Any] = {
            "intrinsics": intrinsics,
            "extrinsics": {name: ext_arr[i] for name, ext_arr in extrinsics.items()},
        }
        if joints is not None:
            frame_data["joints"] = joints[i]
        if action is not None:
            frame_data["action"] = action[i]
        if action_joints is not None:
            frame_data["action_joints"] = action_joints[i]
        frame_data["language_task"] = language_task
        frame_data["language_prompt"] = language_prompt
        with open(lowdim_dir / f"{i:010d}.pkl", "wb") as f:
            pickle.dump(frame_data, f)


# ---------------------------------------------------------------------------
# Dataset-level metadata helpers
# ---------------------------------------------------------------------------


def _build_sequence_metadata(
    seq_dir: Path,
    cameras: List[str],
    frame_counts: Dict[str, int],
    rec_meta: dict,
    camera_infos: Dict[str, Optional[dict]],
) -> None:
    """Write metadata.json inside the sequence directory."""
    resolutions = {
        cam: [info.get("height"), info.get("width")]
        for cam, info in camera_infos.items()
        if info
    }
    unique_res = list({tuple(v) for v in resolutions.values()})
    if len(unique_res) == 1:
        resolution: object = list(unique_res[0])
    elif unique_res:
        resolution = {cam: v for cam, v in resolutions.items()}
    else:
        resolution = None

    meta = {
        "info": {
            "name": rec_meta.get("task_name", ""),
            "raw_id": rec_meta.get("timestamp", ""),
            "tags": ["robotics"],
        },
        "labels": ["rgb", "depth", "action", "language"],
        "cameras": cameras,
        "resolution": resolution,
        "framerate": rec_meta.get("camera_fps", 30),
        "language": {
            "task": rec_meta.get("task_name", ""),
            "prompt": [rec_meta.get("task_instruction", "")],
        },
        "num_frames": max(frame_counts.values()) if frame_counts else 0,
        "rgb": {"extension": "jpg"},
        "depth": {"extension": "npz", "sparse": False, "metric": True},
        "intrinsics": {"model": "pinhole"},
        "extrinsics": {"transform": "cam2world", "metric": True},
        "action": {"format": "joint_cmd", "dims": 14},
    }

    with open(seq_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def _build_split(rec_path: Path, frame_counts: Dict[str, int]) -> None:
    n_frames = max(frame_counts.values()) if frame_counts else 0
    split = {
        "filters": {},
        "size": {"n_seqs": 1, "n_samples": 1, "n_frames": n_frames},
        "files": {_SEQUENCE_NAME: n_frames},
    }
    with open(rec_path / "split_all.json", "w") as f:
        json.dump(split, f, indent=2)


# ---------------------------------------------------------------------------
# Interactive task selection
# ---------------------------------------------------------------------------


def select_tasks(data_dir: str = "data") -> List[str]:
    """Use fzf to select one or more task directories (Tab to multi-select)."""
    base = Path(data_dir) / "raw"
    task_dirs = sorted(
        d
        for d in base.iterdir()
        if d.is_dir()
        and any((sub / "cameras").exists() for sub in d.iterdir() if sub.is_dir())
    )

    if not task_dirs:
        print(f"No tasks found in {base}")
        sys.exit(1)

    labels = {
        f"{d.name}  ({sum(1 for s in d.iterdir() if s.is_dir() and (s / 'cameras').exists())} recording(s))": d
        for d in task_dirs
    }

    from raiden.utils import fzf_select

    selected = fzf_select(list(labels), prompt="Convert task(s)> ", multi=True)
    return [str(labels[s]) for s in selected]


# ---------------------------------------------------------------------------
# Main convert function
# ---------------------------------------------------------------------------


def convert_recording(
    recording_dir: str,
    episode_dir: Optional[str] = None,
    stereo_method: str = "zed",
    ffs_scale: float = 1.0,
    ffs_iters: int = 8,
) -> Dict[str, int]:
    """Convert a recording directory to UnifiedDataset format.

    When *episode_dir* is provided the sequence data is written there instead
    of ``recording_dir/0000``, and dataset-level files (split_all.json,
    metadata_shared.json, calibration_results.json) are skipped so the caller
    (e.g. :func:`convert_task`) can aggregate them.

    Returns per-camera frame counts, or an empty dict if already converted.
    """
    rec_path = Path(recording_dir)
    if not rec_path.exists():
        print(f"Error: directory not found: {rec_path}")
        sys.exit(1)

    cameras_path = rec_path / "cameras"
    if not cameras_path.exists():
        print(f"Error: no cameras/ sub-directory in {rec_path}")
        sys.exit(1)

    svo2_files = sorted(cameras_path.glob("*.svo2"))
    bag_files = sorted(cameras_path.glob("*.bag"))

    if not svo2_files and not bag_files:
        print("No .svo2 or .bag files found in cameras/")
        sys.exit(1)

    seq_dir = Path(episode_dir) if episode_dir else rec_path / _SEQUENCE_NAME
    if (seq_dir / "rgb").exists():
        print(f"Already converted: {seq_dir}")
        return {}

    seq_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting to UnifiedDataset: {rec_path.name}")
    print(f"  Found {len(svo2_files)} SVO2 file(s), {len(bag_files)} bag file(s)\n")

    # ── load supporting data ──────────────────────────────────────────────
    rec_meta: dict = {}
    meta_path = rec_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            rec_meta = json.load(f)

    robot_data: Optional[Dict[str, np.ndarray]] = None
    robot_path = rec_path / "robot_data.npz"
    if robot_path.exists():
        npz = np.load(robot_path, allow_pickle=False)
        robot_data = {k: npz[k] for k in npz.files}

    calib: Optional[dict] = None
    calib_path = rec_path / "calibration_results.json"
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)

    # "right_base_to_left_base" maps left_arm_base → right_arm_base (despite the name).
    # Invert to get the transform that brings right_arm_base points into left_arm_base.
    T_left_base_from_right_base: Optional[np.ndarray] = None
    if calib and "bimanual_transform" in calib:
        mat = calib["bimanual_transform"].get("right_base_to_left_base")
        if mat is not None:
            T_left_base_from_right_base = np.linalg.inv(np.array(mat, dtype=np.float32))
    if T_left_base_from_right_base is None:
        print(
            "  Warning: bimanual_transform not found in calibration; right wrist extrinsics will be in right_arm_base frame"
        )

    # ── extract frames ────────────────────────────────────────────────────
    frame_counts: Dict[str, int] = {}
    camera_infos: Dict[str, Optional[dict]] = {}
    cam_timestamps: Dict[str, Optional[np.ndarray]] = {}

    # SVO2 cameras are extracted together with cross-camera temporal alignment.
    # Skip only when every SVO2 camera already has its rgb dir AND timestamps.npy
    # (the latter is written by _extract_svo2_synchronized).
    svo2_names = [p.stem for p in svo2_files]
    svo2_all_done = svo2_names and all(
        (seq_dir / "rgb" / name).exists()
        and (seq_dir / "rgb" / name / "timestamps.npy").exists()
        for name in svo2_names
    )

    if svo2_all_done:
        for name in svo2_names:
            rgb_dir = seq_dir / "rgb" / name
            n = len(list(rgb_dir.glob("*.jpg")))
            print(f"  Skipping {name} (already extracted, {n} frames)")
            frame_counts[name] = n
            camera_infos[name] = None
            cam_timestamps[name] = np.load(str(rgb_dir / "timestamps.npy"))
    elif svo2_files:
        # Pre-scan to determine the min frame count cap.
        pre_counts: Dict[str, int] = {}
        for svo_path in svo2_files:
            name = svo_path.stem
            rgb_dir_check = seq_dir / "rgb" / name
            if rgb_dir_check.exists():
                pre_counts[name] = len(list(rgb_dir_check.glob("*.jpg")))
            else:
                print(f"  Pre-scanning {svo_path.name} ...")
                pre_counts[name] = _count_svo2_frames(svo_path)
                print(f"    {pre_counts[name]} frames")
        max_frames_svo2 = min(pre_counts.values()) if pre_counts else None
        if max_frames_svo2 is not None:
            print(
                f"  Capping SVO2 extraction at {max_frames_svo2} frames (min across cameras)\n"
            )

        print(f"  Extracting {len(svo2_files)} SVO2 file(s) in sync ...")
        if stereo_method != "zed":
            print(
                f"  Stereo method: {stereo_method}"
                + (f" (scale={ffs_scale})" if ffs_scale != 1.0 else "")
            )
        sync_results = _extract_svo2_synchronized(
            svo_paths=svo2_files,
            names=svo2_names,
            rgb_dirs=[seq_dir / "rgb" / n for n in svo2_names],
            depth_dirs=[seq_dir / "depth" / n for n in svo2_names],
            flips=[n in _FLIP_CAMERAS for n in svo2_names],
            max_frames=max_frames_svo2,
            stereo_method=stereo_method,
            ffs_scale=ffs_scale,
            ffs_iters=ffs_iters,
        )
        for name, (ts_arr, info) in sync_results.items():
            frame_counts[name] = len(ts_arr)
            camera_infos[name] = info
            cam_timestamps[name] = ts_arr
            print(f"  ✓ {name}: {len(ts_arr)} frames")

    # Bag files are extracted independently then aligned to ZED by timestamp.
    bag_max = min(frame_counts.values()) if frame_counts else None
    rs_offsets = rec_meta.get("realsense_clock_offsets", {})
    for bag_path in bag_files:
        name = bag_path.stem
        rgb_dir = seq_dir / "rgb" / name
        depth_dir = seq_dir / "depth" / name

        if rgb_dir.exists():
            n = len(list(rgb_dir.glob("*.jpg")))
            print(f"  Skipping {name} (already extracted, {n} frames)")
            frame_counts[name] = n
            camera_infos[name] = None
            ts_path = rgb_dir / "timestamps.npy"
            if ts_path.exists():
                ts_arr = np.load(str(ts_path))
                clock_offset = rs_offsets.get(name)
                cam_timestamps[name] = (
                    ts_arr + int(clock_offset) if clock_offset is not None else ts_arr
                )
            else:
                cam_timestamps[name] = None
            continue

        flip = name in _FLIP_CAMERAS
        print(f"  Extracting {bag_path.name}" + (" (flipped)" if flip else ""))
        ts_arr, info = _extract_bag(
            bag_path, rgb_dir, depth_dir, flip=flip, max_frames=bag_max
        )
        frame_counts[name] = len(ts_arr)
        camera_infos[name] = info

        # With global_time_enabled, RealSense timestamps are wall-clock (same as
        # ZED).  Old recordings may carry a clock offset in metadata; apply it
        # for backward compatibility.
        clock_offset = rs_offsets.get(name)
        if len(ts_arr) > 0:
            cam_timestamps[name] = (
                ts_arr + int(clock_offset) if clock_offset is not None else ts_arr
            )
        else:
            cam_timestamps[name] = None
        print(f"  ✓ {name}: {len(ts_arr)} frames\n")

    # Align all cameras to their overlapping recording window, correcting for
    # sequential startup offsets between ZED and RealSense cameras.
    cam_timestamps, frame_counts = _align_cameras_by_timestamp(
        seq_dir,
        cam_timestamps,
        frame_counts,
        camera_start_times_ns=rec_meta.get("camera_start_times_ns"),
    )

    # Trim all cameras to the same (minimum) frame count.
    if frame_counts:
        n_min = min(frame_counts.values())
        frame_counts = {k: n_min for k in frame_counts}
        cam_timestamps = {
            k: (ts[:n_min] if ts is not None else None)
            for k, ts in cam_timestamps.items()
        }

    cameras = list(frame_counts.keys())

    # ── lowdim ────────────────────────────────────────────────────────────
    print("\nBuilding lowdim...")
    _build_lowdim(
        seq_dir=seq_dir,
        cameras=cameras,
        n_frames=n_min,
        camera_infos=camera_infos,
        calib=calib,
        robot_data=robot_data,
        rec_meta=rec_meta,
        flip_cameras=_FLIP_CAMERAS,
        right_base_to_left_base=T_left_base_from_right_base,
        cam_timestamps=cam_timestamps,
    )
    print(f"  ✓ lowdim/ ({n_min} frames)")

    # ── sequence metadata ─────────────────────────────────────────────────
    _build_sequence_metadata(seq_dir, cameras, frame_counts, rec_meta, camera_infos)
    print("  ✓ metadata.json")

    if episode_dir is None:
        # ── dataset-level files ───────────────────────────────────────────
        _build_split(rec_path, frame_counts)
        print("  ✓ split_all.json")

        shutil.copy(seq_dir / "metadata.json", rec_path / "metadata_shared.json")
        print("  ✓ metadata_shared.json")

        if calib_path.exists():
            shutil.copy(calib_path, rec_path / "calibration_results.json")
            print(f"  ✓ calibration_results.json (copied from {calib_path})")
        else:
            print(f"  Warning: calibration_results.json not found at {calib_path}")

    # ── update original metadata ──────────────────────────────────────────
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["converted"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\n✓ UnifiedDataset ready: {seq_dir if episode_dir else rec_path}")
    return frame_counts


def convert_task(
    task_dir: str,
    output_dir: Optional[str] = None,
    stereo_method: str = "zed",
    ffs_scale: float = 1.0,
    ffs_iters: int = 8,
    reconvert: bool = False,
    processed_base: Optional[str] = None,
) -> None:
    """Convert all recordings in a task directory into a single UnifiedDataset.

    Each recording becomes a numbered episode (0000, 0001, …) under *out_base*::

        <output_dir>/<task_name>/
            0000/
            0001/
            ...
            split_all.json
            metadata_shared.json
            calibration_results.json

    *output_dir* defaults to ``<task_parent>/processed_data``.
    """
    task_path = Path(task_dir)
    if not task_path.exists():
        print(f"Error: directory not found: {task_path}")
        sys.exit(1)

    recording_dirs = sorted(
        d for d in task_path.iterdir() if d.is_dir() and (d / "cameras").exists()
    )

    if not recording_dirs:
        print(f"Error: no recording directories found in {task_path}")
        sys.exit(1)

    if output_dir:
        out_base = Path(output_dir) / task_path.name
    elif processed_base:
        out_base = Path(processed_base) / task_path.name
    else:
        out_base = Path("data") / "processed" / task_path.name
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(recording_dirs)} recording(s) in {task_path.name}")
    print(f"Output → {out_base}\n")

    episode_frame_counts: Dict[str, int] = {}

    # Lazy import to avoid hard dependency when DB is not set up
    try:
        from raiden.db.database import get_db as _get_db

        _db = _get_db()
    except Exception:
        _db = None

    # Filter recordings: skip failures/pending, and (unless reconvert) already
    # converted ones.  Recordings with no DB entry are treated as unknown and
    # included so directories recorded before the DB was set up are not dropped.
    success_dirs = []
    skipped = 0
    for rec_dir in recording_dirs:
        status = "unknown"
        already_converted = False
        if _db is not None:
            try:
                demo = _db.get_demonstration_by_raw_path(str(rec_dir))
                if demo is not None:
                    status = demo.get("status", "pending")
                    already_converted = bool(demo.get("converted", False))
            except Exception:
                pass
        if status not in ("success", "unknown"):
            print(f"  Skipping {rec_dir.name} (status={status})")
            skipped += 1
        elif already_converted and not reconvert:
            print(
                f"  Skipping {rec_dir.name} (already converted, use --reconvert to force)"
            )
            skipped += 1
        else:
            success_dirs.append(rec_dir)

    if skipped:
        print(f"\nSkipped {skipped} non-success recording(s).")
    if not success_dirs:
        print("No successful recordings to convert.")
        return

    print(f"Converting {len(success_dirs)} successful recording(s)\n")

    for i, rec_dir in enumerate(success_dirs):
        episode_name = f"{i:04d}"
        ep_dir = out_base / episode_name
        print(f"[{i + 1}/{len(success_dirs)}] {rec_dir.name} → {episode_name}/")
        counts = convert_recording(
            str(rec_dir),
            episode_dir=str(ep_dir),
            stereo_method=stereo_method,
            ffs_scale=ffs_scale,
            ffs_iters=ffs_iters,
        )

        if counts:
            episode_frame_counts[episode_name] = max(counts.values())
        else:
            # Already converted — read frame count from existing metadata.
            meta_file = ep_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    ep_meta = json.load(f)
                episode_frame_counts[episode_name] = ep_meta.get("num_frames", 0)
            else:
                episode_frame_counts[episode_name] = 0

        # Update Demonstration status in DB
        if _db is not None:
            try:
                demo = _db.get_demonstration_by_raw_path(str(rec_dir))
                if demo is not None:
                    _db.update_demonstration(
                        demo["id"],
                        converted=True,
                        converted_data_path=str(ep_dir),
                    )
            except Exception:
                pass

    # ── combined split_all.json ────────────────────────────────────────────
    total_frames = sum(episode_frame_counts.values())
    n_eps = len(episode_frame_counts)
    split = {
        "filters": {},
        "size": {"n_seqs": n_eps, "n_samples": n_eps, "n_frames": total_frames},
        "files": episode_frame_counts,
    }
    with open(out_base / "split_all.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"\n✓ split_all.json ({n_eps} episodes, {total_frames} total frames)")

    # ── shared metadata & calibration ─────────────────────────────────────
    first_ep = next(iter(episode_frame_counts))
    first_meta = out_base / first_ep / "metadata.json"
    if first_meta.exists():
        shutil.copy(first_meta, out_base / "metadata_shared.json")
        print("✓ metadata_shared.json")

    first_calib = recording_dirs[0] / "calibration_results.json"
    if first_calib.exists():
        shutil.copy(first_calib, out_base / "calibration_results.json")
        print("✓ calibration_results.json")

    print(f"\n✓ Task dataset ready: {out_base}")
