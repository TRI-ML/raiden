"""Visualize a converted UnifiedDataset recording using Rerun."""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _viz_depth(depth_m: np.ndarray, percentile: int = 95) -> np.ndarray:
    """Plasma-colorize a depth map using inverse-depth mapping.

    Mirrors ``viz_depth`` / ``viz_inv_depth`` from AnyData:
    - compute inverse depth (1 / depth), masking zeros
    - normalize by the *percentile*-th value of valid pixels
    - apply plasma colormap; invalid pixels stay black

    Returns uint8 H×W×3.
    """
    inv = np.where(depth_m > 0, 1.0 / np.clip(depth_m, 1e-8, None), 0.0)
    valid = inv[inv > 0]
    normalizer = float(np.percentile(valid, percentile)) if len(valid) > 0 else 1.0
    inv_norm = (inv / (normalizer + 1e-6)).clip(0.0, 1.0)
    # cv2 COLORMAP_PLASMA produces BGR uint8
    colored_bgr = cv2.applyColorMap(
        (inv_norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
    )
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    colored_rgb[inv == 0] = 0
    return colored_rgb


def _reconstruct_points(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T_cam2world: np.ndarray,
) -> np.ndarray:
    """Unproject every pixel into world space.

    Returns all (H*W, 3) world-space points (including invalid ones so the
    caller can apply a uniform validity mask that aligns with flattened images).
    """
    H, W = depth_m.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    d = depth_m.reshape(-1)
    pts_cam = np.stack(
        [
            (uu.reshape(-1) - cx) * d / fx,
            (vv.reshape(-1) - cy) * d / fy,
            d,
            np.ones_like(d),
        ],
        axis=1,
    )  # (H*W, 4)
    pts_world = (T_cam2world @ pts_cam.T).T[:, :3]  # (H*W, 3)
    return pts_world


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------


def select_task_and_episode(data_dir: str = "data/processed") -> Tuple[str, str]:
    """Use fzf to select a task then an episode. Returns ``(task_dir_path, episode_name)``."""
    base = Path(data_dir)
    task_dirs = sorted(
        d
        for d in base.iterdir()
        if d.is_dir()
        and any((ep / "metadata.json").exists() for ep in d.iterdir() if ep.is_dir())
    )
    if not task_dirs:
        raise FileNotFoundError(f"No converted tasks found in {base}")

    from raiden.utils import fzf_select

    task_labels = {d.name: d for d in task_dirs}
    chosen_task = fzf_select(list(task_labels), prompt="Visualize task> ")[0]
    task_dir = task_labels[chosen_task]

    episode_dirs = sorted(
        d for d in task_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes found in {task_dir}")

    episode_labels: dict = {}
    for ep in episode_dirs:
        label = ep.name
        with open(ep / "metadata.json") as f:
            ep_meta = json.load(f)
        raw_id = ep_meta.get("info", {}).get("raw_id", "")
        if raw_id:
            label = f"{ep.name}  ({raw_id})"
        episode_labels[label] = ep

    chosen_ep = fzf_select(list(episode_labels), prompt="Select episode> ")[0]
    return str(task_dir), episode_labels[chosen_ep].name


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------


def visualize_recording(
    recording_dir: str,
    episode: str = "0000",
    stride: int = 1,
    image_scale: float = 0.25,
    frustum_scale: float = 0.1,
    web: bool = False,
    web_port: int = 9090,
) -> None:
    """Visualize a converted recording directory using Rerun.

    Mirrors RerunDisplay from AnyData: logs camera frustums (Transform3D +
    Pinhole), RGB images (at entity path for frustum texture and at
    ``{entity}/rgb`` for the 2-D view), plasma-colorized depth maps, and
    colored world-space point clouds.

    Parameters
    ----------
    recording_dir:
        Path to the converted task directory (e.g. ``data/processed/pick_purrito``).
    episode:
        Episode subdirectory name (default: ``"0000"``).
    stride:
        Log every *stride*-th frame (default: ``1``).
    image_scale:
        Uniform downsample factor for images and point clouds (default: ``0.25``).
    frustum_scale:
        ``image_plane_distance`` passed to ``rr.Pinhole``; controls the rendered
        frustum size in the 3-D view (default: ``0.1``).
    web:
        Serve the viewer over HTTP instead of spawning the native desktop app.
        Use this when connecting via SSH tunnel.
    web_port:
        HTTP port for the web viewer (default: ``9090``).
    """
    import rerun as rr

    # When raiden is installed as a uv tool, the venv's bin/ is not on PATH, so
    # rerun-sdk cannot find its own viewer binary via shutil.which("rerun").
    # Prepend the directory that contains the current Python executable — that is
    # the same bin/ where pip/uv installs the `rerun` binary.
    _venv_bin = str(Path(sys.executable).parent)
    if _venv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

    rec_dir = Path(recording_dir)
    ep_dir = rec_dir / episode

    if not ep_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {ep_dir}")

    with open(ep_dir / "metadata.json") as f:
        meta = json.load(f)

    cam_names: list = meta.get("cameras", [])
    task_name: str = meta.get("info", {}).get("name", "unknown")
    prompts = meta.get("language", {}).get("prompt", [""])
    prompt: str = prompts[0] if prompts else ""

    rgb_dir = ep_dir / "rgb"
    counts = [
        len(sorted((rgb_dir / cam).glob("*.jpg")))
        for cam in cam_names
        if (rgb_dir / cam).exists()
    ]
    if not counts:
        raise RuntimeError(f"No RGB frames found under {rgb_dir}")
    n_frames = min(counts)

    print(f"Recording : {rec_dir.name}")
    print(f"Task      : {task_name}")
    print(f"Prompt    : {prompt}")
    print(f"Cameras   : {cam_names}")
    print(f"Frames    : {n_frames}")

    # Load bimanual transform from the first lowdim frame.
    # Right arm action is stored in right-arm base frame; T_left_from_right converts to left-arm base.
    T_left_from_right: np.ndarray = np.eye(4, dtype=np.float64)
    _first_lowdim = ep_dir / "lowdim" / "0000000000.pkl"
    if _first_lowdim.exists():
        with open(_first_lowdim, "rb") as _f:
            _ld0 = pickle.load(_f)
        _t = _ld0.get("T_left_from_right")
        if _t is not None:
            T_left_from_right = np.array(_t, dtype=np.float64)

    if web:
        from urllib.parse import quote

        rr.init("raiden")
        grpc_port = web_port + 1
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(web_port=web_port, open_browser=False)
        viewer_url = f"http://localhost:{web_port}?url={quote(server_uri, safe='')}"
        print(f"\nOpen in browser: {viewer_url}")
        print(
            f"SSH tunnel:      ssh -L {web_port}:localhost:{web_port} -L {grpc_port}:localhost:{grpc_port} <host>"
        )
        print()
    else:
        rr.init("raiden", spawn=True)

    rr.log(
        "info",
        rr.TextDocument(
            "\n".join(
                [
                    f"Recording : {rec_dir.name}",
                    f"Task      : {task_name}",
                    f"Prompt    : {prompt}",
                    f"Cameras   : {', '.join(cam_names)}",
                    f"Frames    : {n_frames}",
                ]
            ),
            media_type=rr.MediaType.TEXT,
        ),
        static=True,
    )

    frame_list = list(range(0, n_frames, stride))
    print(f"Logging {len(frame_list)} frames (stride={stride}) ...")

    s = image_scale

    # ------------------------------------------------------------------
    # Static trajectory lines: collect all EE positions then log once.
    # Action layout: [l_pos(3), l_rot9(9), l_grip(1), r_pos(3), r_rot9(9), r_grip(1)]
    # Positions are in the left-arm base frame.
    # ------------------------------------------------------------------
    left_pos_list, right_pos_list = [], []
    for fi in range(n_frames):
        p = ep_dir / "lowdim" / f"{fi:010d}.pkl"
        if p.exists():
            with open(p, "rb") as _f:
                a = pickle.load(_f)
            if "action" in a:
                act = a["action"]
                left_pos_list.append(act[0:3])
                r_pos_rb = np.array(act[13:16], dtype=np.float64)
                r_pos_lb = (T_left_from_right @ np.append(r_pos_rb, 1.0))[:3]
                right_pos_list.append(r_pos_lb.astype(np.float32))

    if right_pos_list:
        rr.log(
            "trajectory/right",
            rr.LineStrips3D(
                [np.array(right_pos_list, dtype=np.float32)],
                colors=[[220, 80, 60]],  # red
                radii=0.003,
            ),
            static=True,
        )
    if left_pos_list:
        rr.log(
            "trajectory/left",
            rr.LineStrips3D(
                [np.array(left_pos_list, dtype=np.float32)],
                colors=[[60, 120, 220]],  # blue
                radii=0.003,
            ),
            static=True,
        )

    for log_idx, frame_idx in enumerate(frame_list):
        rr.set_time("frame", sequence=frame_idx)
        frame_name = f"{frame_idx:010d}"
        action_logged = False

        for cam_name in cam_names:
            entity = f"world/cameras/{cam_name}"

            # ------------------------------------------------------------------
            # lowdim: intrinsics + extrinsics
            # ------------------------------------------------------------------
            lowdim_path = ep_dir / "lowdim" / f"{frame_name}.pkl"
            if not lowdim_path.exists():
                continue
            with open(lowdim_path, "rb") as _f:
                ld = pickle.load(_f)
            K = ld["intrinsics"].get(cam_name)  # (3, 3) or None
            T_c2w = ld["extrinsics"].get(cam_name)  # (4, 4) or None
            if K is None or T_c2w is None:
                continue
            K = K.astype(np.float64)
            T_c2w = T_c2w.astype(np.float64)

            # Scale intrinsics to match display resolution
            fx_s = K[0, 0] * s
            fy_s = K[1, 1] * s
            cx_s = K[0, 2] * s
            cy_s = K[1, 2] * s

            # ------------------------------------------------------------------
            # Camera frustum: Transform3D + Pinhole
            # ------------------------------------------------------------------
            rr.log(
                entity,
                rr.Transform3D(translation=T_c2w[:3, 3], mat3x3=T_c2w[:3, :3]),
            )

            # ------------------------------------------------------------------
            # RGB: load + downscale
            # ------------------------------------------------------------------
            img_rgb = None
            rgb_path = ep_dir / "rgb" / cam_name / f"{frame_name}.jpg"
            if rgb_path.exists():
                img_bgr = cv2.imread(str(rgb_path))
                if img_bgr is not None:
                    H0, W0 = img_bgr.shape[:2]
                    H_d = max(1, int(H0 * s))
                    W_d = max(1, int(W0 * s))
                    if s != 1.0:
                        img_bgr = cv2.resize(
                            img_bgr, (W_d, H_d), interpolation=cv2.INTER_AREA
                        )
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    K = np.array(
                        [[fx_s, 0, cx_s], [0, fy_s, cy_s], [0, 0, 1]], dtype=np.float32
                    )
                    rr.log(
                        entity,
                        rr.Pinhole(
                            image_from_camera=K,
                            width=W_d,
                            height=H_d,
                            image_plane_distance=frustum_scale,
                        ),
                    )
                    # Log at entity path → textures the frustum in the 3-D view
                    rr.log(entity, rr.Image(img_rgb))
                    # Log as child → visible in the 2-D Spatial View
                    rr.log(f"{entity}/rgb", rr.Image(img_rgb))

            # ------------------------------------------------------------------
            # Depth: plasma-colorized inverse depth (mirrors _log_depth)
            # ------------------------------------------------------------------
            depth_path = ep_dir / "depth" / cam_name / f"{frame_name}.npz"
            if depth_path.exists():
                depth_mm = np.load(depth_path)["depth"]  # uint16 H×W, mm
                depth_m = depth_mm.astype(np.float32) / 1000.0
                if img_rgb is not None:
                    # Resize depth to match RGB so point cloud colors align
                    depth_m = cv2.resize(
                        depth_m, (W_d, H_d), interpolation=cv2.INTER_NEAREST
                    )
                elif s != 1.0:
                    Hd, Wd = depth_m.shape
                    depth_m = cv2.resize(
                        depth_m,
                        (max(1, int(Wd * s)), max(1, int(Hd * s))),
                        interpolation=cv2.INTER_NEAREST,
                    )
                rr.log(f"{entity}/depth", rr.Image(_viz_depth(depth_m)))

                # Point cloud
                pts = _reconstruct_points(depth_m, fx_s, fy_s, cx_s, cy_s, T_c2w)
                valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
                pts_valid = pts[valid]
                if len(pts_valid) > 0:
                    kwargs: dict = {}
                    if img_rgb is not None:
                        kwargs["colors"] = img_rgb.reshape(-1, 3)[valid]
                    rr.log(f"world/points/{cam_name}", rr.Points3D(pts_valid, **kwargs))

            # ------------------------------------------------------------------
            # Action scalars + EE poses (logged once per frame, from first camera)
            # Layout: [l_pos(3), l_rot9(9), l_grip(1), r_pos(3), r_rot9(9), r_grip(1)]
            # ------------------------------------------------------------------
            if not action_logged and "action" in ld:
                action: np.ndarray = ld["action"]  # (26,)
                l_pos = action[0:3]
                l_rot = action[3:12].reshape(3, 3)
                l_grip = float(action[12])
                # Right arm is in right-arm base frame; transform to left-arm base frame.
                _r_pos_rb = np.array(action[13:16], dtype=np.float64)
                r_pos = (T_left_from_right @ np.append(_r_pos_rb, 1.0))[:3].astype(
                    np.float32
                )
                r_rot = (
                    T_left_from_right[:3, :3] @ action[16:25].reshape(3, 3)
                ).astype(np.float32)
                r_grip = float(action[25])

                # Position and gripper scalars
                for j, lbl in enumerate(["x", "y", "z"]):
                    rr.log(f"action/right/{lbl}", rr.Scalars(float(r_pos[j])))
                    rr.log(f"action/left/{lbl}", rr.Scalars(float(l_pos[j])))
                rr.log("action/right/gripper", rr.Scalars(r_grip))
                rr.log("action/left/gripper", rr.Scalars(l_grip))

                # EE coordinate frames in 3D (rotation matrix → visible axes)
                rr.log(
                    "world/ee/right", rr.Transform3D(translation=r_pos, mat3x3=r_rot)
                )
                rr.log("world/ee/left", rr.Transform3D(translation=l_pos, mat3x3=l_rot))

                # Current EE position marker (moves with the timeline)
                rr.log(
                    "trajectory/right/current",
                    rr.Points3D([r_pos], colors=[[220, 80, 60]], radii=0.01),
                )
                rr.log(
                    "trajectory/left/current",
                    rr.Points3D([l_pos], colors=[[60, 120, 220]], radii=0.01),
                )
                action_logged = True

        if (log_idx + 1) % 20 == 0 or log_idx == len(frame_list) - 1:
            print(f"  {log_idx + 1:>4d}/{len(frame_list)}  (frame {frame_idx})")

    if web:
        print("Done logging. Serving — press Ctrl-C to stop.")
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print("Done. Rerun viewer should now be open.")
