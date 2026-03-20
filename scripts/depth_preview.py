#!/usr/bin/env python3
"""Real-time depth map visualization for ZED cameras.

Single mode (default): runs one depth backend and shows a colorized depth map
alongside the left RGB image, with FPS and inference-time overlays.

Compare mode (``--compare``): runs multiple backends simultaneously and
displays them side-by-side in one window for quality comparison.

Usage::

    # Single mode — live camera
    uv run python scripts/depth_preview.py                    # ZED SDK depth
    uv run python scripts/depth_preview.py --method tri_stereo       # MMT (best available backend)
    uv run python scripts/depth_preview.py --method ffs       # FFS (best available backend)

    # Single mode — SVO2 playback
    uv run python scripts/depth_preview.py --svo path/to/recording.svo2

    # Select a camera by name (from camera.json) or serial number
    uv run python scripts/depth_preview.py --camera scene_1
    uv run python scripts/depth_preview.py --serial 12345678
    uv run python scripts/depth_preview.py --list-cameras   # print available cameras and exit

    # Compare mode — all three methods side-by-side
    uv run python scripts/depth_preview.py --compare
    uv run python scripts/depth_preview.py --compare --methods zed mmt

    # TRI Stereo variant
    uv run python scripts/depth_preview.py --method tri_stereo --tri-stereo-variant c32

Press Q or Esc to quit.
"""

import argparse
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import pyzed.sl as sl

_MAX_DEPTH_DEFAULT = 5.0  # meters
_MAX_DEPTH_WRIST = 1.5  # meters — wrist cameras are close-range
_MIN_DEPTH = 0.1  # meters — fixed near clamp for consistent colormap scale
_FPS_WINDOW = 30  # frames in rolling FPS average

# Cameras that are physically mounted upside-down — images are rotated 180°.
_FLIP_CAMERAS = {"right_wrist_camera"}

# Cameras that operate at close range — use a tighter depth color scale.
_WRIST_CAMERAS = {"left_wrist_camera", "right_wrist_camera"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _colorize_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    """Plasma-colorize a depth map using inverse depth with a fixed scale.

    Uses a fixed normalization range [1/max_depth, 1/_MIN_DEPTH] so the
    colormap is consistent across frames and models regardless of scene content.
    Invalid pixels (depth == 0) stay black.
    """
    valid = (depth > 0) & (depth <= max_depth)
    inv = np.where(valid, 1.0 / np.clip(depth, _MIN_DEPTH, max_depth), 0.0)
    inv_min = 1.0 / max_depth
    inv_max = 1.0 / _MIN_DEPTH
    inv_norm = np.clip((inv - inv_min) / (inv_max - inv_min), 0.0, 1.0)
    colored = cv2.applyColorMap((inv_norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    colored[~valid] = 0
    return colored


def _make_grid(panels: list[np.ndarray], cols: int = 2) -> np.ndarray:
    """Arrange panels into a grid with at most `cols` columns, padding with black if needed."""
    while len(panels) % cols != 0:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i : i + cols]) for i in range(0, len(panels), cols)]
    return np.vstack(rows)


def _overlay_text(img: np.ndarray, lines: list[str]) -> None:
    """Draw text lines in the top-left corner with a dark outline for readability."""
    y = 45
    for line in lines:
        cv2.putText(
            img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            img,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 48


class _FpsCounter:
    def __init__(self, window: int = _FPS_WINDOW) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


def list_cameras() -> None:
    """Print all detected ZED cameras and exit."""
    from raiden.camera_config import CameraConfig  # noqa: PLC0415

    cfg = CameraConfig()
    named: dict[int, str] = {}
    for name, entry in cfg.cameras.items():
        cam_type = entry.get("type", "zed") if isinstance(entry, dict) else "zed"
        if cam_type == "zed":
            serial = entry["serial"] if isinstance(entry, dict) else entry
            named[int(serial)] = name

    devices = sl.Camera.get_device_list()
    if not devices:
        print("No ZED cameras found.")
        return
    print(f"{'Name':<24} {'Serial':<12} {'Model':<20} Available")
    print("-" * 64)
    for d in devices:
        available = d.camera_state == sl.CAMERA_STATE.AVAILABLE
        name = named.get(d.serial_number, "—")
        print(
            f"{name:<24} {d.serial_number:<12} {str(d.camera_model):<20} {'yes' if available else 'no'}"
        )


def _resolve_serial(camera: Optional[str], serial: Optional[int]) -> Optional[int]:
    """Return a serial number from --camera name or --serial, whichever is given."""
    if camera is None:
        return serial
    from raiden.camera_config import CameraConfig  # noqa: PLC0415

    cfg = CameraConfig()
    entry = cfg.cameras.get(camera)
    if entry is None:
        names = list(cfg.cameras.keys())
        raise SystemExit(
            f"Camera {camera!r} not found in camera.json. Known names: {names}"
        )
    cam_type = entry.get("type", "zed") if isinstance(entry, dict) else "zed"
    if cam_type != "zed":
        raise SystemExit(f"Camera {camera!r} is type {cam_type!r}, not a ZED camera.")
    return int(entry["serial"] if isinstance(entry, dict) else entry)


def _open_zed(svo: Optional[str], serial: Optional[int], with_depth: bool) -> sl.Camera:
    cam = sl.Camera()
    params = sl.InitParameters()
    if svo:
        params.set_from_svo_file(svo)
        params.svo_real_time_mode = False
    else:
        params.camera_resolution = sl.RESOLUTION.HD720
        params.camera_fps = 30
        if serial is not None:
            params.set_from_serial_number(serial)

    params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT if with_depth else sl.DEPTH_MODE.NONE
    params.coordinate_units = sl.UNIT.METER
    params.depth_minimum_distance = 0.1

    status = cam.open(params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {status}")
    return cam


def _get_stereo_calib(cam: sl.Camera) -> tuple[float, float]:
    """Return ``(fx, baseline_m)`` for the left camera."""
    info = cam.get_camera_information()
    cal = info.camera_configuration.calibration_parameters
    return float(cal.left_cam.fx), float(abs(cal.get_camera_baseline()))


def _grab(
    cam: sl.Camera,
    left_mat: sl.Mat,
    right_mat: sl.Mat,
    depth_mat: Optional[sl.Mat],
) -> bool:
    status = cam.grab(sl.RuntimeParameters())
    if status != sl.ERROR_CODE.SUCCESS:
        return False
    cam.retrieve_image(left_mat, sl.VIEW.LEFT)
    cam.retrieve_image(right_mat, sl.VIEW.RIGHT)
    if depth_mat is not None:
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    return True


def _sdk_depth(depth_mat: sl.Mat) -> np.ndarray:
    raw = depth_mat.get_data().copy()
    return np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Predictor factory
# ---------------------------------------------------------------------------


def _build_predictor(method: str, tri_stereo_variant: str):
    """Return ``(predictor_or_None, label)`` for the best available backend."""
    if method == "zed":
        return None, "ZED SDK"

    if method == "tri_stereo":
        from raiden.depth.tri_stereo import (  # noqa: PLC0415
            TRIStereoOnnxDepthPredictor,
            TRIStereoTrtDepthPredictor,
        )

        if TRIStereoTrtDepthPredictor.engine_available(variant=tri_stereo_variant):
            pred = TRIStereoTrtDepthPredictor(variant=tri_stereo_variant)
            try:
                pred._ensure_loaded()
                return pred, f"TRI Stereo-{tri_stereo_variant.upper()} TRT"
            except RuntimeError as e:
                print(f"[TRIStereo] TRT engine unusable ({e}), falling back to ONNX")
        if TRIStereoOnnxDepthPredictor.model_available(variant=tri_stereo_variant):
            return TRIStereoOnnxDepthPredictor(
                variant=tri_stereo_variant
            ), f"TRI Stereo-{tri_stereo_variant.upper()} ONNX"
        raise RuntimeError(
            f"No TRI Stereo model found for variant '{tri_stereo_variant}'. "
            f"Run: git lfs pull"
        )

    if method == "ffs":
        from raiden.depth.ffs import (  # noqa: PLC0415
            FFSDepthPredictor,
            FFSOnnxDepthPredictor,
            FFSTrtDepthPredictor,
        )

        if FFSTrtDepthPredictor.engines_available():
            pred = FFSTrtDepthPredictor()
            try:
                pred._ensure_loaded()
                return pred, "FFS TRT"
            except RuntimeError as e:
                print(f"[FFS] TRT engine unusable ({e}), falling back to ONNX/PyTorch")
        if FFSOnnxDepthPredictor.models_available():
            return FFSOnnxDepthPredictor(), "FFS ONNX"
        return FFSDepthPredictor(), "FFS PyTorch"

    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Single mode
# ---------------------------------------------------------------------------


def run_single(
    method: str,
    svo: Optional[str],
    serial: Optional[int],
    flip: bool,
    tri_stereo_variant: str,
    max_depth: float,
) -> None:
    needs_sdk_depth = method == "zed"
    cam = _open_zed(svo, serial, with_depth=needs_sdk_depth)
    fx, baseline = _get_stereo_calib(cam)

    left_mat, right_mat = sl.Mat(), sl.Mat()
    depth_mat = sl.Mat() if needs_sdk_depth else None

    predictor, label = _build_predictor(method, tri_stereo_variant)

    fps_c = _FpsCounter()
    win = f"Depth Preview — {label}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not _grab(cam, left_mat, right_mat, depth_mat):
                break

            left_raw = left_mat.get_data()[:, :, :3].copy()
            right_raw = right_mat.get_data()[:, :, :3].copy()
            display_bgr = cv2.rotate(left_raw, cv2.ROTATE_180) if flip else left_raw

            t0 = time.perf_counter()
            if method == "zed":
                depth = _sdk_depth(depth_mat)
            else:
                # Run inference on raw (pre-rotation) images — the ZED rectifies
                # them in the sensor frame; rotating before inference only adds noise.
                depth = predictor.predict(left_raw, right_raw, fx, baseline)
            if flip:
                depth = cv2.rotate(depth, cv2.ROTATE_180)
            dt = time.perf_counter() - t0

            fps_c.tick()
            colored = _colorize_depth(depth, max_depth)
            _overlay_text(
                colored,
                [
                    label,
                    f"FPS: {fps_c.fps():.1f}",
                    f"Inference: {dt * 1000:.1f} ms",
                ],
            )

            cv2.imshow(win, np.hstack([display_bgr, colored]))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def run_compare(
    methods: list[str],
    svo: Optional[str],
    serial: Optional[int],
    flip: bool,
    tri_stereo_variant: str,
    max_depth: float,
    record_secs: Optional[float],
) -> None:
    needs_sdk_depth = "zed" in methods
    cam = _open_zed(svo, serial, with_depth=needs_sdk_depth)
    fx, baseline = _get_stereo_calib(cam)

    left_mat, right_mat = sl.Mat(), sl.Mat()
    depth_mat = sl.Mat() if needs_sdk_depth else None

    entries = [(m, *_build_predictor(m, tri_stereo_variant)) for m in methods]

    win = "Depth Preview — Compare"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    writer: Optional[cv2.VideoWriter] = None
    record_start: Optional[float] = None

    try:
        while True:
            if not _grab(cam, left_mat, right_mat, depth_mat):
                break

            left_raw = left_mat.get_data()[:, :, :3].copy()
            right_raw = right_mat.get_data()[:, :, :3].copy()
            display_bgr = cv2.rotate(left_raw, cv2.ROTATE_180) if flip else left_raw

            panels = [display_bgr]
            for method, pred, label in entries:
                if method == "zed":
                    depth = _sdk_depth(depth_mat)
                else:
                    depth = pred.predict(left_raw, right_raw, fx, baseline)
                if flip:
                    depth = cv2.rotate(depth, cv2.ROTATE_180)

                colored = _colorize_depth(depth, max_depth)
                _overlay_text(colored, [label])
                panels.append(colored)

            frame = _make_grid(panels)

            if record_secs is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        "compare.mp4",
                        cv2.VideoWriter_fourcc(*"avc1"),
                        15.0,
                        (w, h),
                    )
                    record_start = time.perf_counter()
                    print(f"Recording {record_secs}s to compare.mp4 ...")
                writer.write(frame)
                if time.perf_counter() - record_start >= record_secs:
                    print("Recording done.")
                    break

            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if writer is not None:
            writer.release()
        cam.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=["zed", "tri_stereo", "ffs"],
        default="zed",
        help="Depth backend for single mode (default: zed)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show multiple methods side-by-side",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["zed", "tri_stereo", "ffs"],
        default=["zed", "tri_stereo", "ffs"],
        metavar="METHOD",
        help="Backends for compare mode (default: zed tri_stereo ffs)",
    )
    parser.add_argument(
        "--tri-stereo-variant",
        choices=["c32", "c64"],
        default="c64",
        dest="tri_stereo_variant",
        help="TRI Stereo model variant (default: c64)",
    )
    parser.add_argument(
        "--camera",
        metavar="NAME",
        help="Camera name from camera.json (e.g. scene_1)",
    )
    parser.add_argument(
        "--serial",
        type=int,
        metavar="SN",
        help="ZED camera serial number (default: first available)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print detected ZED cameras and exit",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Rotate images 180° (auto-enabled for right_wrist_camera)",
    )
    parser.add_argument(
        "--svo",
        metavar="PATH",
        help="SVO2 file to replay instead of a live camera",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=_MAX_DEPTH_DEFAULT,
        dest="max_depth",
        metavar="METERS",
        help=f"Color-map upper bound in meters (default: {_MAX_DEPTH_DEFAULT})",
    )
    parser.add_argument(
        "--record-secs",
        type=float,
        default=None,
        dest="record_secs",
        metavar="SECS",
        help="Record the first N seconds of compare mode to compare.mp4 (default: off)",
    )
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    serial = _resolve_serial(args.camera, args.serial)
    flip = args.flip or (args.camera in _FLIP_CAMERAS)
    if args.max_depth == _MAX_DEPTH_DEFAULT and args.camera in _WRIST_CAMERAS:
        args.max_depth = _MAX_DEPTH_WRIST

    if args.compare:
        run_compare(
            args.methods,
            args.svo,
            serial,
            flip,
            args.tri_stereo_variant,
            args.max_depth,
            args.record_secs,
        )
    else:
        run_single(
            args.method, args.svo, serial, flip, args.tri_stereo_variant, args.max_depth
        )


if __name__ == "__main__":
    main()
