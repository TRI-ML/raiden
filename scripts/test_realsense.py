"""Visualize RGB and depth streams from one or more RealSense cameras using OpenCV.

Usage:
    # Auto-detect all connected RealSense cameras
    python scripts/test_realsense.py

    # Specify one or more serial numbers explicitly
    python scripts/test_realsense.py --serials 123456789012 987654321098

Press 'q' to quit.
"""

import argparse

import cv2
import numpy as np
import pyrealsense2 as rs

_FPS = 30


def _colorize_depth(depth_m: np.ndarray, percentile: int = 95) -> np.ndarray:
    """Plasma-colorize a depth map using inverse-depth mapping. Returns BGR uint8."""
    inv = np.where(depth_m > 0, 1.0 / np.clip(depth_m, 1e-8, None), 0.0)
    valid = inv[inv > 0]
    normalizer = float(np.percentile(valid, percentile)) if len(valid) > 0 else 1.0
    inv_norm = (inv / (normalizer + 1e-6)).clip(0.0, 1.0)
    colored = cv2.applyColorMap((inv_norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    colored[inv == 0] = 0
    return colored  # BGR


def detect_realsense_serials() -> list[str]:
    ctx = rs.context()
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]


def open_pipeline(serial: str) -> tuple[rs.pipeline, float]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 0, 0, rs.format.bgr8, _FPS)
    config.enable_stream(rs.stream.depth, 0, 0, rs.format.z16, _FPS)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # Warm up: discard first few frames
    print(f"  [{serial}] warming up...", end=" ", flush=True)
    for _ in range(10):
        pipeline.wait_for_frames()
    print("ready.")

    return pipeline, depth_scale


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize RealSense RGB + depth via OpenCV."
    )
    parser.add_argument(
        "--serials",
        nargs="*",
        default=None,
        help="Serial numbers of cameras to open. Auto-detects all if omitted.",
    )
    args = parser.parse_args()

    serials = args.serials if args.serials else detect_realsense_serials()
    if not serials:
        print("No RealSense cameras detected.")
        return

    print(f"Opening {len(serials)} camera(s): {serials}")
    print("Press 'q' to quit.")

    pipelines: list[tuple[str, rs.pipeline, float]] = []
    for serial in serials:
        try:
            pipeline, depth_scale = open_pipeline(serial)
            pipelines.append((serial, pipeline, depth_scale))
        except Exception as e:
            print(f"  [{serial}] failed: {e}")

    if not pipelines:
        print("No cameras opened successfully.")
        return

    try:
        while True:
            for serial, pipeline, depth_scale in pipelines:
                frames = pipeline.wait_for_frames()

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())  # BGR uint8
                depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
                depth_m = depth_raw.astype(np.float32) * depth_scale

                depth_vis = _colorize_depth(depth_m)

                # Resize depth to match color height for side-by-side display
                h = color.shape[0]
                scale = h / depth_vis.shape[0]
                depth_vis = cv2.resize(
                    depth_vis,
                    (int(depth_vis.shape[1] * scale), h),
                )

                # Overlay center distance on depth image
                cy, cx = depth_vis.shape[0] // 2, depth_vis.shape[1] // 2
                d_cy, d_cx = depth_raw.shape[0] // 2, depth_raw.shape[1] // 2
                center_dist = float(depth_m[d_cy, d_cx])
                cv2.circle(depth_vis, (cx, cy), 5, (255, 255, 255), -1)
                cv2.putText(
                    depth_vis,
                    f"{center_dist:.3f} m",
                    (cx - 60, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                combined = np.hstack([color, depth_vis])
                cv2.imshow(f"RealSense [{serial}]  RGB | Depth", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        for _, pipeline, _ in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
