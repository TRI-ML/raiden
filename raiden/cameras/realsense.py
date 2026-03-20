"""Intel RealSense camera implementation (e.g. D405).

Recording mode opens by serial number and records to .bag via the SDK recorder.
Playback mode opens a .bag file and reads color + depth frames;
use ``RealSenseCamera.from_bag()`` to create a playback instance.

.. warning:: **For dynamic tasks, prefer ZED cameras over RealSense.**
   RealSense cameras have known synchronization limitations that make them
   less suitable for capturing fast robot motion.  See *Synchronization issues*
   below for details.

Intrinsics note
---------------
RealSense cameras store per-device calibration on-board.  get_intrinsics()
reads the color stream's Brown-Conrady coefficients directly from the SDK,
which is the same model OpenCV uses, so the values can be passed straight
to cv2 functions without any conversion.

Synchronization issues
----------------------
RealSense cameras (especially the D405) have several properties that make
multi-camera synchronization harder than with ZED cameras:

**Timestamp reliability**
    The ``global_time_enabled`` option is supposed to stamp frames with the
    host wall-clock time, but support is inconsistent across D4xx firmware
    versions.  When it is not supported, the SDK reports hardware-relative
    timestamps (milliseconds since device boot) that must be converted to
    wall-clock time.  ``_measure_clock_offset()`` compensates for this by
    measuring the offset between the first frame timestamp and
    ``time.time_ns()`` at recording start, but this is only an approximation.

**Frame extraction drain-loop bug**
    The RealSense SDK pre-buffers frames during bag playback when
    ``set_real_time(False)`` is set.  A naive "drain the queue to get the
    latest frame" loop (appropriate for live preview) will consume every
    other buffered frame, halving the apparent FPS to ~15 fps even when the
    bag was recorded at 30 fps.  ``grab()`` therefore skips the drain loop
    in playback mode (``_is_playback = True``).

**Recommendation**
    Use ZED cameras for scene and wrist cameras in dynamic manipulation
    tasks.  ZED timestamps (``sl.TIME_REFERENCE.IMAGE``) are wall-clock
    Unix nanoseconds, consistent across cameras and with the robot
    controller clock, making multi-camera alignment straightforward.
    RealSense cameras are best suited for static or slow-motion scenes
    where their close-range depth quality is the primary requirement.
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs

from .base import Camera, CameraFrame


def _enable_global_time(device: rs.device) -> None:
    """Enable global timestamp on all sensors that support it."""
    for sensor in device.query_sensors():
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1)


class RealSenseCamera(Camera):
    """Intel RealSense D4xx camera – records to .bag"""

    _COLOR_W, _COLOR_H = 640, 480
    _DEPTH_W, _DEPTH_H = 640, 480

    def __init__(self, camera_name: str, serial_number: str, fps: int = 30):
        self._name = camera_name
        self._serial = serial_number
        self._fps = fps
        self._pipeline: Optional[rs.pipeline] = None
        self._config: Optional[rs.config] = None
        self._profile: Optional[rs.pipeline_profile] = None
        self._depth_scale: float = 0.001  # metres per raw unit (D405 default)
        self._latest_frames = None
        # Actual negotiated color resolution (set after pipeline.start()).
        self._color_w: int = self._COLOR_W
        self._color_h: int = self._COLOR_H
        # Offset (ns) to convert RS SDK timestamps to wall-clock:
        # wall_ns = frame.get_timestamp() * 1_000_000 + _clock_offset_ns
        # Measured once at start_recording() by comparing the first frame
        # timestamp to time.time_ns().  None until start_recording() is called.
        self._clock_offset_ns: Optional[int] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def serial_number(self) -> str:
        return str(self._serial)

    @property
    def recording_extension(self) -> str:
        return "bag"

    # ------------------------------------------------------------------
    # Live recording
    # ------------------------------------------------------------------

    def _start_pipeline(self, cfg: "rs.config") -> None:
        """Start the pipeline and record the negotiated color resolution."""
        self._profile = self._pipeline.start(cfg)
        self._config = cfg
        color_stream = self._profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        self._color_w = color_stream.width()
        self._color_h = color_stream.height()

    def open(self) -> None:
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self._serial)
        cfg.enable_stream(
            rs.stream.color, self._COLOR_W, self._COLOR_H, rs.format.bgr8, self._fps
        )
        cfg.enable_stream(
            rs.stream.depth, self._DEPTH_W, self._DEPTH_H, rs.format.z16, self._fps
        )
        self._start_pipeline(cfg)
        print(
            f"  [{self._name}] opened: "
            f"BGR8 {self._color_w}×{self._color_h} @ {self._fps}fps"
        )
        device = self._profile.get_device()
        _enable_global_time(device)
        depth_sensor = device.first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

    def close(self) -> None:
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            self._profile = None

    def start_recording(self, path: Path) -> None:
        """Restart pipeline with bag recorder attached."""
        if self._pipeline:
            self._pipeline.stop()

        cfg = rs.config()
        cfg.enable_device(self._serial)
        cfg.enable_stream(
            rs.stream.color, self._COLOR_W, self._COLOR_H, rs.format.bgr8, self._fps
        )
        cfg.enable_stream(
            rs.stream.depth, self._DEPTH_W, self._DEPTH_H, rs.format.z16, self._fps
        )
        cfg.enable_record_to_file(str(path))
        self._start_pipeline(cfg)
        _enable_global_time(self._profile.get_device())
        print(
            f"  [{self._name}] recording: "
            f"BGR8 {self._color_w}×{self._color_h} @ {self._fps}fps"
        )
        self._measure_clock_offset()

    def _measure_clock_offset(self) -> None:
        """Grab the first available frame and compute RS-to-wall-clock offset.

        ``_clock_offset_ns`` is the additive correction such that:
            wall_ns = int(frame.get_timestamp() * 1_000_000) + _clock_offset_ns

        This works whether or not global_time_enabled is supported by the
        device: if it is supported the stored timestamps are already wall-clock
        and the offset will be ~0; if not, the offset captures the difference
        between the RealSense hardware clock and the system clock at the
        moment recording begins, which is stable enough over a 10-60 s episode.
        """
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=2000)
            wall_ns = time.time_ns()
            color_frame = frames.get_color_frame()
            if color_frame:
                rs_ns = int(color_frame.get_timestamp() * 1_000_000)
                self._clock_offset_ns = wall_ns - rs_ns
        except Exception:
            self._clock_offset_ns = None

    def stop_recording(self) -> None:
        """Restart pipeline without recorder to finalise the bag file."""
        if self._pipeline:
            self._pipeline.stop()
        # Rebuild config without enable_record_to_file — there is no
        # disable_record_and_playback() in the pyrealsense2 Python bindings.
        # Use the same color resolution that was negotiated in start_recording().
        cfg = rs.config()
        cfg.enable_device(self._serial)
        cfg.enable_stream(
            rs.stream.color, self._color_w, self._color_h, rs.format.bgr8, self._fps
        )
        cfg.enable_stream(
            rs.stream.depth, self._DEPTH_W, self._DEPTH_H, rs.format.z16, self._fps
        )
        self._config = cfg
        self._profile = self._pipeline.start(cfg)
        _enable_global_time(self._profile.get_device())

    def grab(self) -> bool:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=500)
            if not getattr(self, "_is_playback", False):
                # Drain any additional buffered frames so we always use the most
                # recent one. Bag-file writing overhead can cause the queue to
                # accumulate, making wait_for_frames return increasingly stale frames.
                # Skip this in playback mode: set_real_time(False) pre-buffers the
                # next frame, so draining would skip every other frame (30fps → 15fps).
                while True:
                    ok, newer = self._pipeline.try_wait_for_frames(timeout_ms=0)
                    if not ok:
                        break
                    frames = newer
            self._latest_frames = frames
            return True
        except RuntimeError:
            return False

    def get_current_timestamp_ns(self) -> int:
        """Return the capture timestamp of the most recently grabbed frame.

        With global_time_enabled the RealSense SDK stamps frames with system
        wall-clock time, so this is on the same clock as time.time_ns() and
        can be used directly to align robot data with camera frames.
        """
        if self._latest_frames is not None:
            frame = self._latest_frames.get_color_frame()
            if frame:
                return int(frame.get_timestamp() * 1_000_000)
        import time

        return time.time_ns()

    def get_frame(self) -> CameraFrame:
        if self._latest_frames is None:
            raise RuntimeError("No frames available. Call grab() first.")

        frames = self._latest_frames
        align = getattr(self, "_align", None)
        if align is not None:
            frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color = np.asanyarray(color_frame.get_data())  # BGR uint8
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 raw units
        depth = (depth_raw * self._depth_scale).astype(np.float32)

        # RealSense timestamp is in milliseconds
        timestamp_ns = int(color_frame.get_timestamp() * 1_000_000)

        return CameraFrame(color=color, depth=depth, timestamp_ns=timestamp_ns)

    def get_intrinsics(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Return on-device calibrated intrinsics for the color stream.

        RealSense stores per-unit calibration on-board.  The distortion model
        is Brown-Conrady (identical to OpenCV's), so coefficients are directly
        usable with cv2 functions.
        """
        if self._profile is None:
            raise RuntimeError("Camera is not open. Call open() first.")

        color_stream = self._profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intr = color_stream.get_intrinsics()

        camera_matrix = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # Brown-Conrady: [k1, k2, p1, p2, k3]
        dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64)
        image_size = (intr.width, intr.height)
        return camera_matrix, dist_coeffs, image_size

    # ------------------------------------------------------------------
    # Playback from .bag file
    # ------------------------------------------------------------------

    @classmethod
    def from_bag(cls, camera_name: str, bag_path: Path) -> "RealSenseCamera":
        """Open a .bag file for frame-by-frame extraction."""
        cam = cls.__new__(cls)
        cam._name = camera_name
        cam._serial = ""
        cam._fps = 30
        cam._depth_scale = 0.001
        cam._latest_frames = None

        cam._pipeline = rs.pipeline()
        cam._config = rs.config()
        rs.config.enable_device_from_file(
            cam._config, str(bag_path), repeat_playback=False
        )
        cam._profile = cam._pipeline.start(cam._config)

        # Non-realtime playback so we don't drop frames
        playback = cam._profile.get_device().as_playback()
        playback.set_real_time(False)

        try:
            ds = cam._profile.get_device().first_depth_sensor()
            cam._depth_scale = ds.get_depth_scale()
        except Exception:
            pass

        # Align depth to color so they share the same pixel grid
        cam._align = rs.align(rs.stream.color)
        cam._is_playback = True

        return cam

    def get_camera_info(self) -> dict:
        """Return camera intrinsics and metadata as a plain dict."""
        if not self._profile:
            return {}
        color_stream = self._profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        return {
            "serial_number": self._serial,
            "model": "RealSense",
            "fps": self._fps,
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "k1": intr.coeffs[0],
            "k2": intr.coeffs[1],
            "p1": intr.coeffs[2],
            "p2": intr.coeffs[3],
            "k3": intr.coeffs[4],
        }
