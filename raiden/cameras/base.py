"""Abstract camera interface for recording and conversion"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class CameraFrame:
    """A single camera frame"""

    color: np.ndarray  # HxWx3 BGR uint8
    depth: Optional[np.ndarray]  # HxW float32 meters, None if unavailable
    timestamp_ns: int  # nanosecond timestamp (camera clock)


class Camera(ABC):
    """Abstract base class for cameras.

    Supports two usage modes:
    - Live recording: open() → start_recording(path) → grab() loop → stop_recording() → close()
    - File playback: open_file(path) → grab() loop → get_frame() → close()
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Semantic name assigned to this camera (e.g. 'scene_camera')"""
        ...

    @property
    @abstractmethod
    def serial_number(self) -> str:
        """Camera hardware serial number as string"""
        ...

    @property
    @abstractmethod
    def recording_extension(self) -> str:
        """File extension used for native recording (e.g. 'svo2', 'bag')"""
        ...

    @abstractmethod
    def open(self) -> None:
        """Open live camera connection"""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close camera and release all resources"""
        ...

    @abstractmethod
    def start_recording(self, path: Path) -> None:
        """Begin recording to file.

        For ZED this enables SVO2 recording; every subsequent grab() is stored.
        For RealSense this starts a .bag recorder.
        """
        ...

    @abstractmethod
    def stop_recording(self) -> None:
        """Stop recording and finalize the output file"""
        ...

    @abstractmethod
    def grab(self) -> bool:
        """Grab the next available frame.

        Blocks until a frame is ready (rate-limited to camera FPS).
        While recording is active, the frame is also written to the output file.

        Returns:
            True on success, False on error or end-of-file
        """
        ...

    @abstractmethod
    def get_frame(self) -> CameraFrame:
        """Retrieve color and depth from the most recently grabbed frame.

        Only valid to call after a successful grab().
        """
        ...

    def get_current_timestamp_ns(self) -> int:
        """Current host time in nanoseconds.

        The default implementation uses ``time.time_ns()``.  Camera subclasses
        that have their own hardware clock (e.g. ZED) should override this so
        that the returned value is on the *same clock* as ``CameraFrame.timestamp_ns``,
        enabling direct interpolation between robot data and camera frames.
        """
        import time

        return time.time_ns()

    @abstractmethod
    def get_intrinsics(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Return factory-calibrated camera intrinsics.

        These come directly from the camera SDK (ZED factory calibration,
        RealSense on-device calibration) rather than being computed from images.

        Returns:
            camera_matrix: 3x3 float64 array ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``
            dist_coeffs: 1-D float64 array ``[k1, k2, p1, p2, k3]`` (OpenCV Brown-Conrady)
            image_size: ``(width, height)`` in pixels
        """
        ...
