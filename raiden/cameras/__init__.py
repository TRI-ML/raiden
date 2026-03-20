from .base import Camera, CameraFrame
from .realsense import RealSenseCamera

__all__ = ["Camera", "CameraFrame", "ZedCamera", "RealSenseCamera"]


def __getattr__(name: str):
    if name == "ZedCamera":
        from .zed import ZedCamera

        return ZedCamera
    raise AttributeError(f"module 'raiden.cameras' has no attribute {name!r}")
