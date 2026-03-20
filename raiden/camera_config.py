"""Camera configuration management.

Maps semantic camera names to hardware serial numbers, camera types, and roles.

**Format**::

    {
        "scene_1":       {"serial": 37038161,       "type": "zed",        "role": "scene"},
        "scene_2":       {"serial": 55667788,       "type": "zed",        "role": "scene"},
        "left_wrist":    {"serial": "123456789012", "type": "realsense",  "role": "left_wrist"},
        "right_wrist":   {"serial": 14932342,       "type": "zed",        "role": "right_wrist"}
    }

Roles
-----
- ``"scene"``       : fixed overhead / scene camera (multiple allowed)
- ``"left_wrist"``  : wrist camera on the left arm (at most one)
- ``"right_wrist"`` : wrist camera on the right arm (at most one)

**Legacy format** (ZED only, still fully supported)::

    {
        "scene_camera":        37038161,
        "left_wrist_camera":   16522755
    }

Integer values are assumed to be ZED cameras with no role assigned.
"""

import json
from pathlib import Path

from raiden._config import CAMERA_CONFIG
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from raiden.cameras.base import Camera


def _parse_entry(entry: Any) -> tuple[str, Any]:
    """Return (camera_type, serial) from a config entry (old or new format)."""
    if isinstance(entry, int):
        return "zed", entry
    if isinstance(entry, dict):
        cam_type = entry.get("type", "zed").lower()
        return cam_type, entry["serial"]
    raise ValueError(f"Invalid camera config entry: {entry!r}")


class CameraConfig:
    """Manages camera configuration mapping semantic names to hardware info."""

    def __init__(self, config_file: str = CAMERA_CONFIG):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        if self.config_file.exists():
            with open(self.config_file) as f:
                self.cameras: Dict[str, Any] = json.load(f)
        else:
            self.cameras = {}
            self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        with open(self.config_file, "w") as f:
            json.dump(self.cameras, f, indent=2)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_serial_by_name(self, name: str) -> Optional[Union[int, str]]:
        """Return the serial number for a camera name (int for ZED, str for RealSense)."""
        entry = self.cameras.get(name)
        if entry is None:
            return None
        _, serial = _parse_entry(entry)
        return serial

    def get_camera_type(self, name: str) -> Optional[str]:
        """Return 'zed' or 'realsense' for the named camera, or None if not found."""
        entry = self.cameras.get(name)
        if entry is None:
            return None
        cam_type, _ = _parse_entry(entry)
        return cam_type

    def get_name_by_serial(self, serial: Union[int, str]) -> Optional[str]:
        for name, entry in self.cameras.items():
            _, entry_serial = _parse_entry(entry)
            if str(entry_serial) == str(serial):
                return name
        return None

    def get_role(self, name: str) -> Optional[str]:
        """Return the role for a camera ('scene', 'left_wrist', 'right_wrist'), or None."""
        entry = self.cameras.get(name)
        if isinstance(entry, dict):
            return entry.get("role")
        return None

    def get_camera_by_role(self, role: str) -> Optional[str]:
        """Return the name of the unique camera with the given role, or None.

        Use for 'left_wrist' and 'right_wrist' (at most one each).
        For 'scene' cameras use get_cameras_by_role().
        """
        for name, entry in self.cameras.items():
            if isinstance(entry, dict) and entry.get("role") == role:
                return name
        return None

    def get_cameras_by_role(self, role: str) -> List[str]:
        """Return names of all cameras with the given role (e.g. multiple 'scene' cameras)."""
        return [
            name
            for name, entry in self.cameras.items()
            if isinstance(entry, dict) and entry.get("role") == role
        ]

    def list_cameras(self) -> Dict[str, Any]:
        """Return a copy of the raw config dict."""
        return self.cameras.copy()

    def list_camera_names(self) -> list[str]:
        return list(self.cameras.keys())

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_camera(
        self,
        name: str,
        serial: Union[int, str],
        camera_type: str = "zed",
        role: Optional[str] = None,
    ) -> None:
        """Add or update a camera entry.

        Args:
            name: Unique camera name (key in the config).
            serial: Hardware serial number (int for ZED, str for RealSense).
            camera_type: 'zed' or 'realsense'.
            role: 'scene', 'left_wrist', or 'right_wrist'. Optional but recommended.
        """
        entry: Dict[str, Any] = {"serial": serial, "type": camera_type}
        if role is not None:
            entry["role"] = role
        self.cameras[name] = entry
        self._save()

    def remove_camera(self, name: str) -> bool:
        if name in self.cameras:
            del self.cameras[name]
            self._save()
            return True
        return False

    # ------------------------------------------------------------------
    # Camera factory
    # ------------------------------------------------------------------

    def create_camera(self, name: str) -> "Camera":
        """Create and return a Camera instance for the named camera.

        Does NOT call open() – the caller is responsible for that.

        Raises:
            ValueError: if the name is not in the config or the type is unknown.
        """
        entry = self.cameras.get(name)
        if entry is None:
            raise ValueError(
                f"Camera '{name}' not found in config ({self.config_file})"
            )

        cam_type, serial = _parse_entry(entry)
        fps: int = entry.get("fps", 30) if isinstance(entry, dict) else 30

        if cam_type == "zed":
            from raiden.cameras.zed import ZedCamera

            return ZedCamera(name, int(serial), fps=fps)

        if cam_type == "realsense":
            from raiden.cameras.realsense import RealSenseCamera

            return RealSenseCamera(name, str(serial), fps=fps)

        raise ValueError(
            f"Unknown camera type '{cam_type}' for camera '{name}'. "
            "Supported types: 'zed', 'realsense'"
        )

    # ------------------------------------------------------------------
    # Validation / detection
    # ------------------------------------------------------------------

    def validate_against_hardware(self) -> Dict[str, bool]:
        """Check which configured ZED cameras are currently connected."""
        import pyzed.sl as sl  # noqa: PLC0415

        connected_serials = {cam.serial_number for cam in sl.Camera.get_device_list()}
        result = {}
        for name, entry in self.cameras.items():
            cam_type, serial = _parse_entry(entry)
            if cam_type == "zed":
                result[name] = int(serial) in connected_serials
            else:
                # RealSense availability check not implemented here
                result[name] = True
        return result

    @staticmethod
    def detect_cameras() -> Dict[int, Dict]:
        """Detect all connected ZED cameras and return their info."""
        import pyzed.sl as sl  # noqa: PLC0415

        detected = {}
        for cam_info in sl.Camera.get_device_list():
            detected[cam_info.serial_number] = {
                "model": str(cam_info.camera_model),
                "id": cam_info.id,
                "available": cam_info.camera_state == sl.CAMERA_STATE.AVAILABLE,
            }
        return detected

    def create_default_config(self) -> Dict[str, Any]:
        """Suggest a default config based on detected ZED cameras."""
        detected = self.detect_cameras()
        if not detected:
            return {}
        serials = list(detected.keys())
        suggested: Dict[str, Any] = {}
        roles = [
            ("scene_1", "scene"),
            ("right_wrist", "right_wrist"),
            ("left_wrist", "left_wrist"),
        ]
        for i, serial in enumerate(serials):
            key, role = roles[i] if i < len(roles) else (f"camera_{i}", "scene")
            suggested[key] = {"serial": serial, "type": "zed", "role": role}
        return suggested
