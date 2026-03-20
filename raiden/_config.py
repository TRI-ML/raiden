"""Default paths for user configuration files."""

from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "raiden"
DB_DIR = CONFIG_DIR / "db"
WEIGHTS_DIR = CONFIG_DIR / "weights"

CAMERA_CONFIG = str(CONFIG_DIR / "camera.json")
CALIBRATION_FILE = str(CONFIG_DIR / "calibration_results.json")
CALIBRATION_POSES_FILE = str(CONFIG_DIR / "calibration_poses.json")
SPACEMOUSE_CONFIG = str(CONFIG_DIR / "spacemouse.json")
