"""Device listing utilities — cameras, robot arms, and SpaceMouse."""

import json
import subprocess
from pathlib import Path

from raiden._config import CAMERA_CONFIG, SPACEMOUSE_CONFIG

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


def list_realsense_cameras() -> list:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []
    ctx = rs.context()
    results = []
    for dev in ctx.query_devices():
        results.append(
            {
                "serial": dev.get_info(rs.camera_info.serial_number),
                "name": dev.get_info(rs.camera_info.name),
            }
        )
    return results


def list_zed_cameras() -> list:
    try:
        import pyzed.sl as sl
    except ImportError:
        return []
    results = []
    for cam_info in sl.Camera.get_device_list():
        available = cam_info.camera_state == sl.CAMERA_STATE.AVAILABLE
        results.append(
            {
                "serial": cam_info.serial_number,
                "model": str(cam_info.camera_model),
                "id": cam_info.id,
                "available": available,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Robot arms (CAN interfaces)
# ---------------------------------------------------------------------------

_CAN_INTERFACES = {
    "can_follower_r": "Right follower arm",
    "can_follower_l": "Left follower arm",
    "can_leader_r": "Right leader arm",
    "can_leader_l": "Left leader arm",
}


def list_arms() -> list:
    results = []
    for iface, label in _CAN_INTERFACES.items():
        result = subprocess.run(
            ["ip", "link", "show", iface],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        up = "state UP" in result.stdout or "state UNKNOWN" in result.stdout
        results.append({"interface": iface, "label": label, "up": up})
    return results


# ---------------------------------------------------------------------------
# SpaceMouse
# ---------------------------------------------------------------------------


def list_spacemice() -> list:
    try:
        import easyhid
    except ImportError:
        return []
    h = easyhid.Enumeration()
    seen: set = set()
    results = []
    for d in h.find():
        if "3Dconnexion" not in d.description():
            continue
        desc = d.description()
        path = desc.split("|")[0].strip()
        if path not in seen:
            seen.add(path)
            results.append({"path": path, "description": desc})
    return results


# ---------------------------------------------------------------------------
# Combined listing
# ---------------------------------------------------------------------------


def list_devices() -> None:
    """Print all connected cameras, robot arms, and SpaceMouse devices."""

    # ── Cameras ──────────────────────────────────────────────────────────
    zed_cams = list_zed_cameras()
    rs_cams = list_realsense_cameras()

    print(f"\nZED cameras: {len(zed_cams)} found")
    print("-" * 40)
    if zed_cams:
        for cam in zed_cams:
            state = "available" if cam["available"] else "in use"
            print(
                f"  id={cam['id']}  serial={cam['serial']}  model={cam['model']}  [{state}]"
            )
    else:
        print("  (none)")

    print(f"\nRealSense cameras: {len(rs_cams)} found")
    print("-" * 40)
    if rs_cams:
        for cam in rs_cams:
            print(f"  serial={cam['serial']}  name={cam['name']}")
    else:
        print("  (none)")

    # ── Robot arms ────────────────────────────────────────────────────────
    arms = list_arms()
    print(f"\nRobot arms: {len(arms)} CAN interface(s) found")
    print("-" * 40)
    if arms:
        for arm in arms:
            state = "UP" if arm["up"] else "DOWN"
            print(f"  {arm['interface']:<20}  {arm['label']:<22}  [{state}]")
    else:
        print("  (none)  — run scripts/reset_all_can.sh to bring up CAN interfaces")

    # ── SpaceMouse ────────────────────────────────────────────────────────
    mice = list_spacemice()
    print(f"\nSpaceMouse devices: {len(mice)} found")
    print("-" * 40)
    if mice:
        for m in mice:
            print(f"  {m['path']:<20}  {m['description']}")
    else:
        print("  (none)")

    # ── Auto-generate camera.json if missing ─────────────────────────────
    print("\nConfig files stored in: ~/.config/raiden/")
    print("-" * 40)
    if (zed_cams or rs_cams) and not Path(CAMERA_CONFIG).exists():
        _ROLE_ASSIGNMENTS = [
            ("scene_camera", "scene"),
            ("right_wrist_camera", "right_wrist"),
            ("left_wrist_camera", "left_wrist"),
        ]
        all_cameras = [{"serial": c["serial"], "type": "zed"} for c in zed_cams] + [
            {"serial": c["serial"], "type": "realsense"} for c in rs_cams
        ]
        config: dict = {}
        for i, cam in enumerate(all_cameras):
            key, role = (
                _ROLE_ASSIGNMENTS[i]
                if i < len(_ROLE_ASSIGNMENTS)
                else (f"scene_{i + 1}", "scene")
            )
            config[key] = {"serial": cam["serial"], "type": cam["type"], "role": role}
        Path(CAMERA_CONFIG).parent.mkdir(parents=True, exist_ok=True)
        with open(CAMERA_CONFIG, "w") as f:
            json.dump(config, f, indent=2)
        print("  ✓ Generated camera.json:")
        print(f"    {json.dumps(config, indent=2)}")
        print(
            "  WARNING: left/right wrist assignment is based on detection order and"
            " may be swapped — verify and edit camera.json if needed."
        )
    else:
        status = (
            "already exists" if Path(CAMERA_CONFIG).exists() else "no cameras found"
        )
        print(f"  camera.json: {status}")

    # ── Auto-generate spacemouse.json if missing ──────────────────────────
    if mice and not Path(SPACEMOUSE_CONFIG).exists():
        sm_config: dict = {}
        if len(mice) >= 1:
            sm_config["path_r"] = mice[0]["path"]
        if len(mice) >= 2:
            sm_config["path_l"] = mice[1]["path"]
        Path(SPACEMOUSE_CONFIG).parent.mkdir(parents=True, exist_ok=True)
        with open(SPACEMOUSE_CONFIG, "w") as f:
            json.dump(sm_config, f, indent=2)
        print("  ✓ Generated spacemouse.json:")
        print(f"    {json.dumps(sm_config, indent=2)}")
        if len(mice) >= 2:
            print(
                "  WARNING: left/right SpaceMouse assignment is based on detection"
                " order and may be swapped — verify and edit spacemouse.json if needed."
            )
    else:
        status = (
            "already exists"
            if Path(SPACEMOUSE_CONFIG).exists()
            else "no SpaceMouse found"
        )
        print(f"  spacemouse.json: {status}")

    print()
