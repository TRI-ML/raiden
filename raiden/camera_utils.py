"""Device listing utilities — cameras, robot arms, and SpaceMouse."""

import subprocess


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
        h = easyhid.Enumeration()
    except (ImportError, Exception):
        return []
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

    # ── Example camera.json ────────────────────────────────────────
    if zed_cams or rs_cams:
        _ROLE_ASSIGNMENTS = [
            ("scene_1", "scene"),
            ("right_wrist", "right_wrist"),
            ("left_wrist", "left_wrist"),
        ]
        all_cameras = [{"serial": c["serial"], "type": "zed"} for c in zed_cams] + [
            {"serial": c["serial"], "type": "realsense"} for c in rs_cams
        ]
        print("\nExample config/camera.json:")
        print("{")
        entries = []
        for i, cam in enumerate(all_cameras):
            key, role = (
                _ROLE_ASSIGNMENTS[i]
                if i < len(_ROLE_ASSIGNMENTS)
                else (f"scene_{i}", "scene")
            )
            serial_val = cam["serial"] if cam["type"] == "zed" else f'"{cam["serial"]}"'
            comma = "," if i < len(all_cameras) - 1 else ""
            entries.append(
                f'  "{key}": {{"serial": {serial_val}, "type": "{cam["type"]}", "role": "{role}"}}{comma}'
                f"  // TODO: verify assignment"
            )
        print("\n".join(entries))
        print("}")

    print()
