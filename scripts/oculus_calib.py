#!/usr/bin/env python3
"""Oculus controller axis calibration — no robot needed.

Connects to an Oculus Quest via ADB and shows in real-time how the controller
axes map to the robot base frame under the current rmat_reorder. Use this to
find the right rmat_reorder before running a full teleop session.

Usage:
    uv run python scripts/oculus_calib.py
    uv run python scripts/oculus_calib.py --rmat_reorder '-1,-2,3,4'
    uv run python scripts/oculus_calib.py --controller r   # right (default)
    uv run python scripts/oculus_calib.py --controller l   # left

Controls:
    RJ (joystick click)  Zero the orientation (define current pose as origin)
    RG (grip trigger)    Hold to enable delta tracking from latch point
    Ctrl-C               Quit
"""

import argparse
import os
import sys
import time

import numpy as np


def _vec_to_reorder_mat(vec: list) -> np.ndarray:
    X = np.zeros((len(vec), len(vec)))
    for i in range(len(vec)):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X


def _axis_table(rmat_reorder: list) -> list[str]:
    """Human-readable VR → robot axis mapping for the given rmat_reorder."""
    mat = _vec_to_reorder_mat(rmat_reorder)
    vr_labels  = ["+X (right)", "+Y (up)", "+Z (back)"]
    robot_axes = ["X", "Y", "Z"]
    lines = []
    for vr_i, vr_lbl in enumerate(vr_labels):
        for rob_i, rob_ax in enumerate(robot_axes):
            v = mat[rob_i, vr_i]
            if abs(v) > 0.5:
                sign = "+" if v > 0 else "-"
                lines.append(f"    VR {vr_lbl:<20} →  Robot {sign}{rob_ax}")
    return lines


def _render(
    rmat_reorder: list,
    controller_id: str,
    zeroed: bool,
    grip: bool,
    raw_pos: np.ndarray | None,
    robot_pos: np.ndarray | None,
    origin_delta: np.ndarray | None,
) -> str:
    width = 60
    lines = [
        "─" * width,
        "  Oculus Axis Calibration",
        "─" * width,
        f"  rmat_reorder : {rmat_reorder}",
        f"  Controller   : {'right' if controller_id == 'r' else 'left'}",
        "",
        "  RJ = zero orientation    RG = hold to track delta    ^C = quit",
        "",
        f"  Orient zeroed : {'YES' if zeroed else 'NO  (press joystick to zero)'}",
        f"  Grip held     : {'YES' if grip else 'NO'}",
        "",
        "  VR → Robot axis map:",
    ]
    lines += _axis_table(rmat_reorder)
    lines += [""]

    if raw_pos is not None:
        lines.append(
            f"  VR raw pos (m) :  x={raw_pos[0]:+.3f}  y={raw_pos[1]:+.3f}  z={raw_pos[2]:+.3f}"
        )
    if robot_pos is not None:
        lines.append(
            f"  Robot frame (m):  x={robot_pos[0]:+.3f}  y={robot_pos[1]:+.3f}  z={robot_pos[2]:+.3f}"
        )

    if grip and origin_delta is not None:
        lines += [
            "",
            "  Delta from grip latch (move controller to check axes):",
            f"    Δx={origin_delta[0]:+.3f}  Δy={origin_delta[1]:+.3f}  Δz={origin_delta[2]:+.3f}",
        ]
    elif grip:
        lines += ["", "  (grip just latched — move controller to see delta)"]
    else:
        lines += ["", "  Hold RG (grip trigger) to see per-axis delta."]

    lines.append("─" * width)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rmat_reorder",
        default="-2,-1,-3,4",
        help="Signed-index axis map, comma-separated (default: -2,-1,-3,4)",
    )
    parser.add_argument(
        "--controller",
        choices=["r", "l"],
        default="r",
        help="Which controller to read: r=right (default), l=left",
    )
    args = parser.parse_args()

    rmat_reorder = [int(x) for x in args.rmat_reorder.split(",")]
    controller_id = args.controller

    global_to_env_mat: np.ndarray = _vec_to_reorder_mat(rmat_reorder)

    print("Connecting to Oculus Quest via ADB ...")
    from oculus_reader.reader import OculusReader
    reader = OculusReader()
    time.sleep(0.8)

    grip_key = controller_id.upper() + "G"
    joystick_key = controller_id.upper() + "J"

    vr_to_global_mat: np.ndarray = np.eye(4)
    zeroed = False
    reset_orientation = True

    prev_grip = False
    origin_mat: np.ndarray | None = None

    print("Connected. Put on the headset — the Teleop app should be visible.")
    print("Press RJ (joystick) to zero orientation, then hold RG (grip) to track.\n")

    try:
        while True:
            t0 = time.monotonic()

            poses, buttons = reader.get_transformations_and_buttons()
            if not poses or controller_id not in poses:
                time.sleep(0.05)
                continue

            raw_pose = np.asarray(poses[controller_id], dtype=np.float64)
            grip = bool(buttons.get(grip_key, False))
            joystick = bool(buttons.get(joystick_key, False))

            # Orientation zeroing (same logic as _oculus_control_loop)
            stop_updating = joystick or grip
            if reset_orientation:
                try:
                    vr_to_global_mat = np.linalg.inv(raw_pose)
                    zeroed = True
                except np.linalg.LinAlgError:
                    vr_to_global_mat = np.eye(4)
                if stop_updating:
                    reset_orientation = False
            if joystick and not stop_updating:
                reset_orientation = True

            vr_mat = global_to_env_mat @ vr_to_global_mat @ raw_pose
            raw_pos = raw_pose[:3, 3]
            robot_pos = vr_mat[:3, 3]

            # Grip latch
            if grip and not prev_grip:
                origin_mat = vr_mat.copy()
            if not grip:
                origin_mat = None
            prev_grip = grip

            origin_delta = None
            if grip and origin_mat is not None:
                origin_delta = robot_pos - origin_mat[:3, 3]

            frame = _render(
                rmat_reorder, controller_id, zeroed, grip,
                raw_pos, robot_pos, origin_delta,
            )
            os.system("clear")
            print(frame)

            elapsed = time.monotonic() - t0
            remaining = 0.1 - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
