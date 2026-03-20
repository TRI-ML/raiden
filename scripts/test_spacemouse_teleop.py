#!/usr/bin/env python3
"""Test SpaceMouse Cartesian velocity teleop for two YAM follower arms.

Each SpaceMouse controls one follower arm via velocity integration + IK.
Translation is applied in the world (Cartesian) frame; rotation in the EE frame.

Usage
-----
    uv run scripts/test_spacemouse_teleop.py

    # Swap which physical mouse drives which arm:
    uv run scripts/test_spacemouse_teleop.py --path-r /dev/hidraw5 --path-l /dev/hidraw4

    # Slower, more precise motion:
    uv run scripts/test_spacemouse_teleop.py --vel-scale 0.02 --rot-scale 0.2

Controls
--------
    Push/pull/tilt puck   translate EE in world frame
    Rock/twist puck       rotate EE in world frame
    Button 1              close gripper
    Button 2              open gripper
    Foot pedal            soft e-stop (hold 5 s → home → exit)
    Ctrl-C                stop and return to home
"""

import argparse
import time

from raiden.robot.controller import RobotController


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpaceMouse Cartesian velocity teleop for YAM arms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path-r",
        default="/dev/hidraw4",
        help="hidraw path for the right-arm SpaceMouse",
    )
    parser.add_argument(
        "--path-l",
        default="/dev/hidraw5",
        help="hidraw path for the left-arm SpaceMouse",
    )
    parser.add_argument(
        "--vel-scale",
        type=float,
        default=0.12,
        help="Max translational speed in m/s at full SpaceMouse deflection",
    )
    parser.add_argument(
        "--rot-scale",
        type=float,
        default=2.0,
        help="Max rotational speed in rad/s at full SpaceMouse deflection",
    )
    args = parser.parse_args()

    robot = RobotController(
        use_right_leader=False,
        use_left_leader=False,
        use_right_follower=True,
        use_left_follower=True,
    )

    try:
        robot.check_can_interfaces()
        robot.initialize_robots()
        robot.move_to_home_positions(simultaneous=True)
        robot.attach_footpedal()

        robot.attach_spacemice(
            path_r=args.path_r,
            path_l=args.path_l,
        )
        robot.start_spacemouse_teleop(
            vel_scale=args.vel_scale,
            rot_scale=args.rot_scale,
        )

        print()
        print(f"  Right arm → {args.path_r}")
        print(f"  Left arm  → {args.path_l}")
        print()
        print("  Push/pull/tilt puck  :  translate EE (world frame)")
        print("  Rock/twist puck      :  rotate EE (world frame)")
        print("  Button 1             :  close gripper")
        print("  Button 2             :  open gripper")
        print("  Foot pedal           :  soft e-stop")
        print("  Ctrl-C               :  stop")
        print()

        while not robot.session_estop_requested:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nCtrl-C — stopping.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
