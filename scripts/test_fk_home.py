#!/usr/bin/env python3
"""Print the FK home pose for both arms to compare sim vs real.

Run in sim (no robot needed):
    uv run scripts/test_fk_home.py --sim

Run on real robot:
    uv run scripts/test_fk_home.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "i2rt"))

from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import YAM_XML_LINEAR_4310_PATH


def print_pose(label: str, T: np.ndarray) -> None:
    pos = T[:3, 3]
    R = T[:3, :3]
    print(f"\n  {label}")
    print(f"    position : x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}")
    print("    rotation :")
    for row in R:
        print(f"      [{row[0]:+.4f}  {row[1]:+.4f}  {row[2]:+.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Use simulated zero joints instead of reading real robot",
    )
    args = parser.parse_args()

    kin = Kinematics(YAM_XML_LINEAR_4310_PATH, "grasp_site")
    home = np.zeros(6)

    print("=" * 50)
    print("FK at home (q = zeros) — robot base frame")
    print("=" * 50)
    T_home = kin.fk(home)
    print_pose("home (q=zeros)", T_home)

    if not args.sim:
        from raiden.robot.controller import RobotController

        robot = RobotController(
            use_right_leader=False,
            use_left_leader=False,
            use_right_follower=True,
            use_left_follower=True,
        )
        robot.check_can_interfaces()
        robot.initialize_robots()

        print("\n" + "=" * 50)
        print("FK at actual joint positions (real robot)")
        print("=" * 50)

        for name, follower in [("right", robot.follower_r), ("left", robot.follower_l)]:
            if follower is None:
                continue
            q_full = follower.get_joint_pos()
            q_arm = q_full[:6]
            print(f"\n  {name} arm joints: {np.round(q_arm, 4)}")
            T = kin.fk(q_arm)
            print_pose(f"{name} arm EE pose", T)

        robot.close()


if __name__ == "__main__":
    main()
