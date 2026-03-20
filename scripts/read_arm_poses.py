#!/usr/bin/env python3
"""
Test script to read joint poses from all four robot arms using get_joint_pos().
Useful for checking hardware connections and current positions.
"""

import time
import sys
from typing import Dict, List
import numpy as np

# Add i2rt to path
sys.path.insert(0, "../third_party/i2rt")

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType


class ArmReader:
    """Simple wrapper to read arm state"""

    def __init__(self, can_channel: str, gripper_type: GripperType, name: str):
        self.name = name
        self.can_channel = can_channel
        print(f"\n[{name}] Initializing on {can_channel}...")
        try:
            self.robot = get_yam_robot(channel=can_channel, gripper_type=gripper_type)
            print(f"[{name}] ✓ Successfully initialized")
        except Exception as e:
            print(f"[{name}] ✗ Failed to initialize: {e}")
            raise

    def get_state(self) -> Dict:
        """Get current joint positions"""
        joint_pos = self.robot.get_joint_pos()

        state = {
            "joint_pos": joint_pos,
            "name": self.name,
            "channel": self.can_channel,
        }

        return state


def format_joint_pos(joint_pos: np.ndarray) -> str:
    """Format joint positions for display"""
    return "[" + ", ".join([f"{x:7.3f}" for x in joint_pos]) + "]"


def main():
    print("=" * 80)
    print("  ROBOT ARM POSE READER")
    print("=" * 80)
    print("\nInitializing all four arms...")
    print("(This may take a few seconds as motors are being turned on)")

    # Define the arms
    arm_configs = [
        {
            "can_channel": "can_follower_r",
            "gripper_type": GripperType.LINEAR_4310,
            "name": "Follower Right",
        },
        {
            "can_channel": "can_leader_r",
            "gripper_type": GripperType.YAM_TEACHING_HANDLE,
            "name": "Leader Right",
        },
        {
            "can_channel": "can_follower_l",
            "gripper_type": GripperType.LINEAR_4310,
            "name": "Follower Left",
        },
        {
            "can_channel": "can_leader_l",
            "gripper_type": GripperType.YAM_TEACHING_HANDLE,
            "name": "Leader Left",
        },
    ]

    # Initialize all arms (skip any that fail)
    arms: List[ArmReader] = []

    for config in arm_configs:
        try:
            arm = ArmReader(**config)
            arms.append(arm)
        except Exception:
            print(f"✗ Skipping {config['name']} - will continue with other arms")

    if len(arms) == 0:
        print("\n✗ No arms initialized successfully. Exiting.")
        sys.exit(1)

    print("\n" + "=" * 80)
    if len(arms) == 4:
        print("  ✓ ALL 4 ARMS INITIALIZED")
    else:
        print(f"  ✓ {len(arms)}/4 ARMS INITIALIZED")
    print("=" * 80)
    print("\nReading joint poses at 10Hz (Press Ctrl+C to stop)...")
    print()

    try:
        while True:
            # Clear screen (ANSI escape code)
            print("\033[2J\033[H", end="")

            print("=" * 80)
            print(f"  ROBOT ARM POSES - {time.strftime('%H:%M:%S')}")
            print("=" * 80)
            print()

            # Read and display each arm
            for arm in arms:
                try:
                    state = arm.get_state()

                    print(f"{state['name']:20s} ({state['channel']})")
                    print(f"  Joint Pos: {format_joint_pos(state['joint_pos'])}")
                    print()

                except Exception as e:
                    print(f"{arm.name:20s} - Error reading: {e}")
                    print()

            print("=" * 80)
            print("Press Ctrl+C to stop")
            print("=" * 80)

            time.sleep(0.1)  # 10Hz update rate

    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user")
        print("\nCleaning up...")

        # Motors will automatically turn off when objects are destroyed
        del arms

        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()
