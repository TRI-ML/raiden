"""Bimanual teleoperation implementation"""

import os
import signal
import sys
import time

from raiden.robot.controller import RobotController


def run_bimanual_teleop(
    bilateral_kp: float = 0.0,
    control: str = "leader",
    spacemouse_path_r: str = "/dev/hidraw4",
    spacemouse_path_l: str = "/dev/hidraw5",
    vel_scale: float = 0.12,
    rot_scale: float = 3.0,
    invert_rotation: bool = False,
    arms: str = "bimanual",
):
    """Run the bimanual teleoperation system"""

    use_right = arms == "bimanual"
    use_left = True
    use_leaders = control == "leader"

    robot_controller = RobotController(
        use_right_leader=use_leaders and use_right,
        use_left_leader=use_leaders and use_left,
        use_right_follower=use_right,
        use_left_follower=use_left,
    )

    try:

        def signal_handler(signum, frame):
            robot_controller.emergency_stop()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        robot_controller.setup_for_teleop_recording()
        robot_controller.enable_estop()

        if control == "spacemouse":
            robot_controller.attach_spacemice(spacemouse_path_r, spacemouse_path_l)
            robot_controller.warmup_spacemouse_ik()
            robot_controller.start_spacemouse_teleop(
                vel_scale=vel_scale,
                rot_scale=rot_scale,
                invert_rotation=invert_rotation,
            )

            print("\n" + "=" * 60)
            print("  SPACEMOUSE TELEOPERATION ACTIVE")
            print("=" * 60)
            print("\n  Push/pull/tilt puck to move EE, rock/twist to rotate")
            print("  Press Ctrl+C to stop\n")
            print("=" * 60 + "\n")

            while True:
                if robot_controller.session_estop_requested:
                    print("\n[FootPedal] Returning arms to home and exiting.")
                    break
                time.sleep(0.1)
        else:
            robot_controller.start_teleoperation()

            print("\n" + "=" * 60)
            print("  BIMANUAL TELEOPERATION ACTIVE")
            print("=" * 60)
            print("\n  Press the button on any leader arm to return ALL arms to home")
            print("  Press Ctrl+C for EMERGENCY STOP (stops all motion immediately)\n")
            print("=" * 60 + "\n")

            while True:
                if robot_controller.session_estop_requested:
                    print("\n[FootPedal] Returning arms to home and exiting.")
                    break
                pressed_leader = robot_controller.check_button_press()
                if pressed_leader:
                    print(
                        f"\n{pressed_leader}: Button pressed! Returning all arms to home..."
                    )
                    time.sleep(0.5)  # Debounce
                    break
                time.sleep(0.1)

        robot_controller.shutdown()

        print("\nTeleoperation session ended.")
        os._exit(0)

    except Exception as e:
        print(f"\nError: {e}")

        if robot_controller and robot_controller.has_robots():
            robot_controller.emergency_stop()

        sys.exit(1)
