"""Bimanual teleoperation implementation"""

import os
import signal
import sys
import time

from raiden.control import TeleopInterface, YAMInterface, build_interface
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

    interface: TeleopInterface = build_interface(
        control,
        spacemouse_path_r=spacemouse_path_r,
        spacemouse_path_l=spacemouse_path_l,
        vel_scale=vel_scale,
        rot_scale=rot_scale,
        invert_rotation=invert_rotation,
    )

    robot_controller = RobotController(
        use_right_leader=interface.uses_leaders and use_right,
        use_left_leader=interface.uses_leaders and use_left,
        use_right_follower=use_right,
        use_left_follower=use_left,
    )

    interface.open()

    try:

        def signal_handler(signum, frame):
            robot_controller.emergency_stop()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        robot_controller.setup_for_teleop_recording()
        robot_controller.enable_estop()

        interface.setup(robot_controller)
        interface.start(robot_controller)

        print(interface.banner)

        while True:
            if robot_controller.session_estop_requested:
                print("\n[FootPedal] Returning arms to home and exiting.")
                break
            if interface.poll(robot_controller):
                time.sleep(0.5)  # debounce
                break
            time.sleep(0.1)

        interface.stop(robot_controller)
        robot_controller.shutdown()

        print("\nTeleoperation session ended.")

    except Exception as e:
        print(f"\nError: {e}")

        if robot_controller and robot_controller.has_robots():
            robot_controller.emergency_stop()

        sys.exit(1)
    finally:
        interface.close()
        os._exit(0)
