"""Interactive recording of robot poses for calibration"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from raiden._config import CALIBRATION_POSES_FILE, CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.robot.controller import RobotController


@dataclass
class ChArUcoBoardConfig:
    """Configuration for ChArUco calibration board"""

    squares_x: int = 9
    squares_y: int = 9
    square_length: float = 0.03  # meters (checker size: 30mm)
    marker_length: float = 0.023  # meters (marker size: 23mm)
    dictionary: str = "DICT_6X6_250"  # ArUco dictionary

    def to_dict(self):
        return asdict(self)


@dataclass
class CalibrationPose:
    """A single calibration pose with robot joint positions"""

    id: int
    name: str
    follower_r: Optional[List[float]] = None
    follower_l: Optional[List[float]] = None
    timestamp: str = ""
    notes: str = ""

    def to_dict(self):
        result = {
            "id": self.id,
            "name": self.name,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }
        if self.follower_r is not None:
            result["follower_r"] = self.follower_r
        if self.follower_l is not None:
            result["follower_l"] = self.follower_l
        return result


class CalibrationPoseRecorder:
    """Records robot poses for camera calibration"""

    def __init__(
        self,
        output_file: str = CALIBRATION_POSES_FILE,
        camera_config_file: str = CAMERA_CONFIG,
        charuco_config: Optional[ChArUcoBoardConfig] = None,
    ):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.camera_config_file = camera_config_file
        self.charuco_config = charuco_config or ChArUcoBoardConfig()
        self.poses: List[CalibrationPose] = []

        self.robot_controller: Optional[RobotController] = None
        self.calibration_targets: List[str] = []

    def initialize_robots(self):
        """Initialize robots based on which wrist cameras are in camera.json."""
        cfg = CameraConfig(self.camera_config_file)

        has_right = cfg.get_camera_by_role("right_wrist") is not None
        has_left = cfg.get_camera_by_role("left_wrist") is not None

        # Collect wrist camera names for calibration targets
        if has_right:
            cam = cfg.get_camera_by_role("right_wrist")
            self.calibration_targets.append(cam)
        if has_left:
            cam = cfg.get_camera_by_role("left_wrist")
            self.calibration_targets.append(cam)

        # Always include all scene cameras
        for cam in cfg.get_cameras_by_role("scene"):
            if cam not in self.calibration_targets:
                self.calibration_targets.append(cam)

        self.robot_controller = RobotController(
            use_right_leader=has_right,
            use_left_leader=has_left,
            use_right_follower=has_right,
            use_left_follower=has_left,
        )

        # Complete setup: check CAN -> init -> home -> grav comp -> start teleop
        self.robot_controller.setup_for_teleop_recording()

    def record_current_pose(self, notes: str = "") -> CalibrationPose:
        """Record the current robot pose"""
        pose_id = len(self.poses)
        timestamp = datetime.now().isoformat()

        pose = CalibrationPose(
            id=pose_id,
            name=f"pose_{pose_id}",
            timestamp=timestamp,
            notes=notes,
        )

        # Get joint positions from robot controller
        joint_positions = self.robot_controller.get_joint_positions()

        if "follower_r" in joint_positions:
            pose.follower_r = joint_positions["follower_r"].tolist()
        if "follower_l" in joint_positions:
            pose.follower_l = joint_positions["follower_l"].tolist()

        self.poses.append(pose)
        return pose

    def delete_last_pose(self) -> bool:
        """Delete the most recently recorded pose"""
        if self.poses:
            deleted = self.poses.pop()
            print(f"✓ Deleted pose {deleted.id}: {deleted.name}")
            return True
        else:
            print("No poses to delete")
            return False

    def list_poses(self):
        """Print all recorded poses"""
        if not self.poses:
            print("No poses recorded yet")
            return

        print(f"\nRecorded {len(self.poses)} pose(s):")
        for pose in self.poses:
            print(f"  {pose.id}: {pose.name} - {pose.timestamp}")
            if pose.notes:
                print(f"     Notes: {pose.notes}")

    def check_button_press(self) -> bool:
        """Check if any leader button was pressed

        Returns:
            True if a button was pressed (rising edge), False otherwise
        """
        pressed_leader = self.robot_controller.check_button_press()
        return pressed_leader is not None

    def save_poses(self):
        """Save recorded poses to JSON file"""
        data = {
            "version": "1.0",
            "charuco_config": self.charuco_config.to_dict(),
            "poses": [pose.to_dict() for pose in self.poses],
            "calibration_targets": self.calibration_targets,
            "board_mount": "fixed",
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
        }

        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Saved {len(self.poses)} poses to {self.output_file}")

    def run_interactive_recording(self, min_poses: int = 5):
        """Run interactive pose recording session"""
        import select
        import signal
        import termios
        import tty

        print("\n" + "=" * 70)
        print("  Raiden - Camera Calibration Pose Recording")
        print("=" * 70)

        # Initialize robots
        self.initialize_robots()

        # Setup signal handlers for emergency stop (SIGTERM)
        # Note: SIGINT (Ctrl+C) is handled via KeyboardInterrupt for graceful save
        def signal_handler(signum, frame):
            self.robot_controller.emergency_stop()

        signal.signal(signal.SIGTERM, signal_handler)

        print("\nTarget Cameras:")
        for target in self.calibration_targets:
            print(f"  - {target}")

        print("\nChArUco Board Configuration:")
        print(
            f"  - Grid size: {self.charuco_config.squares_x}x{self.charuco_config.squares_y} squares"
        )
        print(f"  - Checker size: {self.charuco_config.square_length * 1000:.1f} mm")
        print(f"  - Marker size: {self.charuco_config.marker_length * 1000:.1f} mm")
        print(f"  - Dictionary: {self.charuco_config.dictionary}")

        print("\nInstructions:")
        print("  1. Position the ChArUco board in a fixed location")
        print("  2. Move the LEADER arms - followers will automatically follow")
        print("  3. Position so the board is clearly visible from the camera(s)")
        print("  4. Press button on any leader arm OR type 'r' to record pose")
        print(
            f"  5. Record at least {min_poses} diverse poses (vary distance and angle)"
        )

        print("\nCommands:")
        print("  r - Record current pose")
        print("  d - Delete last pose")
        print("  l - List recorded poses")
        print("  q - Quit and save")
        print("  h - Show help")

        has_leaders = self.robot_controller.leader_r or self.robot_controller.leader_l
        if has_leaders:
            print("\nButton Input:")
            print("  - Press button on any leader arm to record current pose")

        print("\n" + "=" * 70)
        print("\nReady to record poses!")

        # Save terminal settings for raw mode
        old_settings = None
        if sys.stdin.isatty():
            old_settings = termios.tcgetattr(sys.stdin)

        # Interactive loop
        try:
            while True:
                # Status line
                poses_count = len(self.poses)
                status = f"\nPoses recorded: {poses_count}"
                if poses_count < min_poses:
                    status += f" / {min_poses} (minimum)"
                else:
                    status += " ✓ (minimum reached)"
                print(status)

                print(
                    "\nWaiting for input (button press or command)...",
                    end="",
                    flush=True,
                )

                # Monitor button presses in a loop
                command = None
                while command is None:
                    # Check for button press
                    if has_leaders and self.check_button_press():
                        print("\n\n🔘 Button pressed! Recording pose...")
                        time.sleep(0.5)  # Debounce
                        command = "r"
                        break

                    # Check for keyboard input (non-blocking)
                    if sys.stdin.isatty() and old_settings:
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            # Switch to raw mode to read single character
                            tty.setraw(sys.stdin.fileno())
                            char = sys.stdin.read(1)
                            termios.tcsetattr(
                                sys.stdin, termios.TCSADRAIN, old_settings
                            )

                            if char == "\x03":  # Ctrl+C
                                raise KeyboardInterrupt
                            elif char == "\n" or char == "\r":
                                # Enter pressed, get full command
                                print("\r\nCommand [r/d/l/q/h]: ", end="", flush=True)
                                # Restore normal mode for input
                                command = input().strip().lower()
                                break
                            elif char in ["r", "d", "l", "q", "h"]:
                                # Direct command without Enter
                                command = char
                                print(f"\r\nCommand: {command}")
                                break
                    else:
                        # Fallback for non-TTY or if we can't set raw mode
                        time.sleep(0.1)

                if command == "r":
                    pose = self.record_current_pose()
                    print(f"✓ Recorded pose {pose.id}: {pose.name}")

                    if self.robot_controller.follower_r and pose.follower_r:
                        pos = pose.follower_r[:3]
                        print(
                            f"  Right follower position (first 3 joints): {[f'{p:.3f}' for p in pos]}"
                        )

                    if self.robot_controller.follower_l and pose.follower_l:
                        pos = pose.follower_l[:3]
                        print(
                            f"  Left follower position (first 3 joints): {[f'{p:.3f}' for p in pos]}"
                        )

                elif command == "d":
                    self.delete_last_pose()

                elif command == "l":
                    self.list_poses()

                elif command == "q":
                    if poses_count < min_poses:
                        print(
                            f"\nWarning: Only {poses_count} poses recorded (minimum: {min_poses})."
                        )
                        print("Continue anyway? [y/N]: ", end="", flush=True)
                        # Restore terminal for input
                        if old_settings:
                            termios.tcsetattr(
                                sys.stdin, termios.TCSADRAIN, old_settings
                            )
                        confirm = input().strip().lower()
                        if confirm != "y":
                            print("Continuing recording...")
                            continue

                    self.save_poses()
                    print("\nCalibration pose recording complete!")

                    # Shutdown: stop teleop -> restore control -> go home
                    self.robot_controller.shutdown()

                    break

                elif command == "h":
                    print("\nCommands:")
                    print("  r - Record current pose")
                    print("  d - Delete last pose")
                    print("  l - List recorded poses")
                    print("  q - Quit and save")
                    print("  h - Show help")
                    if has_leaders:
                        print("\nButton Input:")
                        print("  - Press button on any leader arm to record")

                else:
                    print(f"Unknown command: {command}")
                    print("Type 'h' for help")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            if old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            if self.poses:
                print("Save recorded poses before exiting? [Y/n]: ", end="", flush=True)
                save = input().strip().lower()
                if save != "n":
                    self.save_poses()

            # Shutdown: stop teleop -> restore control -> go home
            self.robot_controller.shutdown()
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()

            # Perform emergency stop on error
            if self.robot_controller and self.robot_controller.has_robots():
                self.robot_controller.emergency_stop()
        finally:
            # Cleanup robots
            if self.robot_controller:
                self.robot_controller.cleanup()


def run_calibration_pose_recording(
    min_poses: int = 5,
    output_file: str = CALIBRATION_POSES_FILE,
    camera_config_file: str = CAMERA_CONFIG,
    charuco_config: Optional[ChArUcoBoardConfig] = None,
):
    """Main entry point for calibration pose recording"""
    recorder = CalibrationPoseRecorder(
        output_file=output_file,
        camera_config_file=camera_config_file,
        charuco_config=charuco_config,
    )
    recorder.run_interactive_recording(min_poses=min_poses)
