"""Interactive recording of robot poses for calibration"""

import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import cv2
import rerun as rr

from raiden._config import CALIBRATION_POSES_FILE, CAMERA_CONFIG
from raiden.camera_config import CameraConfig
from raiden.cameras.base import Camera
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
        control: str = "leader",
        spacemouse_path_r: str = "/dev/hidraw7",
        spacemouse_path_l: str = "/dev/hidraw6",
        vel_scale: float = 2.0,
        rot_scale: float = 3.0,
        invert_rotation: bool = False,
    ):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.camera_config_file = camera_config_file
        self.charuco_config = charuco_config or ChArUcoBoardConfig()
        self.poses: List[CalibrationPose] = []

        self.control = control
        self.spacemouse_path_r = spacemouse_path_r
        self.spacemouse_path_l = spacemouse_path_l
        self.vel_scale = vel_scale
        self.rot_scale = rot_scale
        self.invert_rotation = invert_rotation

        self.robot_controller: Optional[RobotController] = None
        self.calibration_targets: List[str] = []

        self.cameras: Dict[str, Camera] = {}
        self._camera_roles: Dict[str, Optional[str]] = {}
        self._rerun_stop_event = threading.Event()

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

        use_leaders = self.control != "spacemouse"
        self.robot_controller = RobotController(
            use_right_leader=use_leaders and has_right,
            use_left_leader=use_leaders and has_left,
            use_right_follower=has_right,
            use_left_follower=has_left,
        )

        # Complete setup: check CAN -> init -> home -> grav comp
        self.robot_controller.setup_for_teleop_recording()

        if self.control == "spacemouse":
            self.robot_controller.warmup_spacemouse_ik()
            self.robot_controller.attach_spacemice(
                self.spacemouse_path_r, self.spacemouse_path_l
            )
            self.robot_controller.start_spacemouse_teleop(
                vel_scale=self.vel_scale,
                rot_scale=self.rot_scale,
                invert_rotation=self.invert_rotation,
            )

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

    def _open_cameras(self, warmup_frames: int = 10) -> None:
        """Open all cameras from config for live Rerun visualization."""
        cfg = CameraConfig(self.camera_config_file)
        for name in cfg.list_camera_names():
            try:
                cam = cfg.create_camera(name)
                cam.open()
                self.cameras[name] = cam
                self._camera_roles[name] = cfg.get_role(name)
            except Exception as e:
                print(f"  Warning: could not open camera '{name}': {e}")
        if self.cameras:
            print(f"  Warming up {len(self.cameras)} camera(s)...")
            for _ in range(warmup_frames):
                for cam in self.cameras.values():
                    cam.grab()

    def _rerun_stream(self) -> None:
        rr.init("raiden_calibration")
        grpc_port = 9878
        web_port = 9877
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(web_port=web_port, open_browser=False)
        viewer_url = f"http://localhost:{web_port}?url={quote(server_uri, safe='')}"
        print(f"\nRerun viewer:    {viewer_url}")
        print(
            f"SSH tunnel:      ssh -L {web_port}:localhost:{web_port} -L {grpc_port}:localhost:{grpc_port} <host>"
        )
        print()
        frame_idx = 0
        while not self._rerun_stop_event.is_set():
            # Drain the camera buffer by grabbing at full rate, but only log once per second.
            for name, cam in self.cameras.items():
                try:
                    if cam.grab():
                        frame = cam.get_frame()
                        color = frame.color
                        if self._camera_roles.get(name) == "right_wrist":
                            color = cv2.rotate(color, cv2.ROTATE_180)
                        img_rgb = color[:, :, ::-1]  # BGR → RGB
                        rr.set_time("frame", sequence=frame_idx)
                        rr.log(f"cameras/{name}", rr.Image(img_rgb))
                except Exception:
                    pass
            frame_idx += 1
            self._rerun_stop_event.wait(1.0)

    def _start_rerun_stream(self) -> None:
        if not self.cameras:
            return
        self._rerun_stop_event.clear()
        self._rerun_thread = threading.Thread(target=self._rerun_stream, daemon=True)
        self._rerun_thread.start()
        print(f"  ✓ Rerun stream started ({len(self.cameras)} camera(s))")

    def _stop_rerun_stream(self) -> None:
        self._rerun_stop_event.set()
        if hasattr(self, "_rerun_thread"):
            self._rerun_thread.join(timeout=2.0)
        for cam in self.cameras.values():
            cam.close()
        self.cameras.clear()

    def run_interactive_recording(self, min_poses: int = 5):
        """Run interactive pose recording session"""
        import select
        import signal
        import termios
        import tty

        print("\n" + "=" * 70)
        print("  Raiden - Camera Calibration Pose Recording")
        print("=" * 70)

        # Open cameras and start Rerun visualization
        print("\nOpening cameras for live visualization...")
        self._open_cameras()
        self._start_rerun_stream()

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
        if self.control == "spacemouse":
            print("  2. Move the robot arms using SpaceMouse")
        else:
            print("  2. Move the LEADER arms - followers will automatically follow")
        print("  3. Position so the board is clearly visible from the camera(s)")
        if self.control == "spacemouse":
            print("  4. Type 'r' to record pose")
        else:
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
                    # Check for button press (leader mode only)
                    if (
                        has_leaders
                        and self.control != "spacemouse"
                        and self.check_button_press()
                    ):
                        print("\n\nButton pressed! Recording pose...")
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
                    if has_leaders and self.control != "spacemouse":
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
            self._stop_rerun_stream()
            # Cleanup robots
            if self.robot_controller:
                self.robot_controller.cleanup()


def run_calibration_pose_recording(
    min_poses: int = 5,
    output_file: str = CALIBRATION_POSES_FILE,
    camera_config_file: str = CAMERA_CONFIG,
    charuco_config: Optional[ChArUcoBoardConfig] = None,
    control: str = "leader",
    spacemouse_path_r: str = "/dev/hidraw7",
    spacemouse_path_l: str = "/dev/hidraw6",
    vel_scale: float = 2.0,
    rot_scale: float = 3.0,
    invert_rotation: bool = False,
):
    """Main entry point for calibration pose recording"""
    recorder = CalibrationPoseRecorder(
        output_file=output_file,
        camera_config_file=camera_config_file,
        charuco_config=charuco_config,
        control=control,
        spacemouse_path_r=spacemouse_path_r,
        spacemouse_path_l=spacemouse_path_l,
        vel_scale=vel_scale,
        rot_scale=rot_scale,
        invert_rotation=invert_rotation,
    )
    recorder.run_interactive_recording(min_poses=min_poses)
