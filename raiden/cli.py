#!/usr/bin/env python3
"""Command-line interface for Raiden teleoperation toolkit"""

import json
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import tyro

from raiden._config import (
    CALIBRATION_FILE,
    CALIBRATION_POSES_FILE,
    CAMERA_CONFIG,
    SPACEMOUSE_CONFIG,
)
from raiden.calibration.recorder import run_calibration_pose_recording
from raiden.calibration.runner import CalibrationRunner
from raiden.converter import convert_task, select_tasks
from raiden.recorder import run_recording
from raiden.robot.replay import run_replay
from raiden.robot.teleop import run_bimanual_teleop
from raiden.shardify import ShardifyConfig, run_shardify, select_processed_task
from raiden.utils import select_processed_recording, select_recording
from raiden.visualizer import select_task_and_episode, visualize_recording


@dataclass
class TeleopCommand:
    """Start bimanual teleoperation with improved synchronization"""

    control: Literal["leader", "spacemouse"] = "leader"
    """Control mode: leader-follower arms or SpaceMouse EE velocity control"""

    arms: Literal["bimanual", "single"] = "bimanual"
    """Which arms to use: both (bimanual) or left arm only (single)"""

    bilateral_kp: float = 0.0
    """Bilateral force feedback gain (default: 0.0 for no feedback)"""

    spacemouse_path_r: str = "/dev/hidraw7"
    """hidraw path for the right-arm SpaceMouse (spacemouse mode only)"""

    spacemouse_path_l: str = "/dev/hidraw6"
    """hidraw path for the left-arm SpaceMouse (spacemouse mode only)"""

    vel_scale: float = 0.07
    """Max translational speed in m/s at full deflection (spacemouse mode only)"""

    rot_scale: float = 0.8
    """Max rotational speed in rad/s at full deflection (spacemouse mode only)"""

    invert_rotation: bool = False
    """Negate all SpaceMouse rotation axes (spacemouse mode only)"""


@dataclass
class RecordCommand:
    """Record a demonstration with cameras and robot data"""

    control: Literal["leader", "spacemouse"] = "leader"
    """Control mode: leader-follower arms or SpaceMouse EE velocity control"""

    data_dir: str = "data"
    """Root data directory (default: ./data); episodes go to <data_dir>/raw/<task>/"""

    s3_bucket: Optional[str] = None
    """S3 bucket name for uploading recordings (optional)"""

    s3_prefix: str = "demonstrations"
    """S3 prefix/folder for uploads (default: demonstrations)"""

    spacemouse_path_r: str = "/dev/hidraw7"
    """hidraw path for the right-arm SpaceMouse (spacemouse mode only)"""

    spacemouse_path_l: str = "/dev/hidraw6"
    """hidraw path for the left-arm SpaceMouse (spacemouse mode only)"""

    vel_scale: float = 0.12
    """Max translational speed in m/s at full deflection (spacemouse mode only)"""

    rot_scale: float = 3.0
    """Max rotational speed in rad/s at full deflection (spacemouse mode only)"""

    invert_rotation: bool = False
    """Negate all SpaceMouse rotation axes (spacemouse mode only)"""

    arms: Literal["bimanual", "single"] = "bimanual"
    """Which arms to use: both (bimanual) or left arm only (single)"""


_CAN_BITRATE = 1000000


@dataclass
class ResetCanCommand:
    """Reset CAN interfaces (bring down then up)"""

    interfaces: List[str] = field(default_factory=list)
    """CAN interfaces to reset (default: all detected interfaces)"""


@dataclass
class ConsoleCommand:
    pass


@dataclass
class ListDevicesCommand:
    pass


@dataclass
class RecordCalibrationPosesCommand:
    """Record robot poses for camera calibration"""

    min_poses: int = 5
    """Minimum number of poses to record (default: 5)"""

    output_file: str = CALIBRATION_POSES_FILE
    """Path to save calibration poses"""

    control: Literal["leader", "spacemouse"] = "leader"
    """Control mode: leader-follower arms or SpaceMouse EE velocity control"""

    spacemouse_path_r: str = "/dev/hidraw7"
    """hidraw path for the right-arm SpaceMouse (spacemouse mode only)"""

    spacemouse_path_l: str = "/dev/hidraw6"
    """hidraw path for the left-arm SpaceMouse (spacemouse mode only)"""

    vel_scale: float = 0.12
    """Max translational speed in m/s at full deflection (spacemouse mode only)"""

    rot_scale: float = 3.0
    """Max rotational speed in rad/s at full deflection (spacemouse mode only)"""

    invert_rotation: bool = False
    """Negate all SpaceMouse rotation axes (spacemouse mode only)"""


@dataclass
class CalibrateCommand:
    """Run camera calibration using recorded poses"""

    poses_file: str = CALIBRATION_POSES_FILE
    """Path to recorded calibration poses"""

    output_file: str = CALIBRATION_FILE
    """Path to save calibration results"""

    camera_config_file: str = CAMERA_CONFIG
    """Path to camera.json"""

    squares_x: int = 9
    """Number of squares along the X axis of the ChArUco board (default: 9)"""

    squares_y: int = 9
    """Number of squares along the Y axis of the ChArUco board (default: 9)"""

    square_length: float = 0.03
    """Checker square side length in metres (default: 0.03 = 30 mm)"""

    marker_length: float = 0.023
    """ArUco marker side length in metres (default: 0.023 = 23 mm)"""

    dictionary: str = "DICT_6X6_250"
    """ArUco dictionary (default: DICT_6X6_250)"""


@dataclass
class ConvertCommand:
    """Convert raw camera recordings (SVO2/bag) to UnifiedDataset format"""

    data_dir: str = "data"
    """Root data directory (default: ./data); reads from <data_dir>/raw/, writes to <data_dir>/processed/"""

    stereo_method: Literal["zed", "ffs", "tri_stereo"] = "zed"
    """Depth estimation backend for ZED cameras: 'zed' uses the ZED SDK (NEURAL_LIGHT), 'ffs' uses Fast Foundation Stereo, 'tri_stereo' uses the TRI Stereo model"""

    ffs_scale: float = 1.0
    """Input resize scale for FFS inference (e.g. 0.5 halves resolution for speed). Only used when stereo_method=ffs"""

    ffs_iters: int = 8
    """FFS update iterations (default 8, range 4–32). Only used when stereo_method=ffs"""

    tri_stereo_variant: Literal["c32", "c64"] = "c64"
    """TRI Stereo model variant: 'c32' (faster) or 'c64' (higher quality). Only used when stereo_method=tri_stereo"""

    reconvert: bool = False
    """Re-convert all successful demonstrations even if already marked as converted (default: False)"""


@dataclass
class ReplayCommand:
    """Replay recorded follower arm motion"""

    recording_dir: Optional[str] = None
    """Path to a recording episode directory (default: interactive fzf selector)"""

    arms: Literal["bimanual", "single"] = "bimanual"
    """Which arms to replay: both follower arms or left only"""

    speed: float = 1.0
    """Playback speed multiplier (default: 1.0 = real-time, 0.5 = half speed)"""

    source: Literal["raw", "processed"] = "raw"
    """Data source: 'raw' loads joint commands directly from robot_data.npz (no IK); 'processed' loads EE poses from lowdim pkls and solves IK"""


@dataclass
class VisualizeCommand:
    """Visualize a converted UnifiedDataset recording using Rerun"""

    stride: int = 1
    """Log every N-th frame (default: 1)"""

    image_scale: float = 0.25
    """Downsample factor for images and point clouds (default: 0.25)"""

    web: bool = False
    """Serve viewer over HTTP instead of spawning the native app (useful over SSH tunnel)"""

    web_port: int = 9090
    """HTTP port for the web viewer (default: 9090)"""


@dataclass
class ShardifyCommand:
    """Export converted episodes to WebDataset sharded .tar format"""

    data_dir: str = "data"
    """Root data directory (default: ./data); reads from <data_dir>/processed/"""

    output_dir: str = "data/shards"
    """Local output directory for shards (default: data/shards)"""

    task_name: Optional[str] = None
    """Task name for S3 upload path (default: basename of data_dir)"""

    s3_bucket: Optional[str] = None
    """S3 bucket to upload shards to after writing (e.g. my-robot-data)"""

    s3_prefix: str = "yam_datasets"
    """S3 key prefix; shards go to s3://{bucket}/{prefix}/{task_name}/shards/ (default: yam_datasets)"""

    past_lowdim_steps: int = 1
    """Number of past timesteps in the lowdim window (default: 1)"""

    future_lowdim_steps: int = 19
    """Number of future timesteps in the lowdim window (default: 19)"""

    max_padding_left: int = 3
    """Max allowable left-side padding before a sample is filtered (default: 3)"""

    max_padding_right: int = 15
    """Max allowable right-side padding before a sample is filtered (default: 15)"""

    samples_per_shard: int = 100
    """Number of samples per .tar shard (default: 100)"""

    jpeg_quality: int = 95
    """JPEG quality for re-encoded images (default: 95)"""

    resize_images: Optional[str] = "384x384"
    """Resize images to HxW before storing, (default: '384x384')"""

    filter_still_samples: bool = False
    """Filter samples where neither arm moves (default: False)"""

    still_threshold: float = 0.05
    """Max EE movement in metres to consider a sample 'still' (default: 0.05)"""

    fail_on_nan: bool = True
    """Raise an error if NaN values are found in lowdim data (default: True)"""

    stride: int = 3
    """Lowdim/action window step spacing in raw frames (does not affect anchor density or image offsets). 3=10 Hz actions from 30 Hz (default: 3)"""

    stats_stride: int = 10
    """Only feed every N-th sample into the stats accumulators to save memory (default: 10)"""

    max_episodes: int = -1
    """Maximum number of episodes to process, -1 = all (default: -1)"""

    num_workers: int = 1
    """Number of worker processes (default: 1)"""


@dataclass
class ServeCommand:
    """Start the chiral policy server"""

    host: str = "0.0.0.0"
    """WebSocket host to bind to"""

    port: int = 8765
    """WebSocket port to listen on"""

    stereo_method: Literal["zed", "ffs", "tri_stereo"] = "zed"
    """Depth backend: 'zed' (SDK NEURAL_LIGHT), 'ffs' (Fast Foundation Stereo), or 'tri_stereo' (TRI Stereo)"""

    ffs_scale: float = 1.0
    """Input resize scale for FFS inference (e.g. 0.5 halves resolution for speed)"""

    ffs_iters: int = 8
    """FFS update iterations (range 4–32)"""

    tri_stereo_variant: Literal["c32", "c64"] = "c64"
    """TRI Stereo model variant: 'c64' (higher quality) or 'c32' (faster)"""

    max_joint_delta: float = 0.8
    """Maximum allowed joint delta per policy step in radians before server aborts"""

    action_type: Literal["joint", "ee_pose"] = "ee_pose"
    """Action space: 'joint' (14-D joint positions, left then right) or 'ee_pose' (20-D EE poses, IK solved on-the-fly)"""

    no_depth: bool = False
    """Disable depth sensing on ZED cameras (faster, no NEURAL_LIGHT inference)"""

    resize_images: Optional[str] = "384x384"
    """Resize images to HxW before sending to the policy (default: '384x384'). Pass empty string to disable."""

    camera_config_file: str = ""
    """Path to camera.json (default: ~/.config/raiden/camera.json)"""

    calibration_file: str = ""
    """Path to calibration_results.json (default: ~/.config/raiden/calibration_results.json)"""


def _load_spacemouse_config(path: str = SPACEMOUSE_CONFIG) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _print_help() -> None:
    print("Raiden - Toolkit for policy learning with YAM bimanual robot arms")
    print()
    print("Usage: rd <command> [options]")
    print()
    print("Commands:")
    print("  teleop                      Start bimanual teleoperation")
    print("  record                      Record teleoperation demonstrations")
    print(
        "  convert                     Convert SVO2/bag recordings to UnifiedDataset format"
    )
    print("  replay                      Replay recorded follower arm motion")
    print("  visualize                   Visualize a converted recording with Rerun")
    print("  record_calibration_poses    Record robot poses for camera calibration")
    print("  calibrate                   Run camera calibration using recorded poses")
    print(
        "  list_devices                List all connected cameras, arms, and SpaceMouse devices"
    )
    print(
        "  shardify                    Export converted episodes to WebDataset shards"
    )
    print("  console                     Open the interactive metadata console (TUI)")
    print("  reset_can                   Reset CAN interfaces (bring down then up)")
    print(
        "  serve                       Start the chiral policy server for live inference"
    )
    print()
    print("Run 'rd <command> --help' for more information on a command.")


def main():
    """Main entry point for rd command"""
    # Handle subcommands manually for better UX
    if len(sys.argv) > 1:
        subcommand = sys.argv[1]

        if subcommand == "teleop":
            sys.argv.pop(1)
            sm = _load_spacemouse_config()
            command = tyro.cli(
                TeleopCommand,
                description="Start bimanual teleoperation with improved synchronization",
                default=TeleopCommand(
                    spacemouse_path_r=sm.get("path_r", "/dev/hidraw7"),
                    spacemouse_path_l=sm.get("path_l", "/dev/hidraw6"),
                ),
            )
            run_bimanual_teleop(
                bilateral_kp=command.bilateral_kp,
                control=command.control,
                spacemouse_path_r=command.spacemouse_path_r,
                spacemouse_path_l=command.spacemouse_path_l,
                vel_scale=command.vel_scale,
                rot_scale=command.rot_scale,
                invert_rotation=command.invert_rotation,
                arms=command.arms,
            )

        elif subcommand == "record":
            sys.argv.pop(1)
            sm = _load_spacemouse_config()
            command = tyro.cli(
                RecordCommand,
                description="Record a demonstration with cameras and robot data",
                default=RecordCommand(
                    spacemouse_path_r=sm.get("path_r", "/dev/hidraw7"),
                    spacemouse_path_l=sm.get("path_l", "/dev/hidraw6"),
                ),
            )
            run_recording(
                s3_bucket=command.s3_bucket,
                s3_prefix=command.s3_prefix,
                control=command.control,
                spacemouse_path_r=command.spacemouse_path_r,
                spacemouse_path_l=command.spacemouse_path_l,
                vel_scale=command.vel_scale,
                rot_scale=command.rot_scale,
                invert_rotation=command.invert_rotation,
                arms=command.arms,
                data_dir=command.data_dir,
            )

        elif subcommand == "replay":
            sys.argv.pop(1)
            command = tyro.cli(
                ReplayCommand, description="Replay recorded follower arm motion"
            )
            if command.recording_dir is not None:
                from pathlib import Path

                recording_dir = Path(command.recording_dir)
            elif command.source == "processed":
                recording_dir = select_processed_recording()
            else:
                recording_dir = select_recording()
            run_replay(recording_dir, arms=command.arms, speed=command.speed)

        elif subcommand == "list_devices":
            sys.argv.pop(1)
            tyro.cli(
                ListDevicesCommand,
                description="List all connected cameras, robot arms, and SpaceMouse devices",
            )
            from raiden.camera_utils import list_devices

            list_devices()

        elif subcommand == "record_calibration_poses":
            sys.argv.pop(1)
            command = tyro.cli(
                RecordCalibrationPosesCommand,
                description="Record robot poses for camera calibration",
            )
            sm = _load_spacemouse_config()
            run_calibration_pose_recording(
                min_poses=command.min_poses,
                output_file=command.output_file,
                camera_config_file=CAMERA_CONFIG,
                control=command.control,
                spacemouse_path_r=sm.get("path_r", command.spacemouse_path_r),
                spacemouse_path_l=sm.get("path_l", command.spacemouse_path_l),
                vel_scale=command.vel_scale,
                rot_scale=command.rot_scale,
                invert_rotation=command.invert_rotation,
            )

        elif subcommand == "calibrate":
            sys.argv.pop(1)
            command = tyro.cli(
                CalibrateCommand,
                description="Run camera calibration using recorded poses",
            )
            from raiden.calibration.recorder import ChArUcoBoardConfig
            from raiden.camera_config import CameraConfig

            _cfg = CameraConfig(command.camera_config_file)
            charuco_config = ChArUcoBoardConfig(
                squares_x=command.squares_x,
                squares_y=command.squares_y,
                square_length=command.square_length,
                marker_length=command.marker_length,
                dictionary=command.dictionary,
            )
            runner = CalibrationRunner(
                camera_config_file=command.camera_config_file,
                poses_file=command.poses_file,
                output_file=command.output_file,
                charuco_config=charuco_config,
            )
            runner.run_calibration(
                left_wrist_camera_id=_cfg.get_camera_by_role("left_wrist"),
                right_wrist_camera_id=_cfg.get_camera_by_role("right_wrist"),
                scene_camera_ids=_cfg.get_cameras_by_role("scene") or None,
            )

        elif subcommand == "convert":
            sys.argv.pop(1)
            from pathlib import Path as _Path

            command = tyro.cli(
                ConvertCommand,
                description="Convert raw camera recordings (SVO2/bag) to PNG frames and depth maps",
            )
            for task_dir in select_tasks(command.data_dir):
                convert_task(
                    task_dir,
                    stereo_method=command.stereo_method,
                    ffs_scale=command.ffs_scale,
                    ffs_iters=command.ffs_iters,
                    reconvert=command.reconvert,
                    processed_base=str(_Path(command.data_dir) / "processed"),
                    tri_stereo_variant=command.tri_stereo_variant,
                )

        elif subcommand == "visualize":
            sys.argv.pop(1)
            command = tyro.cli(
                VisualizeCommand,
                description="Visualize a converted UnifiedDataset recording using Rerun",
            )
            recording_dir, episode = select_task_and_episode()
            visualize_recording(
                recording_dir=recording_dir,
                episode=episode,
                stride=command.stride,
                image_scale=command.image_scale,
                web=command.web,
                web_port=command.web_port,
            )

        elif subcommand == "shardify":
            sys.argv.pop(1)
            command = tyro.cli(
                ShardifyCommand,
                description="Export converted episodes to WebDataset sharded .tar format",
            )
            from pathlib import Path as _Path

            selected_tasks = select_processed_task(command.data_dir)

            resize: tuple | None = None
            if command.resize_images:
                h, w = command.resize_images.split("x")
                resize = (int(h), int(w))

            for task_dir, episode_dirs in selected_tasks:
                print(f"Found {len(episode_dirs)} episodes in {task_dir}")

                task_name = command.task_name or task_dir.name
                s3_full_prefix = (
                    f"{command.s3_prefix}/{task_name}/shards"
                    if command.s3_bucket
                    else None
                )

                cfg = ShardifyConfig(
                    output_dir=_Path(command.output_dir) / task_name,
                    past_lowdim_steps=command.past_lowdim_steps,
                    future_lowdim_steps=command.future_lowdim_steps,
                    max_padding_left=command.max_padding_left,
                    max_padding_right=command.max_padding_right,
                    samples_per_shard=command.samples_per_shard,
                    jpeg_quality=command.jpeg_quality,
                    resize_images_size=resize,
                    filter_still_samples=command.filter_still_samples,
                    still_threshold=command.still_threshold,
                    fail_on_nan=command.fail_on_nan,
                    stride=command.stride,
                    max_episodes_to_process=command.max_episodes,
                    num_workers=command.num_workers,
                    stats_stride=command.stats_stride,
                )
                run_shardify(
                    episode_dirs,
                    cfg,
                    s3_bucket=command.s3_bucket,
                    s3_prefix=s3_full_prefix,
                )

        elif subcommand == "console":
            sys.argv.pop(1)
            tyro.cli(
                ConsoleCommand,
                description=(
                    "Interactive TUI for managing Raiden demonstrations and metadata.\n\n"
                    "Tabs:\n"
                    "  Dashboard       Live counts and per-task / per-teacher breakdown\n"
                    "  Demonstrations  Full list of recorded episodes — mark success or failure,\n"
                    "                  reassign teacher / task, delete entries\n"
                    "  Teachers        Add, rename, or remove teacher records\n"
                    "  Tasks           Add, edit, or remove task definitions\n\n"
                    "Workflow:\n"
                    "  Demonstrations are marked Success or Failure directly from the\n"
                    "  teaching hardware during recording. Open the console only to\n"
                    "  correct mistakes. Only successful demonstrations are converted\n"
                    "  when you run  rd convert.\n\n"
                    "Marking during recording (without opening the console):\n"
                    "  Left pedal / leader button          Start or stop recording\n"
                    "  Middle pedal / top leader button    Mark as Success\n"
                    "  Right pedal / bottom leader button  Mark as Failure\n\n"
                    "Key bindings inside the console:\n"
                    "  r  Refresh all panes\n"
                    "  s  Settings (DB path and record counts)\n"
                    "  ?  Help screen\n"
                    "  q  Quit"
                ),
            )
            from raiden.tui.app import RaidenConsole

            app = RaidenConsole()
            app.run()

        elif subcommand == "reset_can":
            sys.argv.pop(1)
            command = tyro.cli(
                ResetCanCommand,
                description="Reset CAN interfaces (bring down then up at the target bitrate)",
            )
            from raiden.robot.controller import list_can_interfaces, reset_can_interface

            interfaces = command.interfaces or list_can_interfaces()
            if not interfaces:
                print("No CAN interfaces found.")
                sys.exit(1)
            print(
                f"Resetting {len(interfaces)} CAN interface(s) at {_CAN_BITRATE} bps..."
            )
            all_ok = True
            for iface in interfaces:
                ok = reset_can_interface(iface, bitrate=_CAN_BITRATE)
                if ok:
                    print(f"  ✓ {iface}")
                else:
                    all_ok = False
            if all_ok:
                print("Done.")
            else:
                sys.exit(1)

        elif subcommand == "serve":
            sys.argv.pop(1)
            command = tyro.cli(
                ServeCommand,
                description="Start the chiral policy server for live inference",
            )
            from raiden.server import run_server

            resize: tuple | None = None
            if command.resize_images:
                h, w = command.resize_images.split("x")
                resize = (int(h), int(w))
            run_server(
                camera_config_file=command.camera_config_file,
                calibration_file=command.calibration_file,
                host=command.host,
                port=command.port,
                stereo_method=command.stereo_method,
                ffs_scale=command.ffs_scale,
                ffs_iters=command.ffs_iters,
                tri_stereo_variant=command.tri_stereo_variant,
                max_joint_delta=command.max_joint_delta,
                action_type=command.action_type,
                no_depth=command.no_depth,
                resize_images_size=resize,
            )

        else:
            _print_help()
            sys.exit(1)
    else:
        _print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
