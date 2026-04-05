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
    WEIGHTS_DIR,
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
class MakeFfsOnnxCommand:
    """Export Fast Foundation Stereo model to ONNX (and optionally TensorRT engines)"""

    model_dir: Optional[str] = None
    """Path to .pth checkpoint (default: most recently modified *.pth in ~/.config/raiden/weights/)"""

    save_path: Optional[str] = None
    """Directory to write ONNX files and onnx.yaml (default: ~/.config/raiden/weights/onnx/)"""

    height: int = 448
    """ONNX input height in pixels, must be divisible by 32"""

    width: int = 640
    """ONNX input width in pixels, must be divisible by 32"""

    valid_iters: int = 8
    """Number of GRU update iterations during forward pass"""

    max_disp: int = 192
    """Max disparity for the geometry encoding volume"""

    build_engines: bool = False
    """After exporting ONNX, also compile TensorRT engines via the Python API"""


@dataclass
class MakeTriStereoEngineCommand:
    """Compile TRI Stereo TensorRT engine from ONNX model"""

    variant: Literal["c32", "c64"] = "c64"
    """Model variant"""

    onnx: Optional[str] = None
    """Input .onnx file (default: weights/tri_stereo/stereo_<variant>.onnx)"""

    engine: Optional[str] = None
    """Output .engine file (default: weights/tri_stereo/stereo_<variant>.engine)"""

    fp16: bool = True
    """Use FP16 precision"""


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

    stride: int = 1
    """Subsample every N-th frame to match shardify stride (default: 1 = native rate, 3 = 10 Hz from 30 Hz recordings)"""

    visualize: bool = False
    """Stream commanded and actual EE trajectories to a Rerun web viewer"""

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

    use_depth: bool = False
    """Include depth images (.depth.png) in shards (default: False)"""


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

    visualize: bool = False
    """Stream camera images to a Rerun web viewer at 30 FPS (accessible via browser or SSH tunnel)"""

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
    print(
        "  make_ffs_onnx               Export Fast Foundation Stereo model to ONNX / TensorRT engines"
    )
    print(
        "  make_tri_stereo_engine      Compile TRI Stereo TensorRT engine from ONNX model"
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
            run_replay(
                recording_dir,
                arms=command.arms,
                speed=command.speed,
                stride=command.stride,
                visualize=command.visualize,
            )

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
                    use_depth=command.use_depth,
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
                description="Reset CAN interfaces (bring down then up)",
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
                visualize=command.visualize,
            )

        elif subcommand == "make_ffs_onnx":
            sys.argv.pop(1)
            command = tyro.cli(
                MakeFfsOnnxCommand,
                description="Export Fast Foundation Stereo model to ONNX (and optionally TensorRT engines)",
            )
            import os
            from pathlib import Path as _Path  # noqa: PLC0415

            import onnx
            import torch
            import yaml
            from omegaconf import OmegaConf

            os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
            os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

            _ffs_dir = (
                _Path(__file__).parent.parent / "third_party" / "Fast-FoundationStereo"
            )
            sys.path.insert(0, str(_ffs_dir))
            from core.foundation_stereo import (  # noqa: PLC0415
                TrtFeatureRunner,
                TrtPostRunner,
                build_gwc_volume_triton,
            )

            save_path = _Path(command.save_path or str(WEIGHTS_DIR / "onnx"))
            save_path.mkdir(parents=True, exist_ok=True)

            assert command.height % 32 == 0 and command.width % 32 == 0, (
                f"height ({command.height}) and width ({command.width}) must both be divisible by 32"
            )

            model_dir = command.model_dir
            if model_dir is None:
                ckpts = sorted(
                    WEIGHTS_DIR.glob("*.pth"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not ckpts:
                    print(f"No *.pth checkpoint found in '{WEIGHTS_DIR}'")
                    sys.exit(1)
                model_dir = str(ckpts[0])

            print(f"Checkpoint : {model_dir}")
            print(f"Input size : {command.height} x {command.width}")
            print(f"Output dir : {save_path}")

            torch.autograd.set_grad_enabled(False)
            print("\nLoading model...")
            model = torch.load(model_dir, map_location="cpu", weights_only=False)
            model.args.max_disp = command.max_disp
            model.args.valid_iters = command.valid_iters
            model.cuda().eval()

            feature_runner = TrtFeatureRunner(model).cuda().eval()
            post_runner = TrtPostRunner(model).cuda().eval()

            left_img = (
                torch.randn(1, 3, command.height, command.width).cuda().float() * 255
            )
            right_img = (
                torch.randn(1, 3, command.height, command.width).cuda().float() * 255
            )

            feature_onnx_path = save_path / "feature_runner.onnx"
            print("\nExporting feature_runner.onnx ...")
            torch.onnx.export(
                feature_runner,
                (left_img, right_img),
                str(feature_onnx_path),
                opset_version=17,
                input_names=["left", "right"],
                output_names=[
                    "features_left_04",
                    "features_left_08",
                    "features_left_16",
                    "features_left_32",
                    "features_right_04",
                    "stem_2x",
                ],
                do_constant_folding=True,
            )
            _data_file = feature_onnx_path.with_suffix(".onnx.data")
            if _data_file.exists():
                print("  Merging external data into single file ...")
                _m = onnx.load(str(feature_onnx_path))
                onnx.save(_m, str(feature_onnx_path))
                _data_file.unlink(missing_ok=True)

            print("Exporting post_runner.onnx ...")
            (
                features_left_04,
                features_left_08,
                features_left_16,
                features_left_32,
                features_right_04,
                stem_2x,
            ) = feature_runner(left_img, right_img)
            gwc_volume = build_gwc_volume_triton(
                features_left_04.half(),
                features_right_04.half(),
                command.max_disp // 4,
                model.cv_group,
            )
            torch.onnx.export(
                post_runner,
                (
                    features_left_04,
                    features_left_08,
                    features_left_16,
                    features_left_32,
                    features_right_04,
                    stem_2x,
                    gwc_volume,
                ),
                str(save_path / "post_runner.onnx"),
                opset_version=17,
                input_names=[
                    "features_left_04",
                    "features_left_08",
                    "features_left_16",
                    "features_left_32",
                    "features_right_04",
                    "stem_2x",
                    "gwc_volume",
                ],
                output_names=["disp"],
                do_constant_folding=True,
                dynamo=False,
            )

            cfg = OmegaConf.to_container(model.args)
            cfg["image_h"] = command.height
            cfg["image_w"] = command.width
            cfg["cv_group"] = model.cv_group
            with open(save_path / "onnx.yaml", "w") as f:
                yaml.safe_dump(cfg, f)

            print(f"\nDone. Files written to {save_path}/")

            if command.build_engines:
                import tensorrt as trt  # noqa: PLC0415

                trt_logger = trt.Logger(trt.Logger.WARNING)
                trt.init_libnvinfer_plugins(trt_logger, "")
                for name in ("feature_runner", "post_runner"):
                    onnx_path = save_path / f"{name}.onnx"
                    engine_path = save_path / f"{name}.engine"
                    print(f"\nBuilding {name}.engine (may take several minutes)...")
                    builder = trt.Builder(trt_logger)
                    network = builder.create_network()
                    parser = trt.OnnxParser(network, trt_logger)
                    with open(onnx_path, "rb") as f:
                        ok = parser.parse(f.read())
                    if not ok:
                        for i in range(parser.num_errors):
                            print(f"  Parse error: {parser.get_error(i)}")
                        print(f"Failed to parse {onnx_path}")
                        sys.exit(1)
                    config = builder.create_builder_config()
                    config.set_flag(trt.BuilderFlag.FP16)
                    engine_mem = builder.build_serialized_network(network, config)
                    if engine_mem is None:
                        print(f"TensorRT failed to build engine for {name}")
                        sys.exit(1)
                    engine_bytes = bytes(engine_mem)
                    print(f"  Engine size: {len(engine_bytes) / 1024 / 1024:.1f} MiB")
                    with open(engine_path, "wb") as f:
                        f.write(engine_bytes)
                    print(f"  Saved {engine_path}")
                print(
                    "\nTensorRT engines ready. rd convert will use them automatically."
                )

        elif subcommand == "make_tri_stereo_engine":
            sys.argv.pop(1)
            command = tyro.cli(
                MakeTriStereoEngineCommand,
                description="Compile TRI Stereo TensorRT engine from ONNX model",
            )
            import tensorrt as trt  # noqa: PLC0415
            from pathlib import Path as _Path  # noqa: PLC0415

            _tri_stereo_weights = (
                _Path(__file__).parent.parent / "weights" / "tri_stereo"
            )
            onnx_path = (
                _Path(command.onnx)
                if command.onnx
                else _tri_stereo_weights / f"stereo_{command.variant}.onnx"
            )
            engine_path = (
                _Path(command.engine)
                if command.engine
                else _tri_stereo_weights / f"stereo_{command.variant}.engine"
            )

            if not onnx_path.exists():
                print(f"ONNX model not found: {onnx_path}")
                print("Run: git lfs pull")
                sys.exit(1)

            print(f"ONNX    : {onnx_path}")
            print(f"Engine  : {engine_path}")
            print(f"FP16    : {command.fp16}")
            print("\nBuilding TensorRT engine (may take several minutes)...")

            trt_logger = trt.Logger(trt.Logger.ERROR)
            trt_builder = trt.Builder(trt_logger)
            trt_network = trt_builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            trt_parser = trt.OnnxParser(trt_network, trt_logger)
            if not trt_parser.parse_from_file(str(onnx_path)):
                print(f"Failed to parse ONNX model: {onnx_path}")
                sys.exit(1)
            trt_config = trt_builder.create_builder_config()
            if command.fp16:
                trt_config.set_flag(trt.BuilderFlag.FP16)
            serialized = trt_builder.build_serialized_network(trt_network, trt_config)
            if serialized is None:
                print("TensorRT engine build failed.")
                sys.exit(1)
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            with open(engine_path, "wb") as f:
                f.write(serialized)
            print(f"\nSaved engine to {engine_path}")

        else:
            _print_help()
            sys.exit(1)
    else:
        _print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
