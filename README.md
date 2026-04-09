# Raiden

Raiden is an end-to-end data collection toolkit for YAM robot arms. It covers
the full pipeline from hardware setup to policy-ready datasets: camera
calibration, teleoperation, multi-camera recording, dataset conversion, and
visualization.

**[Documentation](https://tri-ml.github.io/raiden/)** · **[Get started](https://tri-ml.github.io/raiden/guide/)**

**Key features**

- **Flexible control** — leader-follower teleoperation or SpaceMouse end-effector control, in bimanual or single-arm configurations.
- **Manipulability-aware IK** — uses [PyRoki](https://github.com/chungmin99/pyroki) and [J-Parse](https://jparse-manip.github.io/) for smooth and singularity-aware control.
- **Multiple depth backends** — IR structured light (RealSense), ZED SDK stereo, TRI Stereo, and [Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo) for high-quality depth tailored to manipulation scenes.
- **Heterogeneous cameras** — mix ZED and Intel RealSense cameras freely in a single session, across scene and wrist roles.
- **Automated extrinsic calibration** — hand-eye calibration for wrist cameras and static extrinsic estimation for scene cameras via ChArUco boards.
- **Metadata console** — a terminal UI (`rd console`) for reviewing demonstrations, correcting success/failure labels, and managing tasks and teachers.
- **Policy-ready output** — converts recordings to a simple, flat file format with synchronized frames, per-frame extrinsics, and interpolated joint poses, ready to plug into policy training frameworks.
- **[Fin-ray gripper support](https://tri-ml.github.io/raiden/guide/hardware/#fin-ray-gripper)** — 3D-printable compliant grippers that conform to object shapes for robust and gentle grasping.

## Installation

See the **[Installation guide](https://tri-ml.github.io/raiden/guide/installation/)** for full instructions.

## Commands

| Command | Description |
|---|---|
| `rd list_devices` | List all connected cameras, arms, and SpaceMouse devices |
| `rd record_calibration_poses` | Record robot poses for camera calibration |
| `rd calibrate` | Calibrate cameras (hand-eye + scene extrinsics) |
| `rd teleop` | Teleoperate arms without recording |
| `rd record` | Record teleoperation demonstrations |
| `rd replay` | Replay recorded follower arm motion |
| `rd console` | Browse and correct demonstration metadata in a terminal UI |
| `rd convert` | Convert successful recordings to a structured dataset |
| `rd shardify` | Export converted episodes to WebDataset shards |
| `rd visualize` | Visualize a converted recording with Rerun |
| `rd serve` | Start the policy server for live inference |
| `rd make_ffs_onnx` | Export Fast Foundation Stereo model to ONNX / TensorRT engines |
| `rd make_tri_stereo_engine` | Compile TRI Stereo TensorRT engine from ONNX model |

Run `rd <command> --help` for all options.

## Roadmap

The following features are coming soon:

- **Policy training and inference** — built-in integration for policy training pipelines and closed-loop inference.
- **LeRobot format converter** — export converted episodes to the [LeRobot](https://github.com/huggingface/lerobot) dataset format for compatibility with the Hugging Face ecosystem.
- **Initial scene condition management** — set up and save named initial scene conditions in the console to enable reproducible, side-by-side comparison of multiple policies under identical starting states.

## Disclaimer

Raiden is research software provided **as-is**, without warranty of any kind. Operating robotic arms involves inherent physical risks. The authors and Toyota Research Institute accept **no liability** for any damage to property, equipment, or persons arising from the use of this software.

## Citation

```bibtex
@misc{raiden2026,
  title  = {{RAIDEN}: A Toolkit for Policy Learning with {YAM} Bimanual Robot Arms},
  author = {Iwase, Shun and Miller, Patrick and Yao, Jonathan and Jatavallabhula, {Krishna Murthy} and Zakharov, Sergey},
  year   = {2026},
}
```
