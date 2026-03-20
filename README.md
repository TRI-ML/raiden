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

## Installation

**Install the `rd` command:**

```bash
uv tool install .
```

**ZED cameras (optional)** — install the [ZED SDK](https://www.stereolabs.com/developers/), then:

```bash
uv run python scripts/install_pyzed.py
uv sync --extra zed
```

**TRI Stereo (optional)** — TRI's learned stereo depth model for ZED cameras:

```bash
git lfs install
git lfs pull
uv sync --extra mmt
```

**Fast Foundation Stereo (optional)** — foundation model stereo depth for ZED cameras:

```bash
uv run python scripts/install_ffs.py
uv sync --extra ffs
```

For TensorRT acceleration, see the [TensorRT guide](https://tri-ml.github.io/raiden/guide/tensorrt/).

## Commands

| Command | Description |
|---|---|
| `rd list_devices` | List all connected cameras, arms, and SpaceMouse devices |
| `rd calibrate` | Calibrate cameras (hand-eye + scene extrinsics) |
| `rd teleop` | Teleoperate arms without recording |
| `rd record` | Record teleoperation demonstrations |
| `rd console` | Browse and correct demonstration metadata in a terminal UI |
| `rd convert` | Convert successful recordings to a structured dataset |
| `rd visualize` | Visualize a converted recording with Rerun |

Run `rd <command> --help` for all options.

## Roadmap

The following features are coming soon:

- **Fin-ray gripper support** — support for fin-ray compliant grippers, which conform to object shapes for robust and gentle grasping.
- **Policy training and inference** — built-in integration for policy training pipelines and closed-loop inference.

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
