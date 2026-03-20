# Raiden

[![Docs](https://github.com/TRI-ML/raiden/actions/workflows/docs.yml/badge.svg)](https://github.com/TRI-ML/raiden/actions/workflows/docs.yml)
[![Lint](https://github.com/TRI-ML/raiden/actions/workflows/lint.yml/badge.svg)](https://github.com/TRI-ML/raiden/actions/workflows/lint.yml)

Raiden is an end-to-end data collection toolkit for YAM robot arms. It covers
the full pipeline from hardware setup to policy-ready datasets: camera
calibration, teleoperation, multi-camera recording, dataset conversion, and
visualization.

**[Documentation](https://tri-ml.github.io/raiden/)**

**Key features**

- **Heterogeneous cameras** — mix ZED and Intel RealSense cameras freely in a single session, across scene and wrist roles.
- **Multiple depth backends** — IR structured light (RealSense), ZED SDK stereo, and [Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo) for high-quality depth at object boundaries.
- **Flexible control** — leader-follower teleoperation or SpaceMouse end-effector control, in bimanual or single-arm configurations.
- **Manipulability-aware IK** — uses [PyRoki](https://github.com/chungmin99/pyroki) and [J-Parse](https://jparse-manip.github.io/) for smooth, singularity-robust inverse kinematics.
- **Metadata console** — terminal UI to browse and correct demonstration status, tasks, and teachers.
- **Policy-ready output** — synchronized frames, per-frame extrinsics, and interpolated joint poses, ready for policy training frameworks.

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

**Fast Foundation Stereo (optional)** — higher-quality depth for ZED cameras:

```bash
uv run python scripts/install_ffs.py
uv sync --extra ffs
# Place the pretrained *.pth checkpoint in data/weights/
```

For TensorRT acceleration, see the [TensorRT guide](docs/guide/tensorrt.md).

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

- **TRI learned stereo depth** — integration of TRI's [learned stereo depth model for mobile manipulation](https://sites.google.com/view/stereoformobilemanipulation) as an additional depth backend for ZED cameras, offering high-quality depth tailored for robot manipulation scenes.
- **Fin-ray gripper support** — support for fin-ray compliant grippers, which conform to object shapes for robust and gentle grasping.
- **Policy training and inference** — built-in integration for policy training pipelines and closed-loop inference.

## Citation

```bibtex
@misc{raiden2026,
  title  = {{RAIDEN}: A Toolkit for Policy Learning with {YAM} Bimanual Robot Arms},
  author = {Iwase, Shun and Miller, Patrick and Yao, Jonathan and Zakharov, Sergey},
  year   = {2026},
}
```
