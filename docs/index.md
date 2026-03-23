# Raiden

<link rel="stylesheet" href="https://cdn.plyr.io/3.7.8/plyr.css">
<video id="teaser" playsinline controls poster="https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/raiden/teaser_thumbnail.jpg">
  <source src="https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/raiden/teaser.mp4" type="video/mp4">
</video>
<script src="https://cdn.plyr.io/3.7.8/plyr.js"></script>
<script>new Plyr('#teaser');</script>

Raiden is an end-to-end data collection toolkit for YAM robot arms. It covers
the full pipeline from hardware setup to policy-ready datasets: camera
calibration, teleoperation, multi-camera recording, dataset conversion, and
visualization.

[Get started :material-arrow-right:](guide/index.md){ .md-button .md-button--primary }

Key features:

- **Flexible control** - leader-follower teleoperation or SpaceMouse end-effector
  control, in bimanual or single-arm configurations.
- **Manipulability-aware IK** - uses [PyRoki](https://github.com/chungmin99/pyroki)
  and [J-Parse](https://jparse-manip.github.io/) for smooth and singularity-aware control.
- **Multiple depth backends** - IR structured light (RealSense), ZED SDK stereo,
  TRI Stereo,
  and [Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo)
  for high-quality depth tailored to manipulation scenes.
- **Heterogeneous cameras** - mix ZED and Intel RealSense cameras freely in a
  single session, across scene and wrist roles.
- **Automated extrinsic calibration** - hand-eye calibration for wrist cameras
  and static extrinsic estimation for scene cameras via ChArUco boards.
- **Metadata console** - a terminal UI (`rd console`) for reviewing
  demonstrations, correcting success/failure labels, and managing tasks and
  teachers.
- **Policy-ready output** - converts recordings to a simple, flat file format
  with synchronized frames, per-frame extrinsics, and interpolated joint poses,
  ready to plug into policy training frameworks.

## Supported configurations

**Arm setups**

| Setup | Follower arms | Leader arms | CAN interfaces | |
|---|---|---|---|---|
| Bimanual | 2 | 2 (one per side) | `can_follower_l`, `can_follower_r`, `can_leader_l`, `can_leader_r` | **Recommended** |
| Single arm | 1 | 1 | `can_follower_l`, `can_leader_l` | |

Leader arms are only required for leader-follower control (`--control leader`).
With SpaceMouse control (`--control spacemouse`), only follower arms are needed.
In single-arm mode the active arm is always named **left** for consistency. The
global coordinate origin is always the left-arm base frame in both setups.

**Control modes**

| Mode | Flag | Bimanual | Single arm | |
|---|---|---|---|---|
| Leader-follower | `--control leader` (default) | 2 leader + 2 follower arms | 1 leader + 1 follower arm | **Recommended** |
| SpaceMouse | `--control spacemouse` | 2 SpaceMice + 2 follower arms | 1 SpaceMouse + 1 follower arm | |

**Camera configurations**

Raiden supports heterogeneous camera setups — ZED and RealSense cameras can
be mixed freely within the same session.

| Configuration | | |
|---|---|---|
| 2 × ZED Mini (wrists) + 1 × ZED 2i (scene) | Bimanual, best synchronization | **Recommended** |
| 1 × ZED Mini (left wrist) + 1 × ZED 2i (scene) | Single arm | |
| Intel RealSense D400 series (any role) | No GPU required; see [Hardware Setup](guide/hardware.md#camera-configuration) for caveats | |

**Depth estimation**

| Method | Cameras | Description |
|---|---|---|
| IR structured light | Intel RealSense D400 series | On-device active IR depth |
| ZED SDK stereo | ZED cameras | NEURAL_LIGHT stereo depth from the ZED SDK (requires GPU) |
| [TRI Stereo](https://sites.google.com/view/stereoformobilemanipulation) | ZED cameras | TRI's learned stereo depth model tailored for robot manipulation scenes; `c32` and `c64` variants with ONNX and TensorRT backends (optional, requires CUDA GPU) |
| [Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo) | ZED cameras | Foundation model stereo depth; higher quality at object boundaries and thin structures (optional, requires CUDA GPU) |

<video controls loop autoplay muted style="width:100%">
  <source src="https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/raiden/compare.mp4" type="video/mp4">
</video>

## Roadmap

The following features are coming soon:

- **Fin-ray gripper support** — support for fin-ray compliant grippers, which conform to object shapes for robust and gentle grasping.
- **Policy training and inference** — built-in integration for policy training pipelines and closed-loop inference.

## Acknowledgments

- **[PyRoki](https://github.com/chungmin99/pyroki)** - Raiden uses PyRoki for
  flexible inverse kinematics, incorporating constraints such as manipulability,
  which gives smoother and better-conditioned motion near singularities. Thanks
  to the PyRoki authors.

- **[J-Parse](https://jparse-manip.github.io/)** - enables manipulability-aware
  IK by efficiently computing task-space Jacobians for articulated robots.
  Thanks to the J-Parse authors for making this available.

- **[TRI Stereo Depth](https://sites.google.com/view/stereoformobilemanipulation)** - optional depth backend for ZED cameras. A learned stereo depth model developed at Toyota Research Institute, tailored for robot manipulation scenes.

- **[Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo)** - optional depth backend for ZED cameras. A foundation model for stereo depth estimation that produces higher-quality depth maps than the ZED SDK, particularly at object boundaries and on thin structures.

## Disclaimer

Raiden is research software provided **as-is**, without warranty of any kind.
Operating robotic arms involves inherent physical risks. The authors and Toyota
Research Institute accept **no liability** for any damage to property, equipment,
or persons arising from the use of this software. See [Safety](guide/safety.md)
for recommended precautions.

## Citation

```bibtex
@misc{raiden2026,
  title  = {{RAIDEN}: A Toolkit for Policy Learning with {YAM} Bimanual Robot Arms},
  author = {Iwase, Shun and Miller, Patrick and Yao, Jonathan and Jatavallabhula, {Krishna Murthy} and Zakharov, Sergey},
  year   = {2026},
}
```
