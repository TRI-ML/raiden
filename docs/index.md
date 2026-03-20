# Raiden

<link rel="stylesheet" href="https://cdn.plyr.io/3.7.8/plyr.css">
<video id="teaser" playsinline controls poster="https://tri-ml.github.io/raiden/assets/teaser_thumbnail.jpg">
  <source src="https://tri-ml.github.io/raiden/assets/teaser.mov" type="video/mp4">
</video>
<script src="https://cdn.plyr.io/3.7.8/plyr.js"></script>
<script>new Plyr('#teaser');</script>

Raiden is an end-to-end data collection toolkit for YAM robot arms. It covers
the full pipeline from hardware setup to policy-ready datasets: camera
calibration, teleoperation, multi-camera recording, dataset conversion, and
visualization.

Key features:

- **Heterogeneous cameras** - mix ZED and Intel RealSense cameras freely in a
  single session, across scene and wrist roles.
- **Multiple depth backends** - IR structured light (RealSense), ZED SDK stereo,
  and [Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo)
  for high-quality depth at object boundaries.
- **Flexible control** - leader-follower teleoperation or SpaceMouse end-effector
  control, in bimanual or single-arm configurations.
- **Manipulability-aware IK** - uses [PyRoki](https://github.com/chungmin99/pyroki)
  and [J-Parse](https://jparse-manip.github.io/) for smooth, singularity-robust
  inverse kinematics.
- **Metadata console** - a terminal UI (`rd console`) for reviewing
  demonstrations, correcting success/failure labels, and managing tasks and
  teachers.
- **Policy-ready output** - converts recordings to a simple, flat file format
  with synchronized frames, per-frame extrinsics, and interpolated joint poses,
  ready to plug into policy training frameworks.

## Supported configurations

**Arm setups**

| Setup | Arms | CAN interfaces |
|---|---|---|
| Bimanual | 2 follower + 2 leader (one pair per side) | `can_follower_l`, `can_follower_r`, `can_leader_l`, `can_leader_r` |
| Single arm | 1 follower + 1 leader | `can_follower_l`, `can_leader_l` |

Leader arms are only required for leader-follower control. With SpaceMouse
control, only follower arms are needed. In single-arm mode the active arm is
always named **left** for consistency. The global coordinate origin is always
the left-arm base frame in both setups.

**Control modes**

| Mode | Flag | Bimanual | Single arm |
|---|---|---|---|
| Leader-follower | `--control leader` (default) | 2 leader + 2 follower arms | 1 leader + 1 follower arm |
| SpaceMouse | `--control spacemouse` | 2 SpaceMice + 2 follower arms | 1 SpaceMouse + 1 follower arm |

**Camera configurations**

Raiden supports heterogeneous camera setups - ZED and RealSense cameras can
be mixed freely within the same session. The primary tested configuration is
**2 × ZED Mini** (wrists) + **1 × ZED 2i** (scene) with a bimanual arm setup.

| Role | Supported cameras |
|---|---|
| `scene` | ZED 2i (or any ZED), Intel RealSense D400 series; multiple scene cameras supported |
| `left_wrist` | ZED Mini, Intel RealSense D400 series |
| `right_wrist` | ZED Mini, Intel RealSense D400 series |

**Depth estimation**

| Method | Cameras | Description |
|---|---|---|
| IR structured light | Intel RealSense D400 series | On-device active IR depth |
| ZED SDK stereo | ZED cameras | NEURAL_LIGHT stereo depth from the ZED SDK (requires GPU) |
| [Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo) | ZED cameras | Foundation model stereo depth; higher quality at object boundaries and thin structures (optional, requires CUDA GPU) |

## Roadmap

The following features are coming soon:

- **TRI learned stereo depth** — integration of TRI's [learned stereo depth model for mobile manipulation](https://sites.google.com/view/stereoformobilemanipulation) as an additional depth backend for ZED cameras, offering high-quality depth tailored for robot manipulation scenes.
- **Fin-ray gripper support** — support for fin-ray compliant grippers, which conform to object shapes for robust and gentle grasping.
- **Policy training and inference** — built-in integration for policy training pipelines and closed-loop inference.

## Acknowledgments

- **[PyRoki](https://github.com/chungmin99/pyroki)** - Raiden uses PyRoki for
  inverse kinematics instead of mink. PyRoki's IK formulation considers
  manipulability, which gives smoother and better-conditioned motion near
  singularities. Thanks to the PyRoki authors, and in particular to the
  contributor behind [PR #85](https://github.com/chungmin99/pyroki/pull/85).

- **[Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo)** - optional depth backend for ZED cameras. A foundation model for stereo depth estimation that produces higher-quality depth maps than the ZED SDK, particularly at object boundaries and on thin structures.

- **[J-Parse](https://jparse-manip.github.io/)** - the Jacobian parsing
  utility integrated via the above PR. J-Parse enables manipulability-aware
  IK by efficiently computing task-space Jacobians for articulated robots.
  Thanks to the J-Parse authors for making this available.

## Citation

```bibtex
@misc{raiden2026,
  title  = {{RAIDEN}: A Toolkit for Policy Learning with {YAM} Bimanual Robot Arms},
  author = {Iwase, Shun and Miller, Patrcik and Yao, Jonathan and Zakharov, Sergey},
  year   = {2026},
}
```
