# Visualization

<video controls loop autoplay muted style="width:100%">
  <source src="https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/raiden/visualization.mp4" type="video/mp4">
</video>

The `rd visualize` command opens a converted episode in the
[Rerun](https://rerun.io) viewer. It logs RGB images, plasma-colorized depth
maps, camera frustums, colored 3-D point clouds, and end-effector action
scalars.

Data must be converted first with `rd convert` before it can be visualized.

## Usage

Running `rd visualize` with no arguments launches an interactive fzf selector:

```bash
rd visualize
```

You will be prompted to choose a task, then an episode:

```
  Visualize task>
  pick_purrito
```

```
  Select episode>
  0000  (pick_purrito_20260303_115816)
  0001  (pick_purrito_20260303_120053)
```

Options can be passed alongside the interactive flow:

```bash
# Every other frame
rd visualize --stride 2

# Full resolution (slower)
rd visualize --image-scale 1.0
```

## Options

| Option | Default | Description |
|---|---|---|
| `--stride` | `1` | Log every N-th frame |
| `--image-scale` | `0.25` | Downsample factor applied to images and point clouds |

## What is logged

| Rerun path | Content |
|---|---|
| `world/cameras/<name>` | Camera pose (`Transform3D`) + intrinsics (`Pinhole`) + RGB texture |
| `world/cameras/<name>/rgb` | RGB image in the 2-D Spatial View |
| `world/cameras/<name>/depth` | Plasma-colorized inverse-depth image |
| `world/points/<name>` | Colored world-space point cloud unprojected from depth |
| `action/right/<label>` | Right end-effector pose + gripper scalar (`pos(3) + rot_mat_flat(9) + gripper(1)`) |
| `action/left/<label>` | Left end-effector pose + gripper scalar |
| `trajectory/right` | Full right-arm EE trajectory as a static 3-D line (red) |
| `trajectory/left` | Full left-arm EE trajectory as a static 3-D line (blue) |
| `trajectory/right/current` | Current right-arm EE position (moves with timeline) |
| `trajectory/left/current` | Current left-arm EE position (moves with timeline) |
| `info` | Recording metadata text |

### Depth colorization

Depth maps are visualized using an inverse-depth plasma colormap (closer
objects appear brighter). The colormap is normalized to the 95th percentile of
valid depth values, and invalid pixels (zero depth) are shown as black.

### Point clouds

World-space point clouds are reconstructed by unprojecting the downsampled
depth map through the per-frame camera intrinsics and extrinsics. Points are
colored by the corresponding RGB pixel. Invalid points (non-finite or at the
world origin) are discarded.

### Action scalars

The `action` key in each lowdim file contains the end-effector 6-D pose and
gripper state for both arms, expressed in the **left-arm base frame** (the
global coordinate origin - see [Arm configuration](quickstart.md#arm-configuration)):

```
right: pos(3)  +  rot_mat_flat(9)  +  gripper(1)   # indices  0–12
left:  pos(3)  +  rot_mat_flat(9)  +  gripper(1)   # indices 13–25
```

The rotation is a **3×3 rotation matrix flattened row-major** (9 values) —
no gimbal lock or wrap-around discontinuities.

Use the Rerun timeline to scrub through frames. The `frame` timeline
corresponds directly to the frame index in the converted episode.
