# Converting recordings

The `rd convert` command turns a raw recording directory (containing
`.svo2` / `.bag` camera files and `robot_data.npz`) into a
a structured dataset directly consumable by policy-training frameworks.

## Usage

```bash
rd convert
```

Running the command opens an interactive fzf selector. Tasks are listed newest
first. Use Tab to toggle individual tasks, Enter to confirm, or select
`*** ALL TASKS ***` at the top to convert everything at once.

By default `rd convert` reads from `./data/raw/` and writes to `./data/processed/`.
Pass `--data-dir` to use a different root:

```bash
rd convert --data-dir /mnt/storage/robot_data
```

### Episode filtering

`rd convert` only processes **successful** demonstrations — recordings marked as
failure or pending in the metadata database are always skipped.

Additionally, recordings already marked as `converted` in the database are
skipped by default, so re-running `rd convert` after adding new demonstrations
only processes the new ones.  To re-convert everything regardless:

```bash
rd convert --reconvert
```

If a recording has no database entry (e.g. recorded before the DB was set up)
its status is treated as unknown and it is included.

## Depth backends for ZED cameras

ZED stereo cameras support three depth estimation backends, selected with
`--stereo-method`:

| Backend | Flag | Description |
|---|---|---|
| ZED SDK (default) | `--stereo-method zed` | On-device NEURAL_LIGHT depth from the ZED SDK. Fast, no extra GPU setup required. |
| Fast Foundation Stereo | `--stereo-method ffs` | [Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo) — a foundation model for stereo depth. Requires a CUDA GPU; see [Installation](installation.md#fast-foundation-stereo-optional) for setup. |
| TRI Stereo | `--stereo-method tri_stereo` | TRI's learned stereo depth model, tailored for robot manipulation scenes. Available in `c32` (faster) and `c64` (higher quality) variants. Requires a CUDA GPU; see [TRI Stereo Depth](tri_stereo.md) for setup. |

```bash
# Use Fast Foundation Stereo
rd convert --stereo-method ffs

# Reduce FFS input resolution for speed (default scale 1.0)
rd convert --stereo-method ffs --ffs-scale 0.5

# Increase FFS update iterations for quality (default 8)
rd convert --stereo-method ffs --ffs-iters 16

# Use TRI Stereo (c64 variant, default)
rd convert --stereo-method tri_stereo

# Use the lighter c32 variant
rd convert --stereo-method tri_stereo --tri-stereo-variant c32
```

When `--stereo-method tri_stereo` is used, Raiden automatically selects the fastest
available backend for the chosen variant:

1. **TensorRT** — if `stereo_c64.engine` (or `stereo_c32.engine`) exists in `~/.config/raiden/weights/tri_stereo/`
2. **ONNX Runtime** — if `stereo_c64.onnx` (or `stereo_c32.onnx`) exists
3. **PyTorch** — falls back to the `.pth` checkpoint

!!! warning "Learned stereo backends are GPU-heavy"
    Both Fast Foundation Stereo and TRI Stereo depth are GPU-intensive models and
    may be slow depending on your hardware. For real-time throughput, compile
    TensorRT engines.

## Multi-camera synchronization

All ZED SVO2 cameras are extracted **simultaneously** in a single pass. On
each output frame slot the converter advances any camera that lags the most
recent camera by more than half a frame period (~16 ms at 30 fps) before
saving. This guarantees that for any output index *N* the images from all
cameras are within 16 ms of each other.

Cameras are also truncated to the **minimum frame count** across all cameras,
so every camera directory contains exactly the same number of frames.

## Output layout

```
<recording_dir>/
    split_all.json
    metadata_shared.json
    calibration_results.json        # copied from the first recording directory
    0000/
        metadata.json
        lowdim/
            0000000000.pkl          # per-frame lowdim (see below)
            0000000001.pkl
            ...
        rgb/
            scene_camera/
                0000000000.jpg      # JPEG quality ≥ 90
                0000000001.jpg
                ...
                timestamps.npy      # int64[N], nanoseconds (wall-clock)
            left_wrist_camera/
                ...
            right_wrist_camera/
                ...
        depth/
            scene_camera/
                0000000000.npz      # array key "depth", uint16, millimetres
                ...
            left_wrist_camera/
                ...
```

## File formats

**`rgb/<camera>/<frame:010d>.jpg`**
: BGR JPEG at quality ≥ 90. For cameras physically mounted upside-down
  (`right_wrist_camera`) the image is rotated 180° so the stored image is
  always right-side-up.

**`depth/<camera>/<frame:010d>.npz`**
: Compressed NumPy archive with a single key `"depth"` holding a uint16
  array (height × width) in **millimetres**. Zero means no-data.

**`rgb/<camera>/timestamps.npy`**
: `int64` NumPy array of length *N*. Each value is the wall-clock capture
  timestamp of the corresponding frame in nanoseconds (Unix epoch), on the
  same clock as `robot_data.npz` `timestamps`. For ZED cameras this comes
  directly from `sl.TIME_REFERENCE.IMAGE`; for RealSense cameras it is
  derived from the hardware timestamp corrected to wall-clock via the
  clock offset measured at recording start.

**`lowdim/<frame:010d>.pkl`**
: Per-frame lowdim file (one per frame, shared across all cameras). Keys:

| Key | Shape | Description |
|---|---|---|
| `intrinsics` | `dict[str → (3, 3) float32]` | `{camera_name: K}` — pinhole camera matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`. Principal point adjusted for upside-down cameras. |
| `extrinsics` | `dict[str → (4, 4) float32]` | `{camera_name: T_cam2world}` in the **left\_arm\_base** frame for this frame. Scene cameras: static calibrated extrinsics. Wrist cameras: computed from forward kinematics + hand-eye calibration. |
| `joints` | `(14,)` float32 | Follower **actual** joint positions at this frame: `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`. |
| `action` | `(26,)` float32 | FK EE poses computed from **commanded** joint positions (`follower_*_joint_cmd`): `[l_pos(3), l_rot9(9), l_gripper(1), r_pos(3), r_rot9(9), r_gripper(1)]`. Left arm pose is in the **left\_arm\_base** frame; right arm pose is in the **right\_arm\_base** frame. Rotation is the 3×3 matrix flattened row-major. |
| `actual_poses` | `(26,)` float32 | FK EE poses computed from **actual** joint positions (`follower_*_joint_pos_7d`): same layout as `action`. Represents where the arm physically was, not where it was commanded to go. |
| `action_joints` | `(14,)` float32 | Commanded joint positions: `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`. Same source as `action` but in joint space. |
| `language_task` | str | Task name. |
| `language_prompt` | str | Task instruction. |

## Coordinate system

The **left-arm base frame** is the global coordinate origin for all poses and
extrinsics. This convention is fixed regardless of whether you are running a
bimanual or single-arm setup - in single-arm mode the sole arm is always
treated as the left arm.

## Wrist camera extrinsics

Extrinsics for `left_wrist_camera` and `right_wrist_camera` are computed
**per frame** using forward kinematics (FK):

```
T_left_base→cam[i] = [T_left_base←right_base @] FK(q[i]) @ T_cam→ee
```

- `FK(q[i])` - MuJoCo forward kinematics for the YAM arm evaluated at the
  follower joint positions interpolated to frame *i*'s timestamp.
- `T_cam→ee` - hand-eye calibration result (camera-to-end-effector). The
  calibration was performed with the raw (upside-down) camera image, so for
  `right_wrist_camera` a 180° Z-axis rotation correction is folded in.
- `T_left_base←right_base` - applied only for `right_wrist_camera` to
  bring the result from right-arm base into the common left-arm base frame.

## Timestamp-based interpolation

`joints`, `action`, and `actual_poses` are interpolated from the ~100 Hz `robot_data.npz`
onto the camera frame timestamps using `numpy.interp`. Both sources are in
the same unit (int64 wall-clock nanoseconds), so no clock-domain conversion
is needed. A single reference camera timestamp grid is used for all cameras
in the episode (preferring wall-clock timestamps from ZED cameras).

## Loading in Python

```python
import numpy as np
import cv2
from pathlib import Path

ep = Path("data/recordings/pick_cube_20260218_143022/0000")

# RGB frames
rgb_dir = ep / "rgb" / "scene_camera"
frames   = sorted(rgb_dir.glob("*.jpg"))
ts_ns    = np.load(rgb_dir / "timestamps.npy")   # int64, nanoseconds

color = cv2.imread(str(frames[0]))                # uint8 BGR

# Depth
depth_npz = np.load(ep / "depth" / "scene_camera" / "0000000000.npz")
depth_mm  = depth_npz["depth"]                   # uint16, millimetres
depth_m   = depth_mm.astype(np.float32) / 1000.0

# Lowdim — one file per frame
import pickle
with open(ep / "lowdim" / "0000000000.pkl", "rb") as f:
    ld = pickle.load(f)

# Intrinsics and extrinsics are dicts keyed by camera name
intrinsics = ld["intrinsics"]   # dict[str → (3, 3)]
extrinsics = ld["extrinsics"]   # dict[str → (4, 4)]
# e.g. intrinsics["scene_camera"]        → (3, 3)  camera matrix K
#      extrinsics["left_wrist_camera"]   → (4, 4)  cam2world at this frame

# Joints and action for this frame
joints       = ld["joints"]        # (14,): [l_arm(6), l_grip(1), r_arm(6), r_grip(1)] — actual
action       = ld["action"]        # (26,): [l_pos(3), l_rot9(9), l_grip(1), r_pos(3), r_rot9(9), r_grip(1)] — FK(commanded)
actual_poses = ld["actual_poses"]  # (26,): same layout — FK(actual)

# Language
task   = ld["language_task"]
prompt = ld["language_prompt"]
```
