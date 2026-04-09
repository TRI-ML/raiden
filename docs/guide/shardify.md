# Exporting to WebDataset shards

The `rd shardify` command converts converted Raiden episodes into
[WebDataset](https://github.com/webdataset/webdataset) sharded `.tar` files
ready for policy training.

## Usage

```bash
rd shardify
```

Running the command opens an interactive fzf selector. Tasks are listed newest
first. Use Tab to toggle individual tasks, Enter to confirm, or select
`*** ALL TASKS ***` at the top to shardify everything at once. When multiple
tasks are selected, each is shardified in sequence. To also upload the output to S3:

```bash
rd shardify --s3-bucket my-robot-data
```

By default `rd shardify` reads from `./data/processed/`. Pass `--data-dir` to
use a different root:

```bash
rd shardify --data-dir /mnt/storage/robot_data
```

## What it produces

```
data/shards/<task_name>/
    shards/
        shard_000000.tar
        shard_000001.tar
        ...
        manifest.jsonl
        stats.json
        preprocessing_config.yaml
        processing_metadata.json
```

### Shard contents

Each `.tar` file contains a fixed number of samples (default 100).  Every
sample is identified by a UUID and consists of four file types:

| File | Description |
|---|---|
| `{uuid}.{cam}_t{idx}.png` | RGB image at raw frame offset `idx` from the anchor (`t-1`, `t0`, etc.) — lossless PNG |
| `{uuid}.{cam}_t{idx}.depth.png` | Depth map at the same offset — 16-bit greyscale PNG, values in millimetres |
| `{uuid}.lowdim.npz` | Windowed arrays of shape `(T, D)` per key (see below) |
| `{uuid}.metadata.json` | Per-sample metadata (episode ID, anchor timestep, padding, …) |
| `{uuid}.language_instructions.json` | Language annotations `{"original": [...]}` |

### `lowdim.npz` keys

All right-arm keys are present only for **bimanual** episodes; they are absent for single-arm data.

#### Action (commanded EE poses)

FK applied to **commanded** joint positions (`follower_*_joint_cmd`).  This is what the
policy should learn to predict.

| Key | Shape | Description |
|---|---|---|
| `robot__action__poses__left::yam__xyz` | `(T, 3)` | Left EE position, left-arm-base frame |
| `robot__action__poses__left::yam__rot_6d` | `(T, 6)` | Left EE rotation, 6D representation |
| `robot__action__poses__left::yam__xyz_relative` | `(T, 3)` | Left EE position relative to anchor actual pose |
| `robot__action__poses__left::yam__rot_6d_relative` | `(T, 6)` | Left EE rotation relative to anchor actual pose |
| `robot__action__grippers__left::yam_hand` | `(T, 1)` | Left gripper command |
| `robot__action__poses__right::yam__xyz` | `(T, 3)` | Right EE position, right-arm-base frame *(bimanual only)* |
| `robot__action__poses__right::yam__rot_6d` | `(T, 6)` | Right EE rotation, 6D representation *(bimanual only)* |
| `robot__action__poses__right::yam__xyz_relative` | `(T, 3)` | Right EE position relative to anchor actual pose *(bimanual only)* |
| `robot__action__poses__right::yam__rot_6d_relative` | `(T, 6)` | Right EE rotation relative to anchor actual pose *(bimanual only)* |
| `robot__action__grippers__right::yam_hand` | `(T, 1)` | Right gripper command *(bimanual only)* |

#### Proprioception (actual EE poses)

FK applied to **actual** measured joint positions (`follower_*_joint_pos_7d`).  Use these
as the proprioceptive observation fed to the policy.

| Key | Shape | Description |
|---|---|---|
| `robot__actual__poses__left::yam__xyz` | `(T, 3)` | Left actual EE position, left-arm-base frame |
| `robot__actual__poses__left::yam__rot_6d` | `(T, 6)` | Left actual EE rotation, 6D representation |
| `robot__actual__poses__left::yam__xyz_relative` | `(T, 3)` | Left actual EE position relative to anchor actual pose |
| `robot__actual__poses__left::yam__rot_6d_relative` | `(T, 6)` | Left actual EE rotation relative to anchor actual pose |
| `robot__actual__grippers__left::yam_hand` | `(T, 1)` | Left actual gripper position |
| `robot__actual__poses__right::yam__xyz` | `(T, 3)` | Right actual EE position, right-arm-base frame *(bimanual only)* |
| `robot__actual__poses__right::yam__rot_6d` | `(T, 6)` | Right actual EE rotation, 6D representation *(bimanual only)* |
| `robot__actual__poses__right::yam__xyz_relative` | `(T, 3)` | Right actual EE position relative to anchor actual pose *(bimanual only)* |
| `robot__actual__poses__right::yam__rot_6d_relative` | `(T, 6)` | Right actual EE rotation relative to anchor actual pose *(bimanual only)* |
| `robot__actual__grippers__right::yam_hand` | `(T, 1)` | Right actual gripper position *(bimanual only)* |

The `_relative` variants express each pose as an SE(3) displacement from the **anchor frame's actual pose**:

```
T_relative = T_anchor_actual_inv @ T_t
```

The anchor is the sample's current timestep (`past_lowdim_steps` into the window), so the relative representation encodes how far the end-effector moves from its current position — useful for policies that predict relative actions.

#### Joint positions

| Key | Shape | Description |
|---|---|---|
| `robot__actual__joint_position__left::yam` | `(T, 7)` | Measured left joint positions (6 arm + 1 gripper) |
| `robot__actual__joint_position__right::yam` | `(T, 7)` | Measured right joint positions *(bimanual only)* |
| `robot__desired__joint_position__left::yam` | `(T, 7)` | Commanded left joint positions |
| `robot__desired__joint_position__right::yam` | `(T, 7)` | Commanded right joint positions *(bimanual only)* |

#### Camera calibration and masks

| Key | Shape | Description |
|---|---|---|
| `intrinsics.{cam}` | `(N, 3, 3)` | Pinhole camera matrix K at each image timestep (`N = len(image_indices)`) |
| `extrinsics.{cam}` | `(N, 4, 4)` | Camera-to-world transform at each image timestep, left-arm-base frame |
| `past_mask` | `(T,)` bool | `True` for past timesteps |
| `future_mask` | `(T,)` bool | `True` for future timesteps |


The window length `T = past_lowdim_steps + 1 + future_lowdim_steps`
(default: `1 + 1 + 19 = 21`).  The anchor frame sits at index
`past_lowdim_steps` in the window.  Frames beyond the episode boundary are
clamped (copy-padded).

**Default mode (30 Hz images, 10 Hz actions):**

- Every raw frame is an anchor, so each episode produces one sample per frame
  at native 30 Hz density.
- `stride=3` — consecutive lowdim/action window steps are 3 raw frames apart,
  giving a 10 Hz action/proprioception sequence.  With `future_lowdim_steps=19`
  the action window covers `19 × 3 / 30 = 1.9` seconds.
- `image_indices` are in **raw frame** units.  The default `[-1, 0]` fetches
  two consecutive 30 Hz frames (1/30 s apart), independent of `stride`.

Set `--stride 1` for native 30 Hz action resolution.

The rotation 6D representation uses the first two rows of the 3×3 rotation
matrix (rows 0 and 1 of R, giving a (6,) vector `[R00,R01,R02,R10,R11,R12]`
per timestep).

### vla_foundry config fields

For a **single-arm** yam dataset:

```yaml
action_fields:
  - robot__action__poses__left::yam__xyz
  - robot__action__poses__left::yam__rot_6d
  - robot__action__grippers__left::yam_hand

proprioception_fields:
  - robot__actual__poses__left::yam__xyz
  - robot__actual__poses__left::yam__rot_6d
  - robot__actual__grippers__left::yam_hand
```

For a **bimanual** yam dataset, add the corresponding `right` keys to both lists.

### `manifest.jsonl`

One JSON line per shard:

```json
{"shard": "shard_000000", "num_sequences": 100}
{"shard": "shard_000001", "num_sequences": 87}
```

### `stats.json`

Per-key statistics over all samples, compatible with the `vla_foundry` training
framework.  Each entry contains:

- `mean`, `std`, `min`, `max` — global scalars per dimension `(D,)`
- `mean_per_timestep`, `std_per_timestep`, `min_per_timestep`, `max_per_timestep` — per `(T, D)`
- `percentile_1/2/5/95/98/99` — global percentiles `(D,)`
- `percentile_*_per_timestep` — per-timestep percentiles `(T, D)`
- `count` — number of samples accumulated

Global std is computed via the parallel Welford combine formula across all
timesteps, so it is not simply the average of per-timestep standard deviations.

### `preprocessing_config.yaml`

Full snapshot of the shardification parameters for reproducibility.

## Sliding window and padding

Each anchor frame `t` generates one sample.  The lowdim window spans raw frames
`[t − past_lowdim_steps × stride, t + future_lowdim_steps × stride]`,
clamped to episode boundaries.  Image frames are fetched at
`t + img_idx` for each index in `image_indices` (raw frame offset, no stride scaling).

Samples are **filtered out** if the required padding exceeds the configured
limits:

- `--max-padding-left` (default 3): maximum allowed left-side padding
- `--max-padding-right` (default 15): maximum allowed right-side padding

With the defaults, an episode needs at least **5 frames** for any sample to
pass the filter, and only the first and last few frames of each episode are
dropped.

## S3 upload

Pass `--s3-bucket` to upload the entire `shards/` directory to S3 after
writing:

```bash
rd shardify \
    --s3-bucket my-robot-data \
    --s3-prefix yam_datasets
```

Shards are uploaded to `s3://{bucket}/{prefix}/{task_name}/shards/`.
AWS credentials must be configured in the environment (standard boto3 credential
chain: env vars, `~/.aws/credentials`, instance profile, etc.).

## Options

| Option | Default | Description |
|---|---|---|
| `--data-dir` | `data` | Root data directory; reads from `<data-dir>/processed/` |
| `--output-dir` | `data/shards` | Local output directory for shards |
| `--task-name` | basename of task dir | Override the task name used in output paths |
| `--s3-bucket` | — | S3 bucket for upload |
| `--s3-prefix` | `yam_datasets` | S3 key prefix |
| `--past-lowdim-steps` | `1` | Past timesteps in the window |
| `--future-lowdim-steps` | `19` | Future timesteps in the window |
| `--max-padding-left` | `3` | Max allowed left padding |
| `--max-padding-right` | `15` | Max allowed right padding |
| `--samples-per-shard` | `100` | Samples per `.tar` file |
| `--resize-images` | `384x384` | Resize images to `HxW` before storing (Lanczos) |
| `--filter-still-samples` | off | Skip samples where neither arm moves |
| `--still-threshold` | `0.05` | Max EE movement (m) to consider a sample still |
| `--fail-on-nan` | on | Raise an error if NaN values are found |
| `--stride` | `3` | Lowdim/action window step spacing in raw frames (3=10 Hz, 1=30 Hz). Does not affect anchor density or image offsets. |
| `--max-episodes` | `-1` (all) | Limit number of episodes processed |
| `--num-workers` | `8` | Number of parallel workers for sample building |

Run `rd shardify --help` for the full list.
