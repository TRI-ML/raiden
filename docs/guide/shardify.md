# Exporting to WebDataset shards

The `rd shardify` command converts converted Raiden episodes into
[WebDataset](https://github.com/webdataset/webdataset) sharded `.tar` files
ready for policy training.

## Usage

```bash
rd shardify
```

Running the command opens an interactive fzf selector to pick which converted
task to process.  To also upload the output to S3:

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
| `{uuid}.{cam}_t{idx}.jpg` | Camera image at time index `idx` relative to the anchor frame (`t-1`, `t0`, etc.) |
| `{uuid}.lowdim.npz` | Windowed arrays of shape `(T, D)` per key (see below) |
| `{uuid}.metadata.json` | Per-sample metadata (episode ID, anchor timestep, padding, …) |
| `{uuid}.language_instructions.json` | Language annotations `{"original": [...]}` |

### `lowdim.npz` keys

| Key | Shape | Description |
|---|---|---|
| `robot__action__poses__left::{robot}__xyz` | `(T, 3)` | Left EE position in left-arm-base frame |
| `robot__action__poses__left::{robot}__rot_6d` | `(T, 6)` | Left EE rotation in 6D representation |
| `robot__action__grippers__left::{robot}_hand` | `(T, 1)` | Left gripper command |
| `robot__action__poses__right::{robot}__xyz` | `(T, 3)` | Right EE position in left-arm-base frame |
| `robot__action__poses__right::{robot}__rot_6d` | `(T, 6)` | Right EE rotation in 6D representation |
| `robot__action__grippers__right::{robot}_hand` | `(T, 1)` | Right gripper command |
| `robot__actual__joint_position__left::{robot}` | `(T, 7)` | Measured left joint positions (6 arm + 1 gripper) |
| `robot__actual__joint_position__right::{robot}` | `(T, 7)` | Measured right joint positions |
| `robot__desired__joint_position__left::{robot}` | `(T, 7)` | Commanded left joint positions |
| `robot__desired__joint_position__right::{robot}` | `(T, 7)` | Commanded right joint positions |
| `intrinsics.{cam}` | `(T, 3, 3)` | Pinhole camera matrix K (tiled from anchor frame) |
| `extrinsics.{cam}` | `(T, 4, 4)` | Camera-to-world transform in left-arm-base frame |
| `past_mask` | `(T,)` bool | `True` for past timesteps |
| `future_mask` | `(T,)` bool | `True` for future timesteps |

`{robot}` defaults to `yam` (configurable with `--robot-suffix`).

The window length `T = past_lowdim_steps + 1 + future_lowdim_steps`
(default: `1 + 1 + 19 = 21`).  The anchor frame sits at index
`past_lowdim_steps` in the window.  Frames beyond the episode boundary are
clamped (copy-padded).

The rotation 6D representation uses the first two columns of the 3×3 rotation
matrix (columns 0 and 1 of R, giving a (6,) vector per timestep).

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

Each anchor frame `t` generates one sample.  The window spans frames
`[t - past_lowdim_steps, t + future_lowdim_steps]`, clamped to episode
boundaries.

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
| `--robot-suffix` | `yam` | Robot name used in lowdim key names |
| `--jpeg-quality` | `95` | JPEG quality for re-encoded images |
| `--resize-images` | `384x384` | Resize images to `HxW` before storing |
| `--filter-still-samples` | off | Skip samples where neither arm moves |
| `--still-threshold` | `0.05` | Max EE movement (m) to consider a sample still |
| `--fail-on-nan` | on | Raise an error if NaN values are found |
| `--stride` | `1` | Use every N-th frame as anchor |
| `--max-episodes` | `-1` (all) | Limit number of episodes processed |
| `--num-workers` | `1` | Number of worker processes |

Run `rd shardify --help` for the full list.
