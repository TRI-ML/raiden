# Replay

The `rd replay` command replays recorded follower arm motion on the physical
hardware. Two data sources are supported:

| Source | Flag | Description |
|---|---|---|
| Raw (default) | `--source raw` | Streams joint commands directly from `robot_data.npz`. No IK required. Use this to verify a recording before converting it. |
| Processed | `--source processed` | Loads EE poses from converted `lowdim/` pkl files and solves IK at each keyframe. |

!!! note
    Replay drives the **follower arms only**. Leader arms and cameras are not
    involved.

## Usage

Running `rd replay` with no arguments opens an interactive fzf selector over
`data/raw/`:

```bash
rd replay
```

To replay from a converted (processed) episode:

```bash
rd replay --source processed
```

Or point directly at an episode directory:

```bash
rd replay --recording_dir data/raw/pick_purrito/pick_purrito_20260312_220000
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--recording_dir` | interactive | Path to an episode directory |
| `--source` | `raw` | Data source: `raw` or `processed` |
| `--arms` | `bimanual` | Which arms to replay: `bimanual` or `single` (left only) |
| `--speed` | `1.0` | Playback speed multiplier (e.g. `0.5` = half speed, `2.0` = double speed) |

## Workflow

1. The selected episode directory is loaded (auto-detected as raw or processed).
2. Both follower arms move smoothly to the **first recorded pose** (3-second ramp).
3. Poses are streamed at `control_hz` (150 Hz), scaled by `--speed`.
4. On completion (or Ctrl-C), the arms return to their home positions.

## Example

```bash
# Replay raw recording at half speed for inspection
rd replay --speed 0.5

# Replay a processed episode with IK
rd replay --source processed --speed 0.5

# Replay a specific raw episode, single arm
rd replay --recording_dir data/raw/pick_purrito/pick_purrito_20260312_220000 \
                 --arms single
```
