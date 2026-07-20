# Data quality inspection

The `rd inspect` command scans raw SVO2 recordings for **frame-stutter** and
**left-eye blackouts** before you convert them, so silent capture problems are
caught while they can still be re-recorded — not discovered as corrupt training
data weeks later.

## Usage

```bash
rd inspect
```

Running the command opens an interactive fzf selector over the task directories
in `./data/raw/`. Use Tab to toggle individual tasks and Enter to confirm; each
selected task is scanned and reported.

Inspect a single task directory directly (skips the selector):

```bash
rd inspect --recording-dir data/raw/pick_cube
```

Point at a different data root, or force a full rescan ignoring the cache:

```bash
rd inspect --data-dir /mnt/storage/robot_data
rd inspect --recording-dir data/raw/pick_cube --force
```

## Why inspect before converting

Two capture faults are invisible in the recording verdict but corrupt the data
downstream:

- **Frame stutter.** [`rd replay`](replay.md) and the training pipeline treat the
  surviving frames as a uniformly-spaced 30 Hz stream. If the camera dropped
  frames during capture, the real time between two surviving frames collapses to
  a single 1/30 s step — producing bursts of unphysically fast motion and
  mislabelled actions. `rd inspect` measures the ratio of real elapsed duration
  to the ideal 30 Hz duration (`shrink`) and the largest inter-frame gap.
- **Black left eye.** [`rd convert`](conversion.md) writes the stereo **LEFT**
  view to training RGB. A dropped left sensor produces uniformly black frames
  that the timestamp scan cannot see, so the images are checked directly.

The check runs on **every** `cameras/*.svo2` view (wrist cameras can drop frames
independently of the scene camera) and is camera-name agnostic — any directory
holding a `cameras/` folder with at least one `.svo2` file is treated as an
episode.

## Verdicts

Each episode gets a single verdict — the **worst** of its per-camera views,
escalated if a view is missing or has anomalously few frames.

| Verdict | Meaning |
|---|---|
| `clean` | Steady 30 Hz, no meaningful gaps. Use as-is. |
| `borderline` | Minor stutter (shrink > 1.05 or a gap > 200 ms). Still usable. |
| `bad` | Noticeable stutter (shrink > 1.10 or a gap > 1 s). Inspect before training on it. |
| `broken` | Severe stutter (shrink > 1.20, a gap > 2 s, or fewer than 10 frames). Re-record. |
| `black` | More than 10 % of left-eye frames are black — the RGB is corrupt regardless of timing. Re-record. |
| `missing` | A view present in other episodes is absent here, or a `.svo2` failed to open. |

A view is additionally flagged `DROP` when its frame count is below 95 % of the
busiest view in the same episode (cameras grab in lock-step, so a large
discrepancy means one view lost frames).

## What it checks

For each view the scan reports:

| Metric | Description |
|---|---|
| `n` | Frames actually decoded from the SVO2. |
| `median` / `max` / `p99` (ms) | Inter-frame time gaps. `max` drives the timing verdict. |
| `g>50` / `g>100` / `g>500` | Count of gaps exceeding 50 / 100 / 500 ms. |
| `blk%` | Fraction of frames whose left eye is black. |
| `shrink` | Real duration ÷ ideal 30 Hz duration. 1.00 = perfect; higher = dropped frames. |

## Output

Results are cached and aggregated so re-runs are cheap:

```
data/raw/<task>/
    inspect_report.json         # task-level aggregate (rebuilt every run)
    0000/
        inspect.json            # per-camera cache, keyed by each .svo2 mtime
        cameras/*.svo2
    0001/
        inspect.json
        ...
```

Each camera's result is cached against its `.svo2` modification time, so a re-run
only re-scans views whose file changed (or are new). Pass `--force` to ignore the
cache entirely.

`inspect_report.json` records the per-episode verdicts, the drop/missing views,
the full per-view metrics, and the thresholds used — a machine-readable manifest
you can gate conversion or export on.

## Options

| Flag | Default | Description |
|---|---|---|
| `--recording-dir` | *interactive* | Inspect one task directory directly instead of the fzf selector. |
| `--data-dir` | `data` | Root data directory; the selector lists tasks under `<data_dir>/raw/`. |
| `--force` | `False` | Ignore the per-episode `inspect.json` cache and rescan every view. |
| `--workers` | `min(cpu, 8)` | Concurrent SVO2 scans. |

Run `rd inspect --help` for the full list.
