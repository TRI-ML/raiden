# Recording demonstrations

## Overview

The `rd record` command records a full teleoperation demonstration:

- **Camera data** at 30 fps - ZED cameras write `.svo2` files; RealSense
  cameras write `.bag` files. Both formats store the raw sensor stream
  (stereo pair + depth) and are converted to a structured dataset afterwards
  with `rd convert`.
- **Robot joint data** at ~100 Hz - both leader and follower joint positions
  and velocities are saved to `robot_data.npz`. Robot timestamps are recorded
  via the reference camera's clock:
  - **ZED cameras** use `sl.TIME_REFERENCE.IMAGE`, which returns the host
    wall-clock time at the moment the frame was captured. This is on the same
    clock as `time.time_ns()` and the frame timestamps in the SVO2 file.
    No correction is needed at conversion time.
  - **RealSense cameras** attempt to use `global_time_enabled` to stamp frames
    with host wall-clock time. Because support varies across D4xx firmware
    versions, Raiden also measures the offset between the first frame's
    hardware timestamp and `time.time_ns()` at recording start and stores it
    in `metadata.json` as `realsense_clock_offsets`. The converter applies this
    offset automatically as a fallback for older firmware.

All data for one recording episode lands in a single timestamped directory
under the output folder.

## Single-arm vs bimanual

Pass `--arms single` to record with the left arm only:

```bash
rd record --arms single
```

In single-arm mode the active arm is always named **left** for consistency.
All poses and extrinsics are expressed in the **left-arm base frame** - this
convention applies in both bimanual and single-arm setups.

## Thread layout

| Thread | Purpose |
|---|---|
| `teleop-right` | Follower-right mirrors leader-right at 100 Hz (runs for whole session) |
| `teleop-left` | Follower-left mirrors leader-left at 100 Hz (runs for whole session) |
| `camera-<name>` | One per camera; calls `camera.grab()` in a tight loop (SDK limits to 30 fps); active only during an episode |
| `robot-recorder` | Reads all joint observations at ~100 Hz and timestamps each sample using `camera.get_current_timestamp_ns()` (ZED SDK clock); active only during an episode |

## Output layout

```
<output_dir>/<task>_<timestamp>/
    metadata.json        # task name, duration, fps statistics
    robot_data.npz       # joint positions and velocities for all arms
    cameras/
        scene_camera.svo2
        left_wrist_camera.svo2
        right_wrist_camera.svo2
```

### `robot_data.npz` keys

Each robot arm contributes keys of the form `<arm>_<field>`, e.g.:

| Key | Shape | Description |
|---|---|---|
| `timestamps` | `(N,)` int64 | Absolute nanosecond timestamps from the ZED SDK clock. Directly comparable with the `timestamps.npy` files written by `rd convert`. |
| `leader_r_joint_pos` | `(N, 6)` float32 | Leader-right joint positions (rad) |
| `leader_r_joint_vel` | `(N, 6)` float32 | Leader-right joint velocities (rad/s) |
| `follower_r_joint_pos` | `(N, 7)` float32 | Follower-right joint positions including gripper (rad) |
| `follower_r_joint_vel` | `(N, 7)` float32 | Follower-right joint velocities (rad/s) |
| `follower_l_joint_pos` | `(N, 7)` float32 | Follower-left joint positions including gripper (rad) |

## Usage

```bash
# Leader-follower control (default)
rd record

# SpaceMouse EE-velocity control (paths loaded from ~/.config/raiden/spacemouse.json)
rd record --control spacemouse

# Single arm with SpaceMouse
rd record --control spacemouse --arms single

# Upload to S3 after each episode
rd record --s3-bucket my-robot-data --s3-prefix demonstrations

# Store data in a custom root directory (default: ./data)
rd record --data-dir /mnt/storage/robot_data
```

Episodes are written to `<data-dir>/raw/<task_name>/`.

!!! warning "Gripper — Risk of Mechanical Damage"
    The follower gripper position is mapped directly from the leader trigger
    (0 → open, 1 → closed).  **Fully depressing the trigger drives the fingers
    to their hard stop and can break the gripper.**  Do not hold the trigger
    fully depressed.  See [Safety](safety.md#gripper-control) for details.

During the session:

- Press the **button on any leader arm** to start an episode; press it again
  to stop and save.
- After each episode, press the **leader button** again to start the next one,
  or press **`q`** to end the session.
- Press the **left foot pedal** (if connected) to trigger a soft e-stop
  **during recording**: all arms freeze for 5 seconds, then return to home and
  the session exits. The current episode is saved as incomplete. The e-stop is
  only active while an episode is recording — pressing the left pedal before
  recording starts will begin the episode instead. See
  [Soft E-Stop](hardware.md#soft-e-stop) for hardware details.
- Press **Ctrl-C** for an immediate emergency stop.

## Marking demonstrations

At the end of each episode, mark it as success or failure using the teaching
hardware. Only successful demonstrations are included when you run `rd convert`.

| Input | Action |
|---|---|
| Leader button / left foot pedal | Start or stop recording |
| Top leader button / middle foot pedal | Mark as **Success** |
| Bottom leader button / right foot pedal | Mark as **Failure** |

If no mark is given, the status stays `pending`. You can correct it later in
the [console](console.md).

After recording, convert the raw camera files with [rd convert](conversion.md).

## Uploading to S3

Pass `--s3-bucket` to automatically upload each episode to S3 immediately
after it is saved:

```bash
rd record --s3-bucket my-robot-data
```

By default episodes are uploaded under the `demonstrations/` prefix. Override
with `--s3-prefix`:

```bash
rd record --s3-bucket my-robot-data --s3-prefix pick_purrito/session_01
```

Each episode directory is uploaded recursively. The S3 key for each file is:

```
<prefix>/<episode_dir_name>/<file>
```

For example, with `--s3-prefix demonstrations` and episode directory
`pick_purrito_20260312_220000`:

```
demonstrations/pick_purrito_20260312_220000/metadata.json
demonstrations/pick_purrito_20260312_220000/robot_data.npz
demonstrations/pick_purrito_20260312_220000/cameras/scene_camera.svo2
demonstrations/pick_purrito_20260312_220000/cameras/left_wrist_camera.svo2
demonstrations/pick_purrito_20260312_220000/cameras/right_wrist_camera.svo2
```

**Credentials** — Raiden uses the standard AWS credential chain via `boto3`.
Configure credentials with any of the usual methods before recording:

```bash
# Option A — environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# Option B — AWS CLI profile
aws configure
```

## RealSense bag file size

> **Warning:** RealSense `.bag` files are large. A 10-second recording at 30 fps (640×480 BGR8 color + 640×480 depth) typically produces **~500 MB–1 GB per camera**. For longer demonstrations or setups with multiple RealSense cameras, disk space can fill up quickly.
>
> **Mitigations:**
>
> - Reduce recording duration where possible.
> - Ensure the recording disk has sufficient free space before starting a session.
