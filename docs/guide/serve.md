# Running the policy server

The `rd serve` command starts a live inference server that streams camera
observations and robot proprioception to a remote policy over WebSocket using
the [chiral](https://github.com/TRI-ML/chiral) protocol.

## Usage

```bash
rd serve
```

The server binds on `0.0.0.0:8765` by default.  Connect a chiral policy client
to `ws://<host>:8765`.

## Action space

By default the server operates in **EE pose** mode.  The policy receives
observations and returns a 20-D flat action vector:

```
[l_xyz(3), r_xyz(3), l_rot6d(6), r_rot6d(6), l_grip(1), r_grip(1)]
```

Poses are in each arm's own base frame (left arm in left-base frame, right arm
in right-base frame) — the same convention used in `lowdim.npz` during
shardification.

The server solves IK on-the-fly to convert the policy's EE pose target into
joint commands, seeding each solve from the current measured joint positions for
a warm-started, smooth solution.

To operate in **joint** mode instead (14-D, right arm first):

```bash
rd serve --action-type joint
```

| `--action-type` | Action dims | Layout |
|---|---|---|
| `ee_pose` (default) | 20 | `l_xyz(3) + r_xyz(3) + l_rot6d(6) + r_rot6d(6) + l_grip(1) + r_grip(1)` |
| `joint` | 14 | `l_joints(7) + r_joints(7)` |

## Depth backends

The server supports three depth backends for ZED cameras, selected with
`--stereo-method`:

| Backend | Flag | Description |
|---|---|---|
| ZED SDK (default) | `--stereo-method zed` | On-device NEURAL_LIGHT depth. No extra setup required. |
| Fast Foundation Stereo | `--stereo-method ffs` | GPU-based foundation model stereo depth. See [Installation](installation.md#fast-foundation-stereo-optional). |
| TRI Stereo | `--stereo-method tri_stereo` | TRI's learned stereo model. See [TRI Stereo Depth](tri_stereo.md). |

For learned stereo backends, depth inference for all cameras runs in a single
dedicated thread to avoid GPU contention.  The model is loaded once at startup.

```bash
# Use TRI Stereo c64 (default variant)
rd serve --stereo-method tri_stereo

# Use the lighter c32 variant
rd serve --stereo-method tri_stereo --tri-stereo-variant c32
```

## Safety

The server enforces a per-step joint delta limit.  If any joint command deviates
from the current measured position by more than `--max-joint-delta` radians, the
server prints an error and calls `os._exit(1)` before sending the command.

```bash
# Tighter limit (default is 0.2 rad)
rd serve --max-joint-delta 0.1
```

A footpedal soft e-stop is attached automatically if present.  Pressing the
pedal holds all arms for 5 s and then shuts down the server.

## Options

| Option | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | WebSocket bind address |
| `--port` | `8765` | WebSocket port |
| `--action-type` | `ee_pose` | Action space: `ee_pose` (20-D IK) or `joint` (14-D direct) |
| `--stereo-method` | `zed` | Depth backend: `zed`, `ffs`, or `tri_stereo` |
| `--ffs-scale` | `1.0` | FFS input resize scale |
| `--ffs-iters` | `8` | FFS update iterations |
| `--tri-stereo-variant` | `c64` | TRI Stereo variant: `c64` or `c32` |
| `--max-joint-delta` | `0.2` | Safety limit in radians per step |
| `--no-depth` | `false` | Disable ZED depth sensing (skips NEURAL_LIGHT inference for faster startup and lower GPU load) |
| `--camera-config-file` | `~/.config/raiden/camera.json` | Path to camera config |
| `--calibration-file` | `~/.config/raiden/calibration_results.json` | Path to calibration file |

Run `rd serve --help` for the full list.
