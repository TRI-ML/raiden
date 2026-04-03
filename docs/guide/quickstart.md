# Quick Start

## Setup

Detect all connected cameras, arms, and SpaceMouse devices, and generate a
starter `~/.config/raiden/camera.json`:

```bash
rd list_devices
```

Each camera entry requires a `role` field (`"scene"`, `"left_wrist"`, or
`"right_wrist"`):

```json
{
  "scene_camera":     {"serial": 37038161,       "type": "zed",       "role": "scene"},
  "left_wrist_camera":  {"serial": "123456789012", "type": "realsense", "role": "left_wrist"},
  "right_wrist_camera": {"serial": 14932342,       "type": "zed",       "role": "right_wrist"}
}
```

## Arm configuration

Raiden supports **bimanual** (default) and **single-arm** setups via the
`--arms` flag (`bimanual` or `single`). In single-arm mode the active arm is
always called the **left arm** and the global coordinate origin is always the
**left-arm base frame**.

## Workflow

Follow these steps in order:

1. **[Calibration](calibration.md)** - hand-eye calibration for wrist cameras
   and static extrinsics for scene cameras.

2. **[Teleoperation](recording.md#teleoperation)** - practice controlling the
   arms before recording. Leader-follower mode (default) or
   [SpaceMouse](quickstart.md#spacemouse):

    ```bash
    rd teleop
    rd teleop --control spacemouse
    ```

3. **[Recording](recording.md)** - capture teleoperation demonstrations:

    ```bash
    rd record
    rd record --control spacemouse
    ```

4. **[Conversion](conversion.md)** - extract frames and build the dataset:

    ```bash
    rd convert
    ```

5. **[Visualization](visualization.md)** - inspect converted recordings:

    ```bash
    rd visualize
    ```

## SpaceMouse

Install the udev rule once so devices are accessible without `sudo`:

```bash
sudo bash scripts/install_spacemouse_udev.sh
```


