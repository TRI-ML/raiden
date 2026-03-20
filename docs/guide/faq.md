# FAQ

## Warnings

### "FootPedal not available - Continuing WITHOUT soft e-stop"

The foot switch is optional. If it is not connected, or if the udev rule has
not been installed, Raiden prints this warning and continues normally. To
enable it:

```bash
sudo bash scripts/install_footpedal_udev.sh
```

Then unplug and replug the foot switch. See [Safety](safety.md) for details.

---

### "Warning: CAN interface `<iface>` exists but is not UP"

The CAN interface was detected but is in a DOWN state. Bring all interfaces
up with the reset script:

```bash
sudo bash scripts/reset_all_can.sh
```

This sets each detected CAN interface to 1 Mbit/s and brings it UP. You may
need to run it after every reboot if your CAN interfaces are not configured to
come up automatically.

---

### "No CAN interfaces found"

No `can*` interfaces were detected at all. Possible causes:

- The CAN adapter is not connected or not powered.
- The kernel module is not loaded - try `sudo modprobe can` and
  `sudo modprobe can_raw`.
- Run `ip link show` to list all network interfaces and check whether any
  `can*` entries appear.

---

### "Warning: bimanual_transform not found in calibration; right wrist extrinsics will be in right_arm_base frame"

The `calibration_results.json` file does not contain the
`bimanual_transform` entry that maps the right-arm base into the left-arm
base frame. This is computed during calibration when both wrist cameras are
calibrated together. Re-run `rd calibrate` with both wrist cameras present
in `~/.config/raiden/camera.json`.

---

### "Warning: calibration_results.json not found"

The converter could not find the calibration file. Make sure calibration has
been run and the result was saved inside the recording directory as
`calibration_results.json`.

---

### "Warning: Only N poses recorded (minimum: M)"

Too few calibration poses were recorded before quitting. Move the robot to
more diverse positions and re-run `rd record_calibration_poses`. At least 5
poses are required; 7–10 is recommended.

---

### "Warning: cameras did not produce a first frame within N s: `<names>`"

One or more cameras failed to deliver their first frame before the policy
server timeout. Check that the cameras are powered, connected, and listed
correctly in `~/.config/raiden/camera.json`. For RealSense cameras, verify the
serial number with `rd list_devices`.

---

### "Warning: robot observation failed"

A single robot joint read failed during recording. This is usually transient
(CAN bus glitch) and the frame is skipped. If it happens frequently, check
CAN bus health with `rd list_devices` and reset interfaces with
`scripts/reset_all_can.sh`.

---

## Shell scripts

### `scripts/reset_all_can.sh`

Resets all detected CAN interfaces to 1 Mbit/s and brings them UP.
Run this after every reboot if the arms are not responding:

```bash
sudo bash scripts/reset_all_can.sh
```

---

### `scripts/install_footpedal_udev.sh`

Installs a udev rule so the PCsensor USB foot switch is accessible without
`sudo`. Run once after first connecting the foot switch:

```bash
sudo bash scripts/install_footpedal_udev.sh
```

Then unplug and replug the device. See [Safety](safety.md).

---

### `scripts/install_spacemouse_udev.sh`

Installs a udev rule for all 3Dconnexion SpaceMouse devices (vendor ID
`256f`) so they are accessible without `sudo`. Run once:

```bash
sudo bash scripts/install_spacemouse_udev.sh
```

Then unplug, replug, and log out/in (or reboot) for the group change to take
effect. See [SpaceMouse setup](quickstart.md#spacemouse).

---

### `scripts/install_ffs.py`

Clones [Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo)
into `third_party/`:

```bash
uv run python scripts/install_ffs.py
```

Then download the pretrained weights from the
[Fast Foundation Stereo repository](https://github.com/NVlabs/Fast-FoundationStereo)
and place the `*.pth` file in `~/.config/raiden/weights/`.

See [Installation - Fast Foundation Stereo](installation.md#fast-foundation-stereo-optional).

---

### `sudo apt-get install tensorrt` fails with Python version conflict

The `tensorrt` apt meta-package depends on `python3-libnvinfer` which requires
Python < 3.11. Install the C++ tools directly instead, then add the Python
bindings via pip:

```bash
sudo apt-get install --no-install-recommends libnvinfer-bin libnvinfer-dev libnvinfer-headers-dev
uv add tensorrt
```

`libnvinfer-bin` provides `trtexec` and `uv add tensorrt` installs the Python
bindings into the project environment.
