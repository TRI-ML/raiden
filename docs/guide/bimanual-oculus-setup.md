# Bimanual YAM + Oculus Teleoperation Setup

Step-by-step guide for setting up dual YAM arms with Oculus Quest teleop and camera recording. Assumes hardware is assembled and connected.

**Tested on:** Ubuntu 24.04.4 LTS

---

## 1. One-time setup

### Install raiden

```bash
git clone --recurse-submodules <repo-url>
cd raiden
uv tool install -e ".[zed]"   # add extras as needed: realsense, ffs, tri-stereo
source .venv/bin/activate     # activate the raiden venv
```

See [Installation](installation.md) for full details.

### Install i2rt (standalone, for gravity compensation)

i2rt is used independently of raiden for testing individual arms. It lives in its own venv to keep dependencies separate.

```bash
git clone https://github.com/i2rt-robotics/i2rt.git
cd i2rt

# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create venv and install
uv venv --python 3.11
source .venv/bin/activate
sudo apt install build-essential python3-dev linux-headers-$(uname -r)
.venv/bin/python -m ensurepip
.venv/bin/python -m pip install -e .
```

!!! note "Two separate venvs"
    Use the **`i2rt` venv** for gravity comp (`python i2rt/robots/motor_chain_robot.py ...`).
    Use the **`raiden` venv** for everything else (`rd teleop`, `rd record`, etc.).
    Do not mix them.

### CAN interface names

Raiden expects these persistent CAN interface names:

| Interface | Arm |
|---|---|
| `can0` | Left follower |
| `can1` | Right follower |
| `can_leader_l` | Left leader |
| `can_leader_r` | Right leader |

Connect and name arms **one at a time** (all others disconnected):

```bash
ip link show        # find which can* appeared after plugging in one arm
# assign persistent name following the i2rt guide (udev rule or systemd-networkd)
```

See [Hardware Setup](hardware.md#can-bus-setup) for the detailed procedure.

### Oculus Quest — enable ADB

1. Enable Developer Mode on the Quest headset (via Meta Quest mobile app → Settings → Developer Mode).
2. Connect the Quest to the PC via USB.
3. Accept the ADB debugging prompt on the headset.
4. Verify: `adb devices` should list the headset.

The `oculus-reader` library (installed as a dependency) uses ADB automatically when `rd teleop --control oculus` starts.

---

## 2. Per-session startup

Bring all CAN interfaces up after each reboot:

```bash
rd reset_can
```

Verify arms and cameras are detected:

```bash
rd list_devices
```

---

## 3. Test gravity compensation

Test each follower arm individually before running teleop. Run from the `i2rt` directory with the `i2rt` venv active:

**Left arm (can0):**
```bash
python i2rt/robots/motor_chain_robot.py --channel can0 --gripper linear_4310
```

**Right arm (can1):**
```bash
python i2rt/robots/motor_chain_robot.py --channel can1 --gripper linear_4310
```

The arm should float freely under gravity compensation at ~250 Hz. Press `Ctrl+C` to exit; the arm will return to home. If gravity comp feels wrong (arm drifts or fights you), check the zero-position calibration of the motors.

---

## 4. Oculus teleoperation

Switch to the `raiden` venv and run from the `raiden` directory.

**Both arms:**
```bash
rd teleop --control oculus
```

**Single arm (left, can0):**
```bash
rd teleop --control oculus --arms single
```

### Controller mapping (default — standing in front of the robot)

| Control | Action |
|---|---|
| **RJ** (joystick click) | Zero controller orientation — point forward, then click |
| **RG** (grip trigger, hold) | Deadman switch — arm only moves while held |
| **RT** (index trigger) | Gripper (squeeze = close) |
| **A button** | Mark success / end episode |
| **B button** | Mark failure |

Right Touch controller drives the **left arm**; left Touch controller drives the **right arm**.

### Standing behind the robot

If you are standing behind the arms (same side as the robot base), swap the controller-to-arm assignment so axes align with your perspective:

```bash
rd teleop --control oculus --oculus-swap-controllers
```

### Axis tuning

If translation feels wrong, re-zero the orientation with **RJ** before grabbing the grip trigger. For persistent axis issues, adjust the frame mapping:

```bash
rd teleop --control oculus --oculus-rmat-reorder '[-2,-1,-3,4]'
```

Use `scripts/oculus_calib.py` to visualize the controller axes without the robot.

---

## 5. Camera setup (required for recording)

### Detect cameras and create config

```bash
rd list_devices
```

Edit `~/.config/raiden/camera.json` to assign a role to each camera:

```json
{
  "scene_camera":      {"serial": 37038161, "type": "zed", "role": "scene"},
  "left_wrist_camera": {"serial": 16522755, "type": "zed", "role": "left_wrist"},
  "right_wrist_camera":{"serial": 14932342, "type": "zed", "role": "right_wrist"}
}
```

Roles: `"scene"` (fixed overhead, multiple allowed), `"left_wrist"`, `"right_wrist"`.

!!! note "Right wrist ZED Mini"
    The ZED Mini on the right wrist must be mounted **upside down**.

### Calibrate cameras (one-time, repeat if cameras are moved)

Print a 9×9 ChArUco board (30 mm squares, 23 mm ArUco markers). Move the robot through 7–10 diverse poses with the board visible to all cameras:

```bash
rd record_calibration_poses   # move arms, press leader button to record each pose
rd calibrate                  # computes extrinsics → ~/.config/raiden/calibration_results.json
```

See [Calibration](calibration.md) for details.

---

## 6. Recording demonstrations

With cameras calibrated:

```bash
rd record --control oculus
```

**Key differences from `rd teleop`:**

- Cameras must be configured in `camera.json` with roles.
- Calibration results must exist in `~/.config/raiden/calibration_results.json`.
- Data is saved to `data/raw/<task>/<timestamp>/` (change root with `--data-dir`).

Mark each episode during recording:

| Action | Default control |
|---|---|
| Start / stop recording | Left foot pedal (or A button) |
| Mark success | Middle pedal (or A button) |
| Mark failure | Right pedal (or B button) |

After collecting demonstrations, convert to a structured dataset:

```bash
rd convert    # extracts frames and depth → data/processed/
rd visualize  # inspect a converted episode
```
