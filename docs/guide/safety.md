# Safety

## Soft E-Stop

Raiden supports an optional USB foot switch as a soft emergency stop during
teleoperation, recording, and policy evaluation.

!!! warning "Soft E-Stop Disclaimer"
    This is a **software-level ("soft") e-stop only**. YAM arms do not have brakes
    in their motors. When activated, the software holds all arm joints at their
    current positions for 5 seconds before commanding them to the home position and
    exiting the session. It does **not** cut motor power or guarantee immediate
    mechanical stoppage. Do not rely on this as a primary or sole safety mechanism.

### Behavior

1. Press the foot switch at any time during teleoperation, recording, or policy evaluation.
2. All active arms freeze at their current positions.
3. After 5 seconds, all arms return to the home position and the session exits cleanly.
4. Pressing the foot switch again during the 5-second hold resets the countdown.

In recording mode, pressing the foot switch saves the current episode as
**incomplete**. Incomplete episodes are automatically detected and overridden
on the next run.

### Setup

The foot switch is a USB HID device. The first time you use it, install a
udev rule to grant read access without requiring `sudo`:

```bash
sudo bash scripts/install_footpedal_udev.sh
```

This only needs to be run once. After that, the foot switch is auto-detected
when Raiden starts.

The foot switch is **optional** - if it is not connected, Raiden prints a
warning and continues normally without the soft e-stop feature.

### Hardware

| Component | Qty | Link |
|---|---|---|
| PCsensor USB Foot Switch | 1 | [Amazon](https://a.co/d/04osof8S) |
