#!/usr/bin/env python3
"""PCsensor FootSwitch (FS2020) test script.

Usage
-----
  # One-time setup (run once, then no sudo needed):
  sudo bash scripts/install_footpedal_udev.sh

  # List all input devices:
  python scripts/test_footpedal.py --list

  # Read events from auto-detected pedal:
  python scripts/test_footpedal.py

  # Read events from a specific device:
  python scripts/test_footpedal.py --device /dev/input/event15

Press any pedal — the script prints the event type, code, and value.
Press Ctrl-C to exit.
"""

import argparse
import sys
from pathlib import Path

from evdev import InputDevice, categorize, ecodes

PEDAL_NAME_HINT = "PCsensor FootSwitch Keyboard"


def list_devices() -> None:
    """List all input devices by reading /sys — no device-open permission needed."""
    print(f"{'Path':<22} {'Name':<42} Phys")
    print("-" * 90)
    for event_dir in sorted(
        Path("/sys/class/input").glob("event*"), key=lambda p: int(p.name[5:])
    ):
        dev_dir = event_dir / "device"
        name = (
            (dev_dir / "name").read_text().strip()
            if (dev_dir / "name").exists()
            else "?"
        )
        phys = (
            (dev_dir / "phys").read_text().strip()
            if (dev_dir / "phys").exists()
            else ""
        )
        print(f"/dev/input/{event_dir.name:<11} {name:<42} {phys}")


def find_pedal() -> InputDevice:
    """Find the pedal by reading names from /sys (no open permission needed),
    then open only the matching device."""
    for event_dir in sorted(
        Path("/sys/class/input").glob("event*"), key=lambda p: int(p.name[5:])
    ):
        name_file = event_dir / "device" / "name"
        if (
            name_file.exists()
            and PEDAL_NAME_HINT.lower() in name_file.read_text().lower()
        ):
            return InputDevice(f"/dev/input/{event_dir.name}")
    raise RuntimeError(
        f"No device matching '{PEDAL_NAME_HINT}' found.\n"
        "Run --list to see all devices, then use --device /dev/input/eventN."
    )


def read_events(device: InputDevice) -> bool:
    print(f"Opened : {device.name}")
    print(f"Path   : {device.path}")
    print(f"Phys   : {device.phys}")
    print()

    caps = device.capabilities(verbose=True)
    print("Capabilities:")
    for event_type, events in caps.items():
        print(
            f"  {event_type}: {[e[0] if isinstance(e, tuple) else e for e in events]}"
        )
    print()
    print("Waiting for pedal events (Ctrl-C to stop)...\n")

    stop_signal_received = False

    for event in device.read_loop():
        if event.type == ecodes.EV_KEY:
            key_event = categorize(event)
            state = {0: "released", 1: "pressed", 2: "held"}[event.value]
            print(f"  KEY  code={event.code:<5} ({key_event.keycode})  {state}")

            # ── stop-signal detection ──────────────────────────────────
            # Adjust once you know which pedal/code you want, e.g.:
            #   if event.value == 1 and event.code == ecodes.KEY_A:
            if event.value == 1:  # any pedal pressed
                print("\n  *** STOP SIGNAL RECEIVED ***\n")
                stop_signal_received = True
                break  # remove to keep reading after the first press

        elif event.type == ecodes.EV_ABS:
            print(f"  ABS  code={event.code:<5} value={event.value}")

        elif event.type == ecodes.EV_REL:
            print(f"  REL  code={event.code:<5} value={event.value}")

    return stop_signal_received


def main() -> None:
    parser = argparse.ArgumentParser(description="PCsensor FootSwitch test")
    parser.add_argument(
        "--list", action="store_true", help="List all input devices and exit"
    )
    parser.add_argument(
        "--device", help="Path to input device, e.g. /dev/input/event15"
    )
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    try:
        device = InputDevice(args.device) if args.device else find_pedal()
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error opening device: {e}")
        print("\nRun the one-time setup to grant access without sudo:")
        print("  sudo bash scripts/install_footpedal_udev.sh")
        sys.exit(1)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    try:
        triggered = read_events(device)
    except KeyboardInterrupt:
        triggered = False
        print("\nInterrupted.")

    print(f"\nResult: stop signal {'WAS' if triggered else 'was NOT'} received.")


if __name__ == "__main__":
    main()
