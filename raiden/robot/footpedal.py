"""PCsensor FootSwitch integration.

Runs a background thread that reads key-press events from the footpedal and
invokes registered callbacks with the key code.

Usage::

    pedal = FootPedal()   # auto-detects the device via /sys
    pedal.on_press(lambda code: print(f"pedal pressed: {code}"))
    pedal.open()
    pedal.start()
    ...
    pedal.close()

The device path is resolved from /sys/class/input without needing to open any
/dev/input file, so auto-detection works before the udev rule is installed.
Opening the device itself does require the udev rule (or sudo).  Run once::

    sudo bash scripts/install_footpedal_udev.sh
"""

import threading
from pathlib import Path
from typing import Callable, List, Optional

from evdev import InputDevice, ecodes

DEVICE_NAME = "PCsensor FootSwitch Keyboard"

# Default key codes emitted by the 3-pedal PCsensor FootSwitch.
# Adjust if your device is configured differently (use `evtest` to confirm).
PEDAL_LEFT = 30  # KEY_A — start / stop recording
PEDAL_MIDDLE = 48  # KEY_B — mark demonstration as success
PEDAL_RIGHT = 46  # KEY_C — mark demonstration as failure


class FootPedal:
    """Reads PCsensor FootSwitch button presses in a background thread."""

    def __init__(self, device_path: Optional[str] = None):
        """Args:
        device_path: explicit /dev/input/eventN path.  Auto-detected if None.
        """
        self._device_path = device_path or self._find_device_path()
        self._device: Optional[InputDevice] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: List[Callable[[int], None]] = []

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_device_path() -> str:
        """Scan /sys/class/input for a matching device name (no open needed)."""
        for event_dir in sorted(
            Path("/sys/class/input").glob("event*"),
            key=lambda p: int(p.name[5:]),
        ):
            name_file = event_dir / "device" / "name"
            if (
                name_file.exists()
                and DEVICE_NAME.lower() in name_file.read_text().lower()
            ):
                return f"/dev/input/{event_dir.name}"
        raise RuntimeError(
            f"FootPedal ({DEVICE_NAME!r}) not found. "
            "Make sure it is plugged in. "
            "If this is the first run, install the udev rule first:\n"
            "  sudo bash scripts/install_footpedal_udev.sh"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._device = InputDevice(self._device_path)
        print(f"  ✓ FootPedal opened: {self._device.name} ({self._device_path})")

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._device is not None:
            self._device.close()
            self._device = None

    def start(self) -> None:
        """Start background event-reading thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="footpedal",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_press(self, callback: Callable[[int], None]) -> None:
        """Register *callback(key_code)* — called on every pedal press.

        The callback is invoked from the footpedal thread; keep it short or
        hand off to another thread if needed.
        """
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        try:
            for event in self._device.read_loop():
                if self._stop_event.is_set():
                    break
                if event.type == ecodes.EV_KEY and event.value == 1:  # key down
                    for cb in self._callbacks:
                        try:
                            cb(event.code)
                        except Exception as e:
                            print(f"  FootPedal callback error: {e}")
        except OSError as e:
            if not self._stop_event.is_set():
                print(f"  FootPedal read error: {e}")


# ---------------------------------------------------------------------------
# Optional helper — try to create a FootPedal, return None with a warning on
# failure (device not connected, permission error, etc.)
# ---------------------------------------------------------------------------

_YELLOW = "\033[1;33m"
_RESET = "\033[0m"


def _warn(msg: str) -> None:
    print(f"{_YELLOW}{'!' * 60}{_RESET}")
    for line in msg.splitlines():
        print(f"{_YELLOW}  {line}{_RESET}")
    print(f"{_YELLOW}{'!' * 60}{_RESET}\n")


def try_open_footpedal(device_path: Optional[str] = None) -> Optional[FootPedal]:
    """Create, open, and return a FootPedal, or return None with a warning.

    Intended for callers that treat the footpedal as optional — recording and
    teleoperation continue normally without it.
    """
    try:
        pedal = FootPedal(device_path)
        pedal.open()
        return pedal
    except RuntimeError as e:
        _warn(f"FootPedal not available — {e}\nContinuing WITHOUT soft e-stop.")
        return None
    except PermissionError as e:
        _warn(
            f"FootPedal permission denied ({e})\n"
            "Run once to fix:  sudo bash scripts/install_footpedal_udev.sh\n"
            "Continuing WITHOUT soft e-stop."
        )
        return None
