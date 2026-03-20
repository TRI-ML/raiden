#!/usr/bin/env python3
"""Read and display raw SpaceMouse axis values without touching the robot.

Run this first to verify both SpaceMice are detected, confirm which path
corresponds to which physical mouse, and check that the axis directions
and button mappings feel correct before running the full teleop script.

Usage
-----
    # List detected SpaceMice then exit:
    uv run scripts/test_spacemouse_read.py --list

    # Read from the two detected hidraw paths (defaults):
    uv run scripts/test_spacemouse_read.py

    # Override paths if yours differ:
    uv run scripts/test_spacemouse_read.py --path-r /dev/hidraw5 --path-l /dev/hidraw4
"""

import argparse
import sys
import threading
import time


def list_devices() -> None:
    """Print all detected SpaceMouse HID paths and exit."""
    try:
        import easyhid
    except ImportError:
        print("easyhid is not installed (should be pulled in by pyspacemouse).")
        return

    h = easyhid.Enumeration()
    devs = [d for d in h.find() if "3Dconnexion" in d.description()]
    if not devs:
        print("No SpaceMouse devices found.")
        return

    seen: set = set()
    for d in devs:
        desc = d.description()
        path = desc.split("|")[0].strip()
        if path not in seen:
            seen.add(path)
            print(desc)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_BLOCK_LINES = 6  # lines printed per device block (including blank separator)


def _bar(value: float, width: int = 20) -> str:
    """Render a signed bar for a value in [-1, 1]."""
    half = width // 2
    filled = min(int(abs(value) * half), half)
    if value >= 0:
        left = " " * half
        right = "█" * filled + " " * (half - filled)
    else:
        left = " " * (half - filled) + "█" * filled
        right = " " * half
    return f"[{left}|{right}]"


def _render_device(label: str, path: str, state) -> str:
    x = getattr(state, "x", 0.0)
    y = getattr(state, "y", 0.0)
    z = getattr(state, "z", 0.0)
    roll = getattr(state, "roll", 0.0)
    pitch = getattr(state, "pitch", 0.0)
    yaw = getattr(state, "yaw", 0.0)
    buttons = getattr(state, "buttons", [])
    btn_str = (
        "  ".join(f"{'▣' if b else '□'} Btn{i + 1}" for i, b in enumerate(buttons))
        if buttons
        else "—"
    )

    lines = [
        f"{_BOLD}SpaceMouse {label}{_RESET}  {_DIM}{path}{_RESET}",
        f"  translate  x {_bar(x)}  {x:+.3f}",
        f"             y {_bar(y)}  {y:+.3f}",
        f"             z {_bar(z)}  {z:+.3f}",
        f"  rotate   rol {_bar(roll)}  {roll:+.3f}   "
        f"pit {_bar(pitch)}  {pitch:+.3f}   "
        f"yaw {_bar(yaw)}  {yaw:+.3f}",
        f"  buttons    {btn_str}",
    ]
    return "\n".join(lines)


def _redraw(blocks: list[str]) -> None:
    """Overwrite the previously printed blocks in-place."""
    total_lines = _BLOCK_LINES * len(blocks)
    # Move cursor up to the start of the display area
    sys.stdout.write(f"\033[{total_lines}A")
    for block in blocks:
        for line in block.split("\n"):
            # Print line, clear to end of line, then newline
            sys.stdout.write(f"{line}\033[K\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read SpaceMouse axes without connecting to the robot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List detected SpaceMouse HID paths and exit",
    )
    parser.add_argument(
        "--path-r",
        default="/dev/hidraw4",
        help="hidraw path for the right-arm SpaceMouse",
    )
    parser.add_argument(
        "--path-l",
        default="/dev/hidraw5",
        help="hidraw path for the left-arm SpaceMouse",
    )
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    try:
        import pyspacemouse
    except ImportError:
        print("pyspacemouse is not installed. Run: uv add pyspacemouse")
        return

    print(f"Opening SpaceMouse R ({args.path_r})...")
    dev_r = pyspacemouse.open_by_path(args.path_r)
    print(f"Opening SpaceMouse L ({args.path_l})...")
    dev_l = pyspacemouse.open_by_path(args.path_l)

    print("\nMove each puck and press its buttons. Ctrl-C to stop.\n")

    # Shared state — updated by reader threads, read by display loop
    states: dict = {"R": dev_r.read(), "L": dev_l.read()}
    lock = threading.Lock()
    running = threading.Event()
    running.set()

    def _reader(label: str, dev) -> None:
        """Blocking read loop — runs in its own thread."""
        while running.is_set():
            s = dev.read()
            with lock:
                states[label] = s

    threading.Thread(target=_reader, args=("R", dev_r), daemon=True).start()
    threading.Thread(target=_reader, args=("L", dev_l), daemon=True).start()

    devices = [("R", args.path_r), ("L", args.path_l)]

    # Initial render — print all lines once so _redraw has something to overwrite
    for label, path in devices:
        with lock:
            s = states[label]
        print(_render_device(label, path, s))

    try:
        while True:
            time.sleep(0.05)  # 20 Hz display
            blocks = []
            with lock:
                for label, path in devices:
                    blocks.append(_render_device(label, path, states[label]))
            _redraw(blocks)

    except KeyboardInterrupt:
        running.clear()
        print("\nDone.")


if __name__ == "__main__":
    main()
