"""Leader-follower teleoperation using YAM teaching handles."""

import threading

from raiden.control.base import TeleopInterface
from raiden.robot.footpedal import (
    PEDAL_LEFT,
    PEDAL_MIDDLE,
    PEDAL_RIGHT,
    try_open_footpedal,
)


class YAMInterface(TeleopInterface):
    """Follower arms mirror leader arms in real time."""

    @property
    def name(self) -> str:
        return "leader"

    @property
    def uses_leaders(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Session-level lifecycle (footpedal)
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._pedal_trigger = threading.Event()
        self._pedal_success = threading.Event()
        self._pedal_failure = threading.Event()
        self._footpedal = try_open_footpedal()
        if self._footpedal is not None:

            def _cb(code: int) -> None:
                if code == PEDAL_LEFT:
                    rc = getattr(self, "_recording_controller", None)
                    if rc is not None:
                        rc.soft_pause()
                    else:
                        self._pedal_trigger.set()
                elif code == PEDAL_MIDDLE:
                    self._pedal_success.set()
                elif code == PEDAL_RIGHT:
                    self._pedal_failure.set()

            self._footpedal.on_press(_cb)
            self._footpedal.start()
            print(
                "  ✓ FootPedal ready: left=trigger/pause  middle=success  right=failure"
            )

    def close(self) -> None:
        if getattr(self, "_footpedal", None) is not None:
            self._footpedal.close()
            self._footpedal = None

    # ------------------------------------------------------------------
    # Episode-level lifecycle
    # ------------------------------------------------------------------

    def setup(self, robot_controller) -> None:
        pass  # leaders are already configured by setup_for_teleop_recording()

    def start(self, robot_controller) -> None:
        robot_controller.start_teleoperation()

    def stop(self, robot_controller) -> None:
        robot_controller.stop_teleoperation()

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self, robot_controller) -> bool:
        if robot_controller.check_button_press() is not None:
            return True
        trigger = getattr(self, "_pedal_trigger", None)
        if trigger is not None and trigger.is_set():
            trigger.clear()
            return True
        return False

    def poll_success(self, robot_controller) -> bool:
        ev = getattr(self, "_pedal_success", None)
        if ev is not None and ev.is_set():
            ev.clear()
            return True
        return False

    def poll_failure(self, robot_controller) -> bool:
        if robot_controller.check_failure_button():
            return True
        ev = getattr(self, "_pedal_failure", None)
        if ev is not None and ev.is_set():
            ev.clear()
            return True
        return False

    @property
    def banner(self) -> str:
        return (
            "\n" + "=" * 60 + "\n"
            "  BIMANUAL TELEOPERATION ACTIVE\n" + "=" * 60 + "\n\n"
            "  Press the button on any leader arm to return ALL arms to home\n"
            "  Press Ctrl+C for EMERGENCY STOP (stops all motion immediately)\n\n"
            + "=" * 60
            + "\n"
        )
