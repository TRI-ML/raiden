"""SpaceMouse Cartesian velocity teleoperation."""

import threading

from raiden.control.base import TeleopInterface
from raiden.robot.footpedal import (
    PEDAL_LEFT,
    PEDAL_MIDDLE,
    PEDAL_RIGHT,
    try_open_footpedal,
)


class SpaceMouseInterface(TeleopInterface):
    """EE velocity control via SpaceMouse puck(s)."""

    def __init__(
        self,
        path_r: str = "/dev/hidraw4",
        path_l: str = "/dev/hidraw5",
        vel_scale: float = 0.07,
        rot_scale: float = 0.8,
        invert_rotation: bool = False,
    ):
        self._path_r = path_r
        self._path_l = path_l
        self._vel_scale = vel_scale
        self._rot_scale = rot_scale
        self._invert_rotation = invert_rotation

    @property
    def name(self) -> str:
        return "spacemouse"

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
        robot_controller.warmup_spacemouse_ik()
        robot_controller.attach_spacemice(self._path_r, self._path_l)

    def start(self, robot_controller) -> None:
        robot_controller.start_spacemouse_teleop(
            vel_scale=self._vel_scale,
            rot_scale=self._rot_scale,
            invert_rotation=self._invert_rotation,
        )

    def stop(self, robot_controller) -> None:
        robot_controller.stop_spacemouse_teleop()

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self, robot_controller) -> bool:
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
        ev = getattr(self, "_pedal_failure", None)
        if ev is not None and ev.is_set():
            ev.clear()
            return True
        return False

    @property
    def banner(self) -> str:
        return (
            "\n" + "=" * 60 + "\n"
            "  SPACEMOUSE TELEOPERATION ACTIVE\n" + "=" * 60 + "\n\n"
            "  Push/pull/tilt puck to move EE, rock/twist to rotate\n"
            "  Press Ctrl+C to stop\n\n" + "=" * 60 + "\n"
        )
