"""Oculus Quest Touch controller absolute-pose teleoperation."""

import threading
import time

from raiden.control.base import TeleopInterface
from raiden.robot.footpedal import (
    PEDAL_LEFT,
    PEDAL_MIDDLE,
    PEDAL_RIGHT,
    try_open_footpedal,
)


class OculusInterface(TeleopInterface):
    """EE absolute-pose control via Oculus Quest Touch controllers.

    Movement is enabled only while the grip trigger (RG) is held (deadman switch).
    Press joystick (RJ) to zero the controller's current orientation as the
    robot's forward direction before grabbing.

    A button: start episode / mark trigger
    B button: mark failure
    Right index trigger: gripper (0=open, 1=closed; inverted from trigger)
    """

    def __init__(
        self,
        right_controller: bool = True,
        spatial_coeff: float = 1.0,
        pos_action_gain: float = 1.0,
        rot_action_gain: float = 1.0,
        rmat_reorder: list | None = None,
        swap_controllers: bool = False,
    ):
        self._right_controller = right_controller
        self._spatial_coeff = spatial_coeff
        self._pos_action_gain = pos_action_gain
        self._rot_action_gain = rot_action_gain
        self._rmat_reorder = rmat_reorder if rmat_reorder is not None else [-2, -1, -3, 4]
        self._swap_controllers = swap_controllers

    @property
    def name(self) -> str:
        return "oculus"

    # ------------------------------------------------------------------
    # Session-level lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        from oculus_reader.reader import OculusReader

        self._btn_a = threading.Event()
        self._btn_b = threading.Event()
        self._pedal_trigger = threading.Event()
        self._pedal_success = threading.Event()
        self._pedal_failure = threading.Event()
        self._btn_shutdown = threading.Event()

        self._oculus_reader = OculusReader()
        time.sleep(0.5)  # allow ADB connection to settle

        self._btn_poll_thread = threading.Thread(
            target=self._button_poll_loop, name="oculus-btn-poll", daemon=True
        )
        self._btn_poll_thread.start()

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

        print("  ✓ OculusReader ready (A=trigger  B=failure  RG=grip  RJ=zero orientation)")

    def close(self) -> None:
        if hasattr(self, "_btn_shutdown"):
            self._btn_shutdown.set()
        if getattr(self, "_footpedal", None) is not None:
            self._footpedal.close()
            self._footpedal = None

    def _button_poll_loop(self) -> None:
        """20 Hz rising-edge detection for A/B buttons."""
        prev_a = False
        prev_b = False
        interval = 1.0 / 20.0
        while not self._btn_shutdown.is_set():
            t0 = time.monotonic()
            try:
                _, buttons = self._oculus_reader.get_transformations_and_buttons()
                a = bool(buttons.get("A", False))
                b = bool(buttons.get("B", False))
                if a and not prev_a:
                    self._btn_a.set()
                if b and not prev_b:
                    self._btn_b.set()
                prev_a = a
                prev_b = b
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # ------------------------------------------------------------------
    # Episode-level lifecycle
    # ------------------------------------------------------------------

    def setup(self, robot_controller) -> None:
        robot_controller.warmup_spacemouse_ik()

    def start(self, robot_controller) -> None:
        # Clear any button events that accumulated during setup/IK warmup
        # to prevent a spurious A-press from immediately exiting the loop.
        self._btn_a.clear()
        self._btn_b.clear()
        for ev_name in ("_pedal_trigger", "_pedal_success", "_pedal_failure"):
            ev = getattr(self, ev_name, None)
            if ev is not None:
                ev.clear()
        robot_controller.start_oculus_teleop(
            oculus_reader=self._oculus_reader,
            right_controller=self._right_controller,
            spatial_coeff=self._spatial_coeff,
            pos_action_gain=self._pos_action_gain,
            rot_action_gain=self._rot_action_gain,
            rmat_reorder=self._rmat_reorder,
            swap_controllers=self._swap_controllers,
        )

    def stop(self, robot_controller) -> None:
        robot_controller.stop_oculus_teleop()

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self, robot_controller) -> bool:
        if self._btn_a.is_set():
            self._btn_a.clear()
            return True
        trigger = getattr(self, "_pedal_trigger", None)
        if trigger is not None and trigger.is_set():
            trigger.clear()
            return True
        return False

    def poll_success(self, robot_controller) -> bool:
        if self._btn_a.is_set():
            self._btn_a.clear()
            return True
        ev = getattr(self, "_pedal_success", None)
        if ev is not None and ev.is_set():
            ev.clear()
            return True
        return False

    def poll_failure(self, robot_controller) -> bool:
        if self._btn_b.is_set():
            self._btn_b.clear()
            return True
        ev = getattr(self, "_pedal_failure", None)
        if ev is not None and ev.is_set():
            ev.clear()
            return True
        return False

    @property
    def waits_for_button_start(self) -> bool:
        return True

    @property
    def start_hint(self) -> str:
        return "Press A button on the Oculus controller to START (then hold grip to move)."

    @property
    def supports_verdict_button(self) -> bool:
        return True

    @property
    def banner(self) -> str:
        return (
            "\n" + "=" * 60 + "\n"
            "  OCULUS TOUCH TELEOPERATION ACTIVE\n" + "=" * 60 + "\n\n"
            "  RJ (joystick click): zero controller orientation\n"
            "  RG (grip trigger):   enable movement (deadman)\n"
            "  RT (index trigger):  gripper (squeeze=close)\n"
            "  A button:            start episode / mark success\n"
            "  B button:            mark failure\n\n"
            "  Tip: point controller in desired forward direction,\n"
            "       click RJ, then grab RG to start moving.\n\n"
            "  Press Ctrl+C to stop\n\n" + "=" * 60 + "\n"
        )
