"""Abstract base class for teleoperation input methods."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raiden.robot.controller import RobotController


class TeleopInterface(ABC):
    """Shared interface for all teleoperation input methods.

    To add a new input method, subclass this, implement all abstract methods,
    and instantiate it from the CLI or calling code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in metadata (e.g. 'leader', 'spacemouse')."""

    @abstractmethod
    def setup(self, robot_controller: "RobotController") -> None:
        """Device-specific setup after robots are ready (attach hardware, warm up IK, etc.)."""

    @abstractmethod
    def start(self, robot_controller: "RobotController") -> None:
        """Start the control loop threads."""

    @abstractmethod
    def stop(self, robot_controller: "RobotController") -> None:
        """Stop the control loop threads."""

    @property
    @abstractmethod
    def banner(self) -> str:
        """Status message printed when teleoperation becomes active."""

    # ------------------------------------------------------------------
    # Optional overrides — defaults suit most future interfaces
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open session-level peripherals (footpedal, etc.).

        Called once per session before the first episode.  Default: no-op.
        """

    def close(self) -> None:
        """Close session-level peripherals.

        Called once at session end.  Default: no-op.
        """

    def set_active_recording(self, robot_controller=None) -> None:
        """Notify the interface whether a recording episode is active.

        When a controller is provided (episode started), footpedal left calls
        ``robot_controller.soft_pause()`` instead of firing a trigger event.
        Pass None when the episode ends.  Default: no-op.
        """
        self._recording_controller = robot_controller

    def poll(self, robot_controller: "RobotController") -> bool:
        """Return True on a trigger event (button press, footpedal left, etc.).

        Used as: start/stop recording in recorder, record-pose in calibration.
        Default: never triggers.
        """
        return False

    def poll_success(self, robot_controller: "RobotController") -> bool:
        """Return True if the user signalled a success outcome (footpedal middle)."""
        return False

    def poll_failure(self, robot_controller: "RobotController") -> bool:
        """Return True if the user signalled a failure outcome (footpedal right / failure button)."""
        return False

    @property
    def uses_leaders(self) -> bool:
        """True if this mode requires leader arms to be initialised."""
        return False

    @property
    def waits_for_button_start(self) -> bool:
        """True if recording should start on a leader-arm button press.

        False means keyboard / Enter key is used instead.
        """
        return self.uses_leaders

    @property
    def supports_verdict_button(self) -> bool:
        """True if leader-arm buttons can be used to mark success/failure."""
        return self.uses_leaders
