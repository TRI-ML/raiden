from raiden.control.base import TeleopInterface
from raiden.control.spacemouse import SpaceMouseInterface
from raiden.control.yam import YAMInterface

__all__ = ["TeleopInterface", "YAMInterface", "SpaceMouseInterface", "build_interface"]


def build_interface(
    control: str,
    spacemouse_path_r: str = "/dev/hidraw7",
    spacemouse_path_l: str = "/dev/hidraw6",
    vel_scale: float = 0.07,
    rot_scale: float = 0.8,
    invert_rotation: bool = False,
) -> TeleopInterface:
    """Construct the right TeleopInterface from CLI-style arguments."""
    if control == "spacemouse":
        return SpaceMouseInterface(
            path_r=spacemouse_path_r,
            path_l=spacemouse_path_l,
            vel_scale=vel_scale,
            rot_scale=rot_scale,
            invert_rotation=invert_rotation,
        )
    return YAMInterface()
