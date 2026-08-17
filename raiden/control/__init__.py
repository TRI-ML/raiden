from raiden.control.base import TeleopInterface
from raiden.control.oculus import OculusInterface
from raiden.control.spacemouse import SpaceMouseInterface
from raiden.control.yam import YAMInterface

__all__ = ["TeleopInterface", "YAMInterface", "SpaceMouseInterface", "OculusInterface", "build_interface"]


def build_interface(
    control: str,
    spacemouse_path_r: str = "/dev/hidraw7",
    spacemouse_path_l: str = "/dev/hidraw6",
    vel_scale: float = 0.07,
    rot_scale: float = 0.8,
    invert_rotation: bool = False,
    oculus_right_controller: bool = True,
    oculus_spatial_coeff: float = 1.0,
    oculus_pos_action_gain: float = 1.0,
    oculus_rot_action_gain: float = 1.0,
    oculus_rmat_reorder: list | None = None,
    oculus_swap_controllers: bool = False,
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
    if control == "oculus":
        return OculusInterface(
            right_controller=oculus_right_controller,
            spatial_coeff=oculus_spatial_coeff,
            pos_action_gain=oculus_pos_action_gain,
            rot_action_gain=oculus_rot_action_gain,
            rmat_reorder=oculus_rmat_reorder,
            swap_controllers=oculus_swap_controllers,
        )
    return YAMInterface()
