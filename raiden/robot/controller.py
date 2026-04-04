"""Modular robot controller for YAM bimanual robot system"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from raiden.robot.footpedal import FootPedal

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import pyspacemouse
import yourdfpy
from scipy.spatial.transform import Rotation

# Add the third_party/i2rt to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "i2rt")
)

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ARM_YAM_XML_PATH as _ARM_YAM_XML_PATH
from i2rt.robots.utils import GripperType

from raiden.robot._jparse import jparse_step
from raiden.robot.footpedal import try_open_footpedal

# Export for convenience
__all__ = [
    "Robot",
    "YAMLeaderRobot",
    "RobotController",
    "smooth_move_joints",
    "FOLLOWER_HOME_POS",
    "LEADER_HOME_POS",
]

# Default home positions
FOLLOWER_HOME_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # 6 joints + gripper
LEADER_HOME_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 6 joints only

# Gripper closing safety threshold in normalized [0,1] gripper space.
# Gripper is stopped from closing further when commanded position is more than
# this amount below the actual position (indicating the fingers are blocked).
# CRANK_4310 stroke ≈ 71 mm → 0.5 mm ≈ 0.007 in normalized units.
_GRIPPER_SAFETY_THRESHOLD = 6.0 / 71.0

# Gripper command speed for SpaceMouse teleop (normalized [0,1] units per second).
# Full stroke (71 mm) closes/opens in 1/_GRIPPER_SPEED seconds.
_GRIPPER_SPEED = 1.0

# ---------------------------------------------------------------------------
# Pyroki / J-PARSE IK helpers
# ---------------------------------------------------------------------------

_YAM_URDF_PATH = str(_Path(_ARM_YAM_XML_PATH).with_suffix(".urdf"))
_YAM_ASSETS_DIR = str(_Path(_ARM_YAM_XML_PATH).parent / "assets")

# Fixed transform from link_6 origin to tcp_site (from yam_4310_linear.xml).
# tcp_site pos in link_6 frame: [0, 0, 0]  (at link_6 origin)
# tcp_site rot in link_6 frame: MuJoCo quat [w=1, x=0, y=0, z=-1] (unnorm)
#   → normalized [1/√2, 0, 0, -1/√2]  →  R = [[0,1,0],[-1,0,0],[0,0,1]]
_T_LINK6_TO_TCP: np.ndarray = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Inverse: pure rotation so inverse = R^T, t stays zero
_T_TCP_TO_LINK6: np.ndarray = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# JIT-compiled jparse_step — compiled once and shared across both arm threads.
_jparse_step_jit = None


def _get_jparse_step_jit():
    """Return (and lazily create) the JIT-compiled jparse_step."""
    global _jparse_step_jit
    if _jparse_step_jit is None:
        _jparse_step_jit = jax.jit(jparse_step, static_argnames=("method",))
    return _jparse_step_jit


def _load_yam_urdf():
    """Load the YAM URDF via yourdfpy, resolving package:// asset paths."""

    def _pkg_handler(fname, dir=None):  # noqa: A002
        if isinstance(fname, str) and fname.startswith("package://assets/"):
            return fname.replace("package://assets/", _YAM_ASSETS_DIR + "/")
        return fname

    return yourdfpy.URDF.load(
        _YAM_URDF_PATH,
        filename_handler=_pkg_handler,
        load_meshes=False,
        load_collision_meshes=True,
    )


def _setup_pyroki(dt: float):
    """Load YAM robot with pyroki, JIT-compile jparse_step, and run warmup.

    Args:
        dt: Control loop period in seconds.

    Returns:
        (pk_robot, link6_idx, step_jit) where step_jit is the JIT-compiled
        IK step function ready for real-time calls.
    """
    urdf = _load_yam_urdf()
    pk_robot = pk.Robot.from_urdf(urdf)
    link6_idx = list(pk_robot.links.names).index("link_6")

    step_jit = jax.jit(jparse_step, static_argnames=("method",))

    # Warmup: force JIT compilation before the control loop starts so the
    # first real command is not delayed by tracing (takes ~2–5 s once).
    _dummy = np.zeros(6, dtype=np.float64)
    result, _ = step_jit(
        robot=pk_robot,
        cfg=_dummy,
        target_link_index=link6_idx,
        target_position=np.zeros(3, dtype=np.float64),
        target_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        method="jparse",
        dt=dt,
        home_cfg=_dummy,
    )
    jax.block_until_ready(result)

    return pk_robot, link6_idx, step_jit


class YAMLeaderRobot:
    """Wrapper for leader robot with teaching handle"""

    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint positions with gripper and button state"""
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.01)
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def get_encoder_io(self, encoder_idx: int) -> np.ndarray:
        """Get io_inputs for the encoder at *encoder_idx*."""
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        return encoder_obs[encoder_idx].io_inputs

    def get_joint_pos(self) -> np.ndarray:
        """Get joint positions (6 DoF)"""
        return self._robot.get_joint_pos()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command joint positions (6 DoF only, no gripper)"""
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """Update PD gains"""
        self._robot.update_kp_kd(kp, kd)


def smooth_move_joints(
    robot: Robot,
    target_joint_positions: np.ndarray,
    start_joint_positions: Optional[np.ndarray] = None,
    time_interval_s: float = 5.0,
    steps: int = 200,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Move the robot to target joint positions with smooth interpolation.

    Args:
        start_joint_positions: Starting configuration for the interpolation.
            Pass the previous *commanded* target (not the measured position) so
            the trajectory matches what was executed during data collection.
            Falls back to ``robot.get_joint_pos()`` when ``None`` (e.g. first step).
    """
    if start_joint_positions is not None:
        current_pos = np.asarray(start_joint_positions, dtype=np.float64)
    else:
        current_pos = robot.get_joint_pos()
    assert len(current_pos) == len(target_joint_positions)

    for i in range(steps + 1):
        if stop_event is not None and stop_event.is_set():
            return
        alpha = i / steps
        target_pos = (1 - alpha) * current_pos + alpha * target_joint_positions
        robot.command_joint_pos(target_pos)
        time.sleep(time_interval_s / steps)


def list_can_interfaces() -> list[str]:
    """Return all CAN interface names visible to the OS."""
    result = subprocess.run(
        ["ip", "-o", "link", "show"],
        capture_output=True,
        text=True,
        check=False,
    )
    interfaces = []
    for line in result.stdout.splitlines():
        # Each line: "<index>: <name>: ..."
        parts = line.split(":")
        if len(parts) >= 2:
            name = parts[1].strip().split("@")[0]  # strip e.g. "can0@..."
            if name.startswith("can"):
                interfaces.append(name)
    return interfaces


def reset_can_interface(interface: str, bitrate: int = 1000000) -> bool:
    """Bring a CAN interface down then back up at the given bitrate.

    Returns True on success, False on failure.
    """
    sudo = [] if os.geteuid() == 0 else ["sudo"]
    for args in [
        sudo + ["ip", "link", "set", interface, "down"],
        sudo
        + [
            "ip",
            "link",
            "set",
            interface,
            "up",
            "type",
            "can",
            "bitrate",
            str(bitrate),
        ],
    ]:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"  ✗ {' '.join(args)}")
            if result.stderr:
                print(f"    {result.stderr.strip()}")
            return False
    return True


def check_can_interface(interface: str) -> bool:
    """Check if a CAN interface exists and is available"""
    try:
        result = subprocess.run(
            ["ip", "link", "show", interface],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False

        if "state UP" in result.stdout or "state UNKNOWN" in result.stdout:
            return True
        else:
            print(f"Warning: CAN interface {interface} exists but is not UP")
            return False

    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False


def spacemouse_to_target_pose(
    state,
    T_current: np.ndarray,
    vel_scale: float,
    rot_scale: float,
    invert_rotation: bool = False,
) -> np.ndarray:
    """Convert a SpaceMouse state into a target EE pose.

    Translation and rotation are both applied in the world (Cartesian) frame.
    Axis mapping (tuned in test_spacemouse_sim.py):

        SpaceMouse y  →  world +x
        SpaceMouse x  →  world −y
        SpaceMouse z  →  world +z
        roll/pitch/yaw → world-frame roll/pitch/yaw

    Args:
        state:            pyspacemouse state object (has .x .y .z .roll .pitch .yaw).
        T_current:        4×4 current EE pose in the arm base frame.
        vel_scale:        Max translational speed in m/s at full deflection.
        rot_scale:        Max rotational speed in rad/s at full deflection.
        invert_rotation:  If True, negate all rotation axes.

    Returns:
        4×4 target EE pose.
    """
    # rot_sign = -1 if invert_rotation else 1

    dx = state.y * vel_scale
    dy = -state.x * vel_scale
    dz = state.z * vel_scale
    drx = state.roll * rot_scale
    dry = state.pitch * rot_scale
    drz = -state.yaw * rot_scale

    T_target = T_current.copy()
    T_target[:3, 3] += [dx, dy, dz]
    T_target[:3, :3] = (
        Rotation.from_euler("xyz", [drx, dry, drz]).as_matrix() @ T_current[:3, :3]
    )
    return T_target


class RobotController:
    """Manages YAM robot initialization, state access, and termination"""

    def __init__(
        self,
        use_right_leader: bool = True,
        use_left_leader: bool = True,
        use_right_follower: bool = True,
        use_left_follower: bool = True,
    ):
        """Initialize robot controller

        Args:
            use_right_leader: Initialize right leader arm
            use_left_leader: Initialize left leader arm
            use_right_follower: Initialize right follower arm
            use_left_follower: Initialize left follower arm
        """
        self.use_right_leader = use_right_leader
        self.use_left_leader = use_left_leader
        self.use_right_follower = use_right_follower
        self.use_left_follower = use_left_follower

        # Robot references
        self.leader_r: Optional[YAMLeaderRobot] = None
        self.leader_l: Optional[YAMLeaderRobot] = None
        self.follower_r: Optional[Robot] = None
        self.follower_l: Optional[Robot] = None

        # Track button state for edge detection (button index 0)
        self.last_button_state: Dict[str, float] = {}
        # Track button index 1 (second physical button) for verdict detection
        self._last_verdict_state: Dict[str, float] = {}
        # Track encoder_obs[1].io_inputs for failure-during-recording detection
        self._last_failure_button_state: Dict[str, float] = {}

        # Track original PD gains
        self.kp_gains: Dict[str, np.ndarray] = {}
        self.kd_gains: Dict[str, np.ndarray] = {}

        # Soft-pause state
        self._pause_until: float = 0.0
        self._pause_timer: Optional[threading.Timer] = None
        # Gate: soft_pause() is a no-op unless recording is active.
        # Call enable_estop() when recording starts, disable_estop() when it ends.
        self._estop_enabled: bool = False

        # Set by _on_pause_expired() after return_to_home() completes so that
        # the recording session loop knows to exit cleanly.
        self._session_estop_event = threading.Event()

        # Optional footpedal — attached in setup_for_teleop_recording()
        self._footpedal: Optional["FootPedal"] = None

        # Last 7-DOF position commanded to each follower (arm joints + gripper).
        # Updated by both the leader-follower and SpaceMouse teleop loops so the
        # recorder can capture the commanded pose alongside the observed pose.
        self._last_commanded_pos: Dict[str, Optional[np.ndarray]] = {
            "right": None,
            "left": None,
        }

    def check_can_interfaces(self) -> bool:
        """Check if required CAN interfaces are available"""
        required_interfaces = []

        if self.use_right_follower:
            required_interfaces.append("can_follower_r")
        if self.use_left_follower:
            required_interfaces.append("can_follower_l")
        if self.use_right_leader:
            required_interfaces.append("can_leader_r")
        if self.use_left_leader:
            required_interfaces.append("can_leader_l")

        missing_interfaces = []
        for interface in required_interfaces:
            if not check_can_interface(interface):
                missing_interfaces.append(interface)

        if missing_interfaces:
            raise RuntimeError(
                f"Missing or unavailable CAN interfaces: {', '.join(missing_interfaces)}"
            )

        print("✓ All required CAN interfaces are available")
        return True

    def initialize_robots(self, gravity_comp_mode: bool = False) -> None:
        """Initialize robots

        Args:
            gravity_comp_mode: If True, disable position control for free movement
        """
        print("\nInitializing robots...")

        # Each arm is on its own CAN bus, so all four can be initialized in
        # parallel.  Results and exceptions are collected via shared dicts.
        results: Dict[str, object] = {}
        errors: Dict[str, Exception] = {}

        def _init(name: str, channel: str, gripper_type) -> None:
            print(f"  - Initializing {name}...")
            try:
                results[name] = get_yam_robot(
                    channel=channel,
                    gripper_type=gripper_type,
                    zero_gravity_mode=False,
                )
            except Exception as exc:
                errors[name] = exc

        threads = []
        if self.use_right_follower:
            threads.append(
                threading.Thread(
                    target=_init,
                    args=("right follower", "can_follower_r", GripperType.LINEAR_4310),
                    daemon=True,
                )
            )
        if self.use_left_follower:
            threads.append(
                threading.Thread(
                    target=_init,
                    args=("left follower", "can_follower_l", GripperType.LINEAR_4310),
                    daemon=True,
                )
            )
        if self.use_right_leader:
            threads.append(
                threading.Thread(
                    target=_init,
                    args=(
                        "right leader",
                        "can_leader_r",
                        GripperType.YAM_TEACHING_HANDLE,
                    ),
                    daemon=True,
                )
            )
        if self.use_left_leader:
            threads.append(
                threading.Thread(
                    target=_init,
                    args=(
                        "left leader",
                        "can_leader_l",
                        GripperType.YAM_TEACHING_HANDLE,
                    ),
                    daemon=True,
                )
            )

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            raise RuntimeError(f"Robot initialization failed: {errors}")

        if self.use_right_follower:
            self.follower_r = results["right follower"]
        if self.use_left_follower:
            self.follower_l = results["left follower"]
        if self.use_right_leader:
            leader_r_base = results["right leader"]
            self.leader_r = YAMLeaderRobot(leader_r_base)
            self.last_button_state["leader_r"] = 0.0
            self._last_verdict_state["leader_r_top"] = 0.0
            self._last_verdict_state["leader_r_bottom"] = 0.0
            self._last_failure_button_state["leader_r"] = 0.0
            self.kp_gains["leader_r"] = self.leader_r._robot._kp.copy()
            self.kd_gains["leader_r"] = self.leader_r._robot._kd.copy()
        if self.use_left_leader:
            leader_l_base = results["left leader"]
            self.leader_l = YAMLeaderRobot(leader_l_base)
            self.last_button_state["leader_l"] = 0.0
            self._last_verdict_state["leader_l_top"] = 0.0
            self._last_verdict_state["leader_l_bottom"] = 0.0
            self._last_failure_button_state["leader_l"] = 0.0
            self.kp_gains["leader_l"] = self.leader_l._robot._kp.copy()
            self.kd_gains["leader_l"] = self.leader_l._robot._kd.copy()

        print("✓ All robots initialized")

        # Enable gravity compensation mode if requested
        if gravity_comp_mode:
            self.enable_gravity_compensation()

    def enable_gravity_compensation(self) -> None:
        """Enable gravity compensation mode (free movement with zero stiffness)"""
        print("  - Enabling gravity compensation mode...")

        if self.leader_r:
            self.leader_r.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
        if self.leader_l:
            self.leader_l.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
        if self.follower_r:
            self.follower_r.update_kp_kd(kp=np.zeros(7), kd=np.zeros(7))
        if self.follower_l:
            self.follower_l.update_kp_kd(kp=np.zeros(7), kd=np.zeros(7))

        print("✓ Gravity compensation mode enabled (robots free to move by hand)")

    def disable_gravity_compensation(self) -> None:
        """Restore position control (disable gravity compensation mode)"""
        print("  - Restoring position control...")

        if self.leader_r and "leader_r" in self.kp_gains:
            self.leader_r.update_kp_kd(
                kp=self.kp_gains["leader_r"], kd=self.kd_gains["leader_r"]
            )
        if self.leader_l and "leader_l" in self.kp_gains:
            self.leader_l.update_kp_kd(
                kp=self.kp_gains["leader_l"], kd=self.kd_gains["leader_l"]
            )
        # Note: Followers use default gains, restored automatically

        print("✓ Position control restored")

    def move_to_home_positions(self, simultaneous: bool = True) -> None:
        """Move all robots to home positions

        Args:
            simultaneous: If True, move all arms simultaneously using threads
        """
        print("\nMoving all arms to home positions...")

        if simultaneous:
            threads = []

            def move_to_home(robot, target_pos, name):
                print(f"  - Moving {name}...")
                smooth_move_joints(robot, target_pos, time_interval_s=2.0, steps=200)
                print(f"  - {name} reached home")

            if self.follower_r:
                threads.append(
                    threading.Thread(
                        target=move_to_home,
                        args=(self.follower_r, FOLLOWER_HOME_POS, "right_follower"),
                    )
                )
            if self.follower_l:
                threads.append(
                    threading.Thread(
                        target=move_to_home,
                        args=(self.follower_l, FOLLOWER_HOME_POS, "left_follower"),
                    )
                )
            if self.leader_r:
                threads.append(
                    threading.Thread(
                        target=move_to_home,
                        args=(self.leader_r._robot, LEADER_HOME_POS, "right_leader"),
                    )
                )
            if self.leader_l:
                threads.append(
                    threading.Thread(
                        target=move_to_home,
                        args=(self.leader_l._robot, LEADER_HOME_POS, "left_leader"),
                    )
                )

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            # Sequential movement
            if self.follower_r:
                print("  - Moving right follower...")
                smooth_move_joints(self.follower_r, FOLLOWER_HOME_POS)
            if self.follower_l:
                print("  - Moving left follower...")
                smooth_move_joints(self.follower_l, FOLLOWER_HOME_POS)
            if self.leader_r:
                print("  - Moving right leader...")
                smooth_move_joints(self.leader_r._robot, LEADER_HOME_POS)
            if self.leader_l:
                print("  - Moving left leader...")
                smooth_move_joints(self.leader_l._robot, LEADER_HOME_POS)

        print("✓ All arms at home positions")

    def get_all_observations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get full observations (joint_pos, joint_vel, joint_torque) from all robots.

        Returns:
            Dict mapping robot name to observation dict.
            Leaders return 6-DoF arrays; followers return 7-DoF (arm + gripper).
        """
        obs: Dict[str, Dict[str, np.ndarray]] = {}

        if self.leader_r:
            obs["leader_r"] = self.leader_r._robot.get_observations()
        if self.leader_l:
            obs["leader_l"] = self.leader_l._robot.get_observations()
        if self.follower_r:
            obs["follower_r"] = self.follower_r.get_observations()
        if self.follower_l:
            obs["follower_l"] = self.follower_l.get_observations()

        return obs

    def get_last_commanded_positions(self) -> Dict[str, Optional[np.ndarray]]:
        """Return the last 7-DOF position commanded to each follower arm.

        Returns a dict with keys ``"follower_r"`` and ``"follower_l"``, each
        mapping to a ``(7,)`` float32 array or ``None`` if no command has been
        issued yet (e.g. at the start of an episode before the first teleop step).
        """
        return {
            "follower_r": self._last_commanded_pos["right"].copy()
            if self._last_commanded_pos["right"] is not None
            else None,
            "follower_l": self._last_commanded_pos["left"].copy()
            if self._last_commanded_pos["left"] is not None
            else None,
        }

    def get_joint_positions(self) -> Dict[str, np.ndarray]:
        """Get current joint positions of all robots

        Returns:
            Dictionary mapping robot names to joint positions
        """
        positions = {}

        if self.follower_r:
            positions["follower_r"] = self.follower_r.get_joint_pos()
        if self.follower_l:
            positions["follower_l"] = self.follower_l.get_joint_pos()
        if self.leader_r:
            positions["leader_r"] = self.leader_r.get_joint_pos()
        if self.leader_l:
            positions["leader_l"] = self.leader_l.get_joint_pos()

        return positions

    def check_button_press(self) -> Optional[str]:
        """Check if any leader button was pressed (rising edge detection)

        Returns:
            Name of leader that was pressed ("leader_r" or "leader_l"), or None
        """
        if self.leader_r:
            _, button_state = self.leader_r.get_info()
            current_button = button_state[0]

            # Detect rising edge
            if current_button > 0.5 and self.last_button_state["leader_r"] < 0.5:
                self.last_button_state["leader_r"] = current_button
                return "leader_r"

            self.last_button_state["leader_r"] = current_button

        if self.leader_l:
            _, button_state = self.leader_l.get_info()
            current_button = button_state[0]

            # Detect rising edge
            if current_button > 0.5 and self.last_button_state["leader_l"] < 0.5:
                self.last_button_state["leader_l"] = current_button
                return "leader_l"

            self.last_button_state["leader_l"] = current_button

        return None

    def check_verdict_button(self) -> Optional[str]:
        """Check if a verdict button was pressed on any leader arm.

        The top button    (io_inputs[0]) signals "success".
        The bottom button (io_inputs[1]) signals "failure".
        Uses rising-edge detection per button.

        Returns:
            "success", "failure", or None.
        """
        for name, leader in [("leader_r", self.leader_r), ("leader_l", self.leader_l)]:
            if leader is None:
                continue
            try:
                _, io_inputs = leader.get_info()
                top = float(io_inputs[0]) if len(io_inputs) > 0 else 0.0
                bottom = float(io_inputs[1]) if len(io_inputs) > 1 else 0.0
                prev_top = self._last_verdict_state.get(f"{name}_top", 0.0)
                prev_bottom = self._last_verdict_state.get(f"{name}_bottom", 0.0)

                self._last_verdict_state[f"{name}_top"] = top
                self._last_verdict_state[f"{name}_bottom"] = bottom

                # Rising edge on top button → success
                if top > 0.5 and prev_top < 0.5:
                    return "success"
                # Rising edge on bottom button → failure
                if bottom > 0.5 and prev_bottom < 0.5:
                    return "failure"
            except Exception:
                pass
        return None

    def check_failure_button(self) -> bool:
        """Check if the failure button (encoder_obs[0].io_inputs[1]) was pressed.

        Returns True on the rising edge so the caller can immediately stop
        recording and mark the demonstration as failure.
        """
        for name, leader in [("leader_r", self.leader_r), ("leader_l", self.leader_l)]:
            if leader is None:
                continue
            try:
                io_inputs = leader.get_encoder_io(encoder_idx=0)
                current = float(io_inputs[1]) if len(io_inputs) > 1 else 0.0
                prev = self._last_failure_button_state.get(name, 0.0)
                self._last_failure_button_state[name] = current
                if current > 0.5 and prev < 0.5:
                    return True
            except Exception:
                pass
        return False

    def has_robots(self) -> bool:
        """Check if any robots are initialized

        Returns:
            True if at least one robot is initialized, False otherwise
        """
        return (
            self.follower_r is not None
            or self.follower_l is not None
            or self.leader_r is not None
            or self.leader_l is not None
        )

    def enable_estop(self) -> None:
        """Allow the footpedal to trigger soft_pause().  Call when recording starts."""
        self._estop_enabled = True

    def disable_estop(self) -> None:
        """Prevent footpedal from triggering soft_pause().  Call when recording ends."""
        self._estop_enabled = False
        if self._pause_timer is not None:
            self._pause_timer.cancel()
            self._pause_timer = None

    def soft_pause(self, duration: float = 5.0) -> None:
        """Freeze all arms at their current positions for *duration* seconds,
        then call return_to_home().

        Safe to call from any thread (e.g. the footpedal callback thread).
        Pressing the pedal again while paused cancels the running timer and
        resets the countdown.

        No-op when recording is not active (i.e. _estop_enabled is False).
        """
        if not self._estop_enabled:
            return
        self._pause_until = time.monotonic() + duration
        print(
            f"\n  [FootPedal] Soft pause — holding all arms for {duration:.0f}s, "
            "then returning to home"
        )

        if self._pause_timer is not None:
            self._pause_timer.cancel()
        self._pause_timer = threading.Timer(duration, self._on_pause_expired)
        self._pause_timer.daemon = True
        self._pause_timer.start()

    def _on_pause_expired(self) -> None:
        """Called by the soft-pause timer thread after the hold duration elapses.

        Only signals the main thread to exit — robot operations (return_to_home,
        close) are intentionally left to the main thread to avoid racing on the
        CAN bus with check_button_press().
        """
        self._pause_timer = None
        print("\n  [FootPedal] Hold complete — signalling session stop.")
        self._session_estop_event.set()

    @property
    def session_estop_requested(self) -> bool:
        """True after the footpedal has triggered a soft-pause and the arms
        have returned home.  Signals the recording loop to exit cleanly."""
        return self._session_estop_event.is_set()

    def _teleoperation_control_loop(self, side: str):
        """Run teleoperation control loop for one side"""
        if side == "right":
            leader = self.leader_r
            follower = self.follower_r
            leader_name = "leader_r"
        else:
            leader = self.leader_l
            follower = self.follower_l
            leader_name = "leader_l"

        if not leader or not follower:
            return

        hold_leader_pos: Optional[np.ndarray] = None
        hold_follower_pos: Optional[np.ndarray] = None
        was_paused = False
        follower_gripper_cmd = follower.get_joint_pos()[6]

        while not self._teleop_shutdown.is_set():
            try:
                now_paused = time.monotonic() < self._pause_until

                if now_paused:
                    if not was_paused:
                        # Capture positions of both arms at the start of the pause
                        hold_leader_pos = leader.get_joint_pos()
                        hold_follower_pos = follower.get_joint_pos()
                        # Restore leader PD gains so it can actively hold position
                        if leader_name in self.kp_gains:
                            leader.update_kp_kd(
                                kp=self.kp_gains[leader_name],
                                kd=self.kd_gains[leader_name],
                            )
                        was_paused = True

                    leader.command_joint_pos(hold_leader_pos)
                    follower.command_joint_pos(hold_follower_pos)
                    time.sleep(0.01)
                    continue

                # Hold just expired — if the session is stopping, exit before
                # doing any new CAN reads to avoid racing with shutdown.
                if self._session_estop_event.is_set():
                    break

                # Normal teleoperation
                leader_pos, _ = leader.get_info()
                leader.command_joint_pos(leader_pos[:6])

                # Gripper: move toward the leader's target at constant speed,
                # then clamp if blocked by an object.
                leader_gripper_target = leader_pos[6]
                step = _GRIPPER_SAFETY_THRESHOLD * 0.9
                follower_gripper_actual = follower.get_joint_pos()[6]
                if leader_gripper_target <= 0.8:
                    follower_gripper_cmd = max(0.0, follower_gripper_actual - step)
                else:
                    follower_gripper_cmd = min(1.0, follower_gripper_actual + step)
                if (
                    follower_gripper_actual - follower_gripper_cmd
                    > _GRIPPER_SAFETY_THRESHOLD
                ):
                    follower_gripper_cmd = follower_gripper_actual
                follower_cmd = np.append(leader_pos[:6], follower_gripper_cmd)

                follower.command_joint_pos(follower_cmd)
                self._last_commanded_pos[side] = follower_cmd
                time.sleep(0.001)  # 1000 Hz control loop

            except Exception as e:
                print(f"  - Error in {side} teleoperation: {e}")
                break

    def start_teleoperation(self):
        """Start teleoperation control loops (followers follow leaders)"""
        # Create shutdown event for teleoperation
        self._teleop_shutdown = threading.Event()
        self._teleop_threads = []

        if self.leader_r and self.follower_r:
            thread = threading.Thread(
                target=self._teleoperation_control_loop,
                args=("right",),
                name="teleop-right",
                daemon=True,
            )
            thread.start()
            self._teleop_threads.append(thread)

        if self.leader_l and self.follower_l:
            thread = threading.Thread(
                target=self._teleoperation_control_loop,
                args=("left",),
                name="teleop-left",
                daemon=True,
            )
            thread.start()
            self._teleop_threads.append(thread)

        if self._teleop_threads:
            time.sleep(0.1)  # Give threads time to start
            print("✓ Teleoperation active (followers will follow leaders)")

    def stop_teleoperation(self):
        """Stop all teleoperation control loops (leader-follower and SpaceMouse)."""
        if hasattr(self, "_teleop_shutdown"):
            self._teleop_shutdown.set()
            if hasattr(self, "_teleop_threads"):
                for thread in self._teleop_threads:
                    thread.join(timeout=1.0)
                self._teleop_threads.clear()
        self.stop_spacemouse_teleop()

    # -------------------------------------------------------------------------
    # SpaceMouse Cartesian velocity teleop
    # -------------------------------------------------------------------------

    def attach_spacemice(
        self,
        path_r: str = "/dev/hidraw4",
        path_l: str = "/dev/hidraw5",
    ) -> None:
        """Open two SpaceMice for Cartesian velocity teleop.

        Uses ``open_by_path`` so each handle unambiguously refers to one
        physical device regardless of how many HID interfaces each exposes.
        Run ``uv run scripts/test_spacemouse_read.py --list`` to see the
        hidraw paths for your connected devices.

        Args:
            path_r: hidraw path for the right-arm SpaceMouse (e.g. /dev/hidraw4).
            path_l: hidraw path for the left-arm SpaceMouse (e.g. /dev/hidraw5).
        """
        print("Opening SpaceMice...")
        if self.follower_r is not None:
            self._spacemouse_r = pyspacemouse.open_by_path(path_r)
            print(f"  ✓ SpaceMouse R ({path_r})")
        if self.follower_l is not None:
            self._spacemouse_l = pyspacemouse.open_by_path(path_l)
            print(f"  ✓ SpaceMouse L ({path_l})")

    def warmup_spacemouse_ik(self, dt: float = 0.01) -> None:
        """Pre-warm the pyroki/J-PARSE JIT for both SpaceMouse arms.

        Runs ``_setup_pyroki`` for each arm in parallel background threads and
        blocks until both finish.  Call this *before* showing the READY prompt
        so that no JIT compilation happens after the user starts recording.

        Stores results in ``self._prewarmed_ik`` keyed by side ("right"/"left").
        """
        self._prewarmed_ik: dict = {}
        errors: dict = {}
        lock = threading.Lock()

        def _warm(side: str) -> None:
            try:
                result = _setup_pyroki(dt)
                with lock:
                    self._prewarmed_ik[side] = result
            except Exception as exc:
                with lock:
                    errors[side] = exc

        threads = []
        if self.follower_r is not None:
            threads.append(threading.Thread(target=_warm, args=("right",), daemon=True))
        if self.follower_l is not None:
            threads.append(threading.Thread(target=_warm, args=("left",), daemon=True))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            raise RuntimeError(f"SpaceMouse IK warmup failed: {errors}")

    def start_spacemouse_teleop(
        self,
        vel_scale: float = 0.05,
        rot_scale: float = 0.5,
        dt: float = 0.01,
        invert_rotation: bool = False,
    ) -> None:
        """Start SpaceMouse Cartesian velocity teleop threads.

        SpaceMouse axis values (−1 to 1) are treated as velocities: each step
        the EE pose is advanced by ``axis_value * scale * dt`` in the current
        EE frame, then differential IK maps the new target pose to joint angles.

        Args:
            vel_scale:        Max translational speed in m/s at full deflection.
            rot_scale:        Max rotational speed in rad/s at full deflection.
            dt:               Control loop period in seconds (default 50 Hz).
            invert_rotation:  If True, negate all rotation axes.
        """
        if not hasattr(self, "_spacemouse_r") and not hasattr(self, "_spacemouse_l"):
            raise RuntimeError(
                "Call attach_spacemice() before start_spacemouse_teleop()."
            )

        self._spacemouse_shutdown = threading.Event()
        self._spacemouse_threads: list = []
        # Latest state from each device — updated by blocking reader threads.
        self._spacemouse_states: dict = {"right": None, "left": None}
        self._spacemouse_state_lock = threading.Lock()

        for side, device in [
            ("right", getattr(self, "_spacemouse_r", None)),
            ("left", getattr(self, "_spacemouse_l", None)),
        ]:
            if device is None:
                continue

            # Dedicated reader thread so read() can block without stalling control.
            def _reader(s=side, d=device) -> None:
                while not self._spacemouse_shutdown.is_set():
                    state = d.read()
                    with self._spacemouse_state_lock:
                        self._spacemouse_states[s] = state

            t = threading.Thread(
                target=_reader, name=f"spacemouse-reader-{side}", daemon=True
            )
            t.start()
            self._spacemouse_threads.append(t)

        for side, device, follower in [
            ("right", getattr(self, "_spacemouse_r", None), self.follower_r),
            ("left", getattr(self, "_spacemouse_l", None), self.follower_l),
        ]:
            if follower is None or device is None:
                continue
            t = threading.Thread(
                target=self._spacemouse_control_loop,
                args=(side, device, vel_scale, rot_scale, dt, invert_rotation),
                name=f"spacemouse-{side}",
                daemon=True,
            )
            t.start()
            self._spacemouse_threads.append(t)

        print("✓ SpaceMouse teleop active")

    def stop_spacemouse_teleop(self) -> None:
        """Stop SpaceMouse teleop threads."""
        if hasattr(self, "_spacemouse_shutdown"):
            self._spacemouse_shutdown.set()
            if hasattr(self, "_spacemouse_threads"):
                for t in self._spacemouse_threads:
                    t.join(timeout=1.0)
                self._spacemouse_threads.clear()

    def _spacemouse_control_loop(
        self,
        side: str,
        device,
        vel_scale: float,
        rot_scale: float,
        dt: float,
        invert_rotation: bool = False,
    ) -> None:
        """Velocity integration + J-PARSE IK + command loop for one arm.

        Delegates axis mapping and target pose computation to
        ``spacemouse_to_target_pose()`` (world-frame translation, EE-frame rotation).

        Uses pyroki + J-PARSE for singularity-aware velocity IK. The JIT-compiled
        step function is warmed up here before the loop starts so the first command
        is not delayed by JAX tracing.

        Gripper: Button 0 → close (0.0), Button 1 → open (1.0).
        """
        follower = self.follower_r if side == "right" else self.follower_l

        prewarmed = getattr(self, "_prewarmed_ik", {}).get(side)
        if prewarmed is not None:
            pk_robot, link6_idx, step_jit = prewarmed
        else:
            print(
                f"  [SpaceMouse {side}] Setting up J-PARSE IK (JIT warmup)...",
                flush=True,
            )
            pk_robot, link6_idx, step_jit = _setup_pyroki(dt)
            print(f"  [SpaceMouse {side}] IK ready.", flush=True)

        _home_cfg = np.zeros(6, dtype=np.float64)

        def _fk_tcp(q: np.ndarray) -> np.ndarray:
            """FK of the grasp_site: link_6 pose × fixed offset → 4×4 matrix."""
            poses = pk_robot.forward_kinematics(jnp.asarray(q))
            T_link6 = np.array(jaxlie.SE3(poses[link6_idx]).as_matrix())
            return T_link6 @ _T_LINK6_TO_TCP

        def _ik_step(q: np.ndarray, T_target_tcp: np.ndarray) -> np.ndarray:
            """One J-PARSE step: grasp_site target → new joint config."""
            # Convert grasp_site target to link_6 target frame.
            T_target_link6 = T_target_tcp @ _T_TCP_TO_LINK6
            target_pos = T_target_link6[:3, 3]
            xyzw = Rotation.from_matrix(T_target_link6[:3, :3]).as_quat()
            target_wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
            q_new, _ = step_jit(
                robot=pk_robot,
                cfg=q.astype(np.float64),
                target_link_index=link6_idx,
                target_position=target_pos,
                target_wxyz=target_wxyz,
                method="jparse",
                dt=dt,
                home_cfg=_home_cfg,
            )
            return np.asarray(q_new)

        hold_pos: Optional[np.ndarray] = None
        was_paused = False

        # Seed the virtual arm state from the real robot once at startup.
        # After this, q_arm is updated from the IK result each step (not re-read
        # from the robot), matching the sim loop and avoiding servo-lag artifacts
        # that would corrupt the rotation matrix used as the rotation frame.
        #
        # Pyroki/URDF joint order is reversed vs i2rt/MuJoCo order, so reverse
        # at the two boundaries: reading from robot and commanding the robot.
        q_full_init = follower.get_joint_pos()
        q_arm = q_full_init[:6][::-1].copy()  # MuJoCo→pyroki
        gripper = q_full_init[6]

        while not self._spacemouse_shutdown.is_set():
            try:
                loop_start = time.monotonic()

                # --- soft pause (footpedal e-stop) ---
                now_paused = time.monotonic() < self._pause_until
                if now_paused:
                    if not was_paused:
                        hold_pos = follower.get_joint_pos().copy()
                        was_paused = True
                    follower.command_joint_pos(hold_pos)
                    time.sleep(dt)
                    continue

                if self._session_estop_event.is_set():
                    break

                if was_paused:
                    # Re-sync virtual state to actual robot position after pause.
                    q_full_resync = follower.get_joint_pos()
                    q_arm = q_full_resync[:6][::-1].copy()  # MuJoCo→pyroki
                    gripper = q_full_resync[6]
                was_paused = False

                # --- read SpaceMouse ---
                # read() blocks until the device sends data; calling it in the
                # control thread would stall the loop when the puck is idle.
                # _spacemouse_states is updated by a dedicated reader thread
                # (started in start_spacemouse_teleop) so we just snapshot it.
                with self._spacemouse_state_lock:
                    state = self._spacemouse_states.get(side)
                if state is None:
                    time.sleep(dt)
                    continue

                # --- gripper (read from robot; not part of IK-tracked state) ---
                gripper_actual = follower.get_joint_pos()[6]

                T_current = _fk_tcp(q_arm)
                T_target = spacemouse_to_target_pose(
                    state, T_current, vel_scale, rot_scale, invert_rotation
                )

                # --- J-PARSE velocity IK ---
                q_arm = _ik_step(q_arm, T_target)

                # --- gripper buttons ---
                gripper = gripper_actual
                buttons = getattr(state, "buttons", [])
                if len(buttons) > 0 and buttons[0]:
                    gripper = max(
                        0.0, gripper_actual - 0.9 * _GRIPPER_SAFETY_THRESHOLD
                    )  # close
                elif len(buttons) > 1 and buttons[1]:
                    gripper = min(
                        1.0, gripper_actual + 0.9 * _GRIPPER_SAFETY_THRESHOLD
                    )  # open

                cmd = np.append(q_arm[::-1], gripper)  # pyroki→MuJoCo
                follower.command_joint_pos(cmd)
                self._last_commanded_pos[side] = cmd.copy()

                # --- pace to dt ---
                elapsed = time.monotonic() - loop_start
                remaining = dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            except Exception as e:
                print(f"  SpaceMouse {side} loop error: {e}")
                time.sleep(dt)

    def signal_ready_with_grippers(self) -> None:
        """Close then re-open both follower grippers once to signal system ready."""
        print("  - Signalling ready: closing and opening grippers...")

        closed_pos = FOLLOWER_HOME_POS.copy()
        closed_pos[6] = 0.0  # closed

        def cycle(robot: Robot) -> None:
            for _ in range(2):
                smooth_move_joints(robot, closed_pos, time_interval_s=0.1, steps=10)
                smooth_move_joints(
                    robot, FOLLOWER_HOME_POS, time_interval_s=0.1, steps=10
                )

        threads = []
        if self.follower_r:
            threads.append(
                threading.Thread(target=cycle, args=(self.follower_r,), daemon=True)
            )
        if self.follower_l:
            threads.append(
                threading.Thread(target=cycle, args=(self.follower_l,), daemon=True)
            )

        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def attach_footpedal(
        self,
        device_path: Optional[str] = None,
        direct_estop: bool = False,
        callback: Optional[callable] = None,
    ) -> None:
        """Attach a footpedal for soft e-stop.  Optional — warns and continues if absent.

        Only the LEFT pedal triggers the e-stop action.  The middle and right
        pedals are reserved for success/failure verdict marking in recorder.py
        and must not trigger a robot e-stop.

        Args:
            device_path: explicit /dev/input/eventN.  Auto-detected if None.
            direct_estop: if True, pressing the left pedal calls
                ``emergency_stop()`` immediately (skips the 5-second
                ``soft_pause`` timer).  Use this in contexts that have no
                teleoperation loop (e.g. the policy server).
            callback: if provided, called (with no arguments) when the left
                pedal is pressed, overriding both ``direct_estop`` and
                ``soft_pause`` behaviour.  Use this to inject pre-estop
                bookkeeping (e.g. setting a server-level flag) before the
                robot stop sequence runs.
        """
        from raiden.robot.footpedal import PEDAL_LEFT

        self._footpedal = try_open_footpedal(device_path)
        if self._footpedal is not None:
            if callback is not None:
                self._footpedal.on_press(
                    lambda code: callback() if code == PEDAL_LEFT else None
                )
            elif direct_estop:
                self._footpedal.on_press(
                    lambda code: self.emergency_stop() if code == PEDAL_LEFT else None
                )
            else:
                self._footpedal.on_press(
                    lambda code: (
                        self.soft_pause(duration=5.0) if code == PEDAL_LEFT else None
                    )
                )
            self._footpedal.start()

    def setup_for_teleop_recording(self):
        """Complete setup for teleoperation-based recording: init -> home -> grav comp"""
        # Check CAN interfaces
        print("Checking CAN interfaces...")
        self.check_can_interfaces()

        # Initialize robots
        self.initialize_robots(gravity_comp_mode=False)

        # Move to home positions
        self.move_to_home_positions(simultaneous=True)

        # Enable gravity compensation mode for leaders only
        print(
            "  - Disabling position control on leaders for gravity compensation mode..."
        )
        if self.leader_r:
            self.leader_r.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
        if self.leader_l:
            self.leader_l.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
        print("✓ Leaders in gravity compensation mode")

        # Gripper ready signal
        self.signal_ready_with_grippers()

        # Footpedal soft e-stop (optional)
        print("Initializing footpedal...")
        self.attach_footpedal()

    def close(self) -> None:
        """Close all robot instances, stopping their server threads and CAN connections."""
        if self._pause_timer is not None:
            self._pause_timer.cancel()
            self._pause_timer = None
        if self._footpedal is not None:
            self._footpedal.close()
            self._footpedal = None
        if self.follower_r:
            self.follower_r.close()
        if self.follower_l:
            self.follower_l.close()
        if self.leader_r:
            self.leader_r._robot.close()
        if self.leader_l:
            self.leader_l._robot.close()

    def return_to_home(self) -> None:
        """Stop teleop, move to home, re-enable leader gravity comp.

        Does NOT close robot connections — safe to call between recording
        episodes when the session should continue.
        """
        self._estop_enabled = False
        self._session_estop_event.clear()
        print("\nStopping teleoperation...")
        self.stop_teleoperation()

        print("\nMoving arms back to home positions...")
        self.disable_gravity_compensation()
        self.move_to_home_positions(simultaneous=True)
        print("✓ All arms returned to home")

        # Re-enable gravity compensation on leaders so they are ready for the
        # next episode without calling setup_for_teleop_recording() again.
        if self.leader_r:
            self.leader_r.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
        if self.leader_l:
            self.leader_l.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))

    def shutdown(self):
        """Complete shutdown: stop teleop -> restore control -> go home -> close robots"""
        time.sleep(1.0)
        self.return_to_home()
        self.close()
        time.sleep(1.0)

    def emergency_stop(self):
        """Emergency stop: hold current positions for 5s, then move all to home"""
        print("\n" + "!" * 60)
        print("  EMERGENCY STOP ACTIVATED")
        print("!" * 60)

        # Step 1: Stop teleoperation threads
        print("\n  Step 1: Stopping teleoperation threads...")
        self.stop_teleoperation()
        print("  Teleoperation stopped.")

        # Step 2: Record current positions of all robots
        print("\n  Step 2: Recording current positions of all arms...")
        hold_positions = self.get_joint_positions()
        for name, pos in hold_positions.items():
            print(f"  - Recorded {name} position: {pos[:3]}...")

        # Step 3: Hold positions for 5 seconds by continuously commanding them
        print("\n  Step 3: Holding all arms at current positions for 5 seconds...")
        start_time = time.time()
        countdown_printed = 5

        while time.time() - start_time < 5.0:
            # Continuously command all robots to hold their positions
            if self.follower_r and "follower_r" in hold_positions:
                self.follower_r.command_joint_pos(hold_positions["follower_r"])
            if self.follower_l and "follower_l" in hold_positions:
                self.follower_l.command_joint_pos(hold_positions["follower_l"])
            if self.leader_r and "leader_r" in hold_positions:
                self.leader_r._robot.command_joint_pos(hold_positions["leader_r"])
            if self.leader_l and "leader_l" in hold_positions:
                self.leader_l._robot.command_joint_pos(hold_positions["leader_l"])

            # Print countdown
            remaining = 5 - int(time.time() - start_time)
            if remaining < countdown_printed and remaining > 0:
                print(f"  - {remaining}...")
                countdown_printed = remaining

            time.sleep(0.01)  # 100 Hz command rate

        # Step 4: Move all arms to home simultaneously
        print("\n  Step 4: Moving all arms to home positions simultaneously...")
        self.disable_gravity_compensation()
        self.move_to_home_positions(simultaneous=True)

        # Detach footpedal before close() so that if emergency_stop() was
        # invoked from the footpedal callback thread, close() does not try to
        # join that same thread (which raises "cannot join current thread").
        self._footpedal = None
        self.close()

        print("\n  All arms reached home position.")
        print("  System will now shut down.")
        print("!" * 60 + "\n")
        os._exit(0)

    def cleanup(self) -> None:
        """Cleanup resources and stop robot control threads"""
        self.stop_teleoperation()
        self.close()
        os._exit(0)
