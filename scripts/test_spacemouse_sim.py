#!/usr/bin/env python3
"""Test SpaceMouse velocity control and J-PARSE IK in a MuJoCo simulation.

No real robot needed. Opens the YAM arm in the MuJoCo viewer and lets you
drive the end effector with a SpaceMouse to verify axis directions, velocity
scales, and IK reliability before touching the hardware.

Uses the same spacemouse_to_target_pose() as the real teleop — tuning here
carries over directly to test_spacemouse_teleop.py.

Usage
-----
    uv run scripts/test_spacemouse_sim.py
    uv run scripts/test_spacemouse_sim.py --path /dev/hidraw5
    uv run scripts/test_spacemouse_sim.py --vel-scale 0.02 --rot-scale 0.3

Controls
--------
    Push/pull/tilt puck   translate EE in world frame
    Rock/twist puck       rotate EE in world frame
    Ctrl-C                exit
"""

import argparse
import os
import sys
import threading
import time

import jax.numpy as jnp
import jaxlie
import mujoco
import mujoco.viewer
import numpy as np
import pyspacemouse
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import YAM_XML_LINEAR_4310_PATH
from scipy.spatial.transform import Rotation as ScipyRot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "i2rt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raiden.robot.controller import (
    _T_LINK6_TO_TCP,
    _T_TCP_TO_LINK6,
    _setup_pyroki,
    spacemouse_to_target_pose,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpaceMouse EE velocity control in MuJoCo sim (J-PARSE IK)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path", default="/dev/hidraw4", help="hidraw path of the SpaceMouse to use"
    )
    parser.add_argument(
        "--vel-scale",
        type=float,
        default=2.0,
        help="Max translational speed in m/s at full deflection",
    )
    parser.add_argument(
        "--rot-scale",
        type=float,
        default=3.0,
        help="Max rotational speed in rad/s at full deflection",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Control loop period in seconds (default: 50 Hz)",
    )
    parser.add_argument(
        "--invert-rotation",
        action="store_true",
        default=False,
        help="Negate all SpaceMouse rotation axes",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Pyroki + J-PARSE IK setup (includes JIT warmup)
    # -----------------------------------------------------------------------
    print("Setting up J-PARSE IK (JIT warmup, ~2–5 s)...")
    pk_robot, link6_idx, step_jit = _setup_pyroki(args.dt)
    print("IK ready.\n")

    _home_cfg = np.zeros(6, dtype=np.float64)

    def fk_tcp(q: np.ndarray) -> np.ndarray:
        """FK of the grasp_site: link_6 pose × fixed offset → 4×4."""
        poses = pk_robot.forward_kinematics(jnp.asarray(q))
        T_link6 = np.array(jaxlie.SE3(poses[link6_idx]).as_matrix())
        return T_link6 @ _T_LINK6_TO_TCP

    def ik_step(q: np.ndarray, T_target_tcp: np.ndarray):
        """Returns (new_q, info) where info contains manipulability etc."""
        T_target_link6 = T_target_tcp @ _T_TCP_TO_LINK6
        target_pos = T_target_link6[:3, 3]
        xyzw = ScipyRot.from_matrix(T_target_link6[:3, :3]).as_quat()
        target_wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        q_new, info = step_jit(
            robot=pk_robot,
            cfg=q.astype(np.float64),
            target_link_index=link6_idx,
            target_position=target_pos,
            target_wxyz=target_wxyz,
            method="jparse",
            dt=args.dt,
            home_cfg=_home_cfg,
        )
        return np.asarray(q_new), info

    # -----------------------------------------------------------------------
    # FK sanity check: compare pyroki vs i2rt at q=zeros
    # -----------------------------------------------------------------------
    kin_ref = Kinematics(YAM_XML_LINEAR_4310_PATH, "tcp_site")

    def _fk_compare(q):
        T_ref = kin_ref.fk(q)
        T_pyroki = fk_tcp(q)
        pos_err = np.linalg.norm(T_ref[:3, 3] - T_pyroki[:3, 3])
        rot_err = np.linalg.norm(T_ref[:3, :3] - T_pyroki[:3, :3])
        print(f"  q={np.round(q, 2)}  pos_err={pos_err:.6f}  rot_err={rot_err:.6f}")
        if pos_err > 1e-3 or rot_err > 1e-3:
            print(f"    i2rt  pos: {T_ref[:3, 3]}")
            print(f"    pyroki pos: {T_pyroki[:3, 3]}")

    print("=== FK comparison (i2rt tcp_site vs pyroki) ===")
    _fk_compare(np.zeros(6))
    for i in range(6):
        q = np.zeros(6)
        q[i] = 0.3
        _fk_compare(q)
    print()

    # -----------------------------------------------------------------------
    # MuJoCo model (viewer only — IK is handled by pyroki)
    # -----------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(YAM_XML_LINEAR_4310_PATH)
    data = mujoco.MjData(model)

    # Pyroki/URDF joint order is the reverse of i2rt/MuJoCo XML order.
    # q is always kept in pyroki convention; reverse when reading/writing qpos.
    q = np.zeros(6)
    data.qpos[:6] = q[::-1]
    mujoco.mj_forward(model, data)

    # -----------------------------------------------------------------------
    # SpaceMouse reader thread
    # -----------------------------------------------------------------------
    print(f"Opening SpaceMouse ({args.path})...")
    dev = pyspacemouse.open_by_path(args.path)

    sm_state = {"state": dev.read()}
    sm_lock = threading.Lock()
    running = threading.Event()
    running.set()

    def _reader() -> None:
        while running.is_set():
            s = dev.read()
            with sm_lock:
                sm_state["state"] = s

    threading.Thread(target=_reader, daemon=True).start()

    # -----------------------------------------------------------------------
    # Launch passive viewer (non-blocking)
    # -----------------------------------------------------------------------
    print("Launching MuJoCo viewer...")
    print(
        f"  vel_scale={args.vel_scale} m/s   rot_scale={args.rot_scale} rad/s   dt={args.dt}s\n"
    )

    steps = 0
    last_info: dict = {}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(
            "Push/pull/tilt puck to move the EE. Close the viewer window or Ctrl-C to stop.\n"
        )

        try:
            while viewer.is_running():
                t0 = time.monotonic()

                with sm_lock:
                    state = sm_state["state"]

                T_current = fk_tcp(q)
                T_target = spacemouse_to_target_pose(
                    state,
                    T_current,
                    args.vel_scale,
                    args.rot_scale,
                    invert_rotation=args.invert_rotation,
                )
                q_prev = q.copy()
                q, last_info = ik_step(q, T_target)

                data.qpos[:6] = q[::-1]  # pyroki→MuJoCo order
                mujoco.mj_forward(model, data)
                viewer.sync()

                steps += 1
                if steps % 50 == 0:
                    ee_pos = T_current[:3, 3]
                    tgt_pos = T_target[:3, 3]
                    dq = q - q_prev
                    manip = float(last_info.get("manipulability", 0.0))
                    print(
                        f"\n--- step {steps} ---"
                        f"\n  SpaceMouse  x={state.x:+.3f}  y={state.y:+.3f}  z={state.z:+.3f}"
                        f"\n  FK pos      x={ee_pos[0]:+.4f}  y={ee_pos[1]:+.4f}  z={ee_pos[2]:+.4f}"
                        f"\n  target pos  x={tgt_pos[0]:+.4f}  y={tgt_pos[1]:+.4f}  z={tgt_pos[2]:+.4f}"
                        f"\n  q (pyroki)  {np.round(q, 4)}"
                        f"\n  dq          {np.round(dq, 4)}"
                        f"\n  qpos (mj)   {np.round(q[::-1], 4)}"
                        f"\n  manip={manip:.4f}",
                        flush=True,
                    )

                elapsed = time.monotonic() - t0
                remaining = args.dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        except KeyboardInterrupt:
            pass

    running.clear()
    print("\nDone.")


if __name__ == "__main__":
    main()
