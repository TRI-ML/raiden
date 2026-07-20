"""Tests for ``raiden.robot.controller`` takeover control.

Hardware-free: exercises the pure ``_grip_clutch`` / ``joint_delta_takeover``
functions directly — no CAN bus, motors, or leader/follower arms required.

Gripper values throughout are normalized gripper space: **0 = closed, 1 = open**
(``Button 0 → close (0.0), Button 1 → open (1.0)``; the leader read computes
``gripper_cmd = 1 - encoder_position``, so squeezing the trigger lowers it).

Run with::

    uv run pytest tests/test_controller.py -v
"""

from __future__ import annotations

import numpy as np

from raiden.robot.controller import (
    _GRIP_FULL_CLOSE_SNAP,
    _grip_clutch,
    joint_delta_takeover,
)

# ---------------------------------------------------------------------------
# _grip_clutch — bumpless absolute gripper for the unmotorized takeover trigger
# ---------------------------------------------------------------------------


def test_clutch_can_fully_close_after_takeover():
    """Policy left the gripper OPEN (1) while the passive trigger is parked
    fully squeezed (0). A relative map could never close it (min reachable == 1).
    The clutch holds open, engages once the trigger catches up, then closes."""
    f0, l0 = 1.0, 0.0  # follower grip open, leader trigger squeezed at takeover
    engaged = False

    cmd, engaged = _grip_clutch(0.0, f0, l0, engaged)
    assert cmd == 1.0 and engaged is False  # no jump at takeover

    cmd, engaged = _grip_clutch(0.5, f0, l0, engaged)
    assert cmd == 1.0 and engaged is False  # still catching up, still held

    cmd, engaged = _grip_clutch(1.0, f0, l0, engaged)
    assert engaged is True and cmd == 1.0  # trigger crossed grip -> engaged

    cmd, engaged = _grip_clutch(0.0, f0, l0, engaged)
    assert cmd == 0.0  # THE FIX: gripper can now fully close (was impossible)


def test_clutch_immediate_when_trigger_matches_grip():
    """If the trigger already sits at the follower grip at takeover, engage at
    once and track absolutely — trivially bumpless."""
    cmd, engaged = _grip_clutch(0.3, 0.3, 0.3, False)
    assert engaged is True and abs(cmd - 0.3) < 1e-9


def test_clutch_does_not_drop_object_at_takeover():
    """Holding an object (grip=0, closed) with the trigger parked open must NOT
    move the gripper in the unintended (opening) direction on takeover."""
    cmd, engaged = _grip_clutch(0.8, 0.0, 0.8, False)
    assert cmd == 0.0 and engaged is False


def test_clutch_latches_through_reversal():
    """Once engaged, reversing the trigger back across the grip value stays in
    absolute control (must not re-hold the old grip)."""
    engaged = False
    for trigger in (0.0, 0.5, 0.8):  # rise from below, cross grip=0.5, engage
        cmd, engaged = _grip_clutch(trigger, 0.5, 0.0, engaged)
    cmd, engaged = _grip_clutch(0.2, 0.5, 0.0, engaged)  # back below grip
    assert engaged is True and abs(cmd - 0.2) < 1e-9


def test_clutch_engage_from_above():
    """Symmetric case: trigger starts ABOVE the grip at takeover."""
    f0, l0 = 0.2, 1.0
    cmd, engaged = _grip_clutch(1.0, f0, l0, False)
    assert cmd == 0.2 and engaged is False  # holds at takeover
    cmd, engaged = _grip_clutch(0.2, f0, l0, engaged)
    assert engaged is True  # crossed -> engaged
    cmd, engaged = _grip_clutch(0.0, f0, l0, engaged)
    assert cmd == 0.0  # full close reachable


def test_clutch_snaps_nearly_bottomed_trigger_to_full_close():
    """The trigger's throw saturates a few percent short of a full squeeze, so
    without a snap the follower never fully closes. A nearly-bottomed trigger
    must command a complete close once engaged."""
    residual = _GRIP_FULL_CLOSE_SNAP / 2.0  # e.g. 0.025 — inside the dead band
    cmd, engaged = _grip_clutch(residual, residual, residual, False)
    assert engaged is True and cmd == 0.0  # snapped to a full close

    # Just outside the dead band the trigger is still tracked verbatim.
    outside = _GRIP_FULL_CLOSE_SNAP * 2.0
    cmd, _ = _grip_clutch(outside, outside, outside, False)
    assert abs(cmd - outside) < 1e-9


def test_clutch_snap_does_not_disturb_a_held_grip():
    """While the clutch is still holding (not engaged), the snap must not leak
    into the commanded value — the policy's grip is passed through untouched."""
    # Trigger parked nearly bottomed (inside the snap band) and still BELOW the
    # policy's grip, so the clutch has not engaged yet.
    cmd, engaged = _grip_clutch(0.0, 0.6, 0.02, False)
    assert engaged is False and abs(cmd - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# joint_delta_takeover — arm seam-free + gripper clutch threaded end-to-end
# ---------------------------------------------------------------------------


def test_takeover_seam_free_edge_and_clutch():
    """q0 is (f0_l, f0_r, l0_l, l0_r) and the action is LEFT-then-RIGHT, so the
    right arm occupies action[7:14] and its gripper is action[13]."""
    f0_r = np.array([0, 0, 0, 0, 0, 0, 1.0])  # follower: arm home, grip open
    l0_r = np.array([0, 0, 0, 0, 0, 0, 0.0])  # leader at takeover: trigger squeezed
    q0 = (None, f0_r, None, l0_r)
    grip_engaged = {"l": False, "r": False}

    # Takeover edge: leader_now == leader_q0 -> zero arm delta, grip held.
    a = joint_delta_takeover(
        {"q7_r": l0_r.copy()}, q0, dof=7, grip_engaged=grip_engaged
    )
    assert np.allclose(a[7:13], 0.0)
    assert a[13] == 1.0 and grip_engaged["r"] is False

    # Operator moves the arm +0.1 and releases the trigger fully (catch-up -> engage).
    a = joint_delta_takeover(
        {"q7_r": np.array([0.1, 0, 0, 0, 0, 0, 1.0])},
        q0,
        dof=7,
        grip_engaged=grip_engaged,
    )
    assert abs(a[7] - 0.1) < 1e-9  # arm still tracks the leader delta
    assert grip_engaged["r"] is True

    # Squeezing the trigger now closes the gripper (the previously-stuck direction).
    a = joint_delta_takeover(
        {"q7_r": np.array([0.1, 0, 0, 0, 0, 0, 0.0])},
        q0,
        dof=7,
        grip_engaged=grip_engaged,
    )
    assert a[13] == 0.0


def test_takeover_ungrabbed_arm_holds_grip():
    """An arm with no leader read holds its follower takeover pose (incl. grip)."""
    f0_l = np.array([0, 0, 0, 0, 0, 0, 0.7])
    q0 = (f0_l, None, None, None)
    a = joint_delta_takeover({}, q0, dof=7, grip_engaged={"l": False, "r": False})
    assert np.allclose(a[0:7], f0_l)
