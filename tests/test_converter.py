"""Tests for ``raiden.converter`` recovery-tag resampling.

``control_source`` is a discrete 0/1 human-takeover flag recorded on the robot
clock and replayed onto the camera-frame grid. Interpolating it would smear it
into a meaningless fraction, so it is resampled by NEAREST timestamp — these
tests pin that behaviour down.

Run with::

    uv run pytest tests/test_converter.py -v
"""

from __future__ import annotations

import numpy as np

from raiden.converter import _resample_flag_nearest

MS = 1_000_000  # nanoseconds per millisecond


def test_identity_when_grids_match():
    robot_ts = np.arange(6, dtype=np.int64) * 10 * MS
    flag = np.array([0, 0, 1, 1, 0, 0], dtype=np.int8)
    out = _resample_flag_nearest(flag, robot_ts, robot_ts)
    assert np.array_equal(out, flag)


def test_output_is_always_a_clean_flag_never_interpolated():
    """The whole point: sampling across a 0->1 edge must never yield a fraction.

    ``np.interp`` on the same inputs produces intermediate values; nearest does
    not.
    """
    robot_ts = np.array([0, 100 * MS], dtype=np.int64)
    flag = np.array([0, 1], dtype=np.int8)
    ref_ts = np.linspace(0, 100 * MS, 11).astype(np.int64)

    out = _resample_flag_nearest(flag, robot_ts, ref_ts)
    assert set(np.unique(out)).issubset({0, 1})

    smeared = np.interp(ref_ts, robot_ts, flag)
    assert not set(np.unique(smeared)).issubset({0, 1})  # documents the contrast


def test_picks_the_nearer_robot_sample():
    robot_ts = np.array([0, 100 * MS], dtype=np.int64)
    flag = np.array([0, 1], dtype=np.int8)
    # 40 ms is nearer the first sample, 60 ms nearer the second.
    out = _resample_flag_nearest(
        flag, robot_ts, np.array([40 * MS, 60 * MS], dtype=np.int64)
    )
    assert out.tolist() == [0, 1]


def test_exact_tie_picks_the_earlier_sample():
    """Ties resolve left (``<=``) so the result is deterministic."""
    robot_ts = np.array([0, 100 * MS], dtype=np.int64)
    flag = np.array([0, 1], dtype=np.int8)
    out = _resample_flag_nearest(flag, robot_ts, np.array([50 * MS], dtype=np.int64))
    assert out.tolist() == [0]


def test_camera_frame_before_first_robot_sample_clamps_to_first():
    robot_ts = np.array([100 * MS, 200 * MS], dtype=np.int64)
    flag = np.array([1, 0], dtype=np.int8)
    out = _resample_flag_nearest(flag, robot_ts, np.array([0], dtype=np.int64))
    assert out.tolist() == [1]


def test_camera_frame_after_last_robot_sample_clamps_to_last():
    robot_ts = np.array([100 * MS, 200 * MS], dtype=np.int64)
    flag = np.array([0, 1], dtype=np.int8)
    out = _resample_flag_nearest(
        flag, robot_ts, np.array([10_000 * MS], dtype=np.int64)
    )
    assert out.tolist() == [1]


def test_downsampling_preserves_an_intervention_block():
    """Robot logs at 100 Hz, cameras at 30 Hz. A takeover spanning robot frames
    40-59 (400-600 ms) must still be marked on the camera grid."""
    robot_ts = (np.arange(100) * 10 * MS).astype(np.int64)
    flag = np.zeros(100, dtype=np.int8)
    flag[40:60] = 1
    ref_ts = (np.arange(30) * int(1e9 / 30)).astype(np.int64)

    out = _resample_flag_nearest(flag, robot_ts, ref_ts)

    assert set(np.unique(out)).issubset({0, 1})
    marked = ref_ts[out == 1]
    assert marked.min() >= 390 * MS and marked.max() <= 610 * MS
    assert out.sum() > 0


def test_all_policy_and_all_human_are_preserved():
    robot_ts = (np.arange(10) * 10 * MS).astype(np.int64)
    ref_ts = (np.arange(5) * 20 * MS).astype(np.int64)
    zeros = _resample_flag_nearest(np.zeros(10, dtype=np.int8), robot_ts, ref_ts)
    ones = _resample_flag_nearest(np.ones(10, dtype=np.int8), robot_ts, ref_ts)
    assert zeros.tolist() == [0] * 5
    assert ones.tolist() == [1] * 5


def test_returns_int8_matching_the_reference_grid_length():
    robot_ts = (np.arange(10) * 10 * MS).astype(np.int64)
    ref_ts = (np.arange(7) * 13 * MS).astype(np.int64)
    out = _resample_flag_nearest(np.ones(10), robot_ts, ref_ts)
    assert out.dtype == np.int8
    assert out.shape == (7,)


def test_column_shaped_flag_is_flattened():
    """``robot_data.npz`` may store the flag as (N, 1); it is reshaped to (N,)."""
    robot_ts = (np.arange(4) * 10 * MS).astype(np.int64)
    flag = np.array([[0], [1], [1], [0]], dtype=np.int8)
    out = _resample_flag_nearest(flag, robot_ts, robot_ts)
    assert out.tolist() == [0, 1, 1, 0]
