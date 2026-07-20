"""Tests for ``raiden.inspector`` verdict classification.

Hardware-free: exercises the pure classification maths, so no ZED SDK and no
SVO2 files are needed (``pyzed`` is imported lazily inside ``_scan_svo2``).

Run with::

    uv run pytest tests/test_inspector.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from raiden.inspector import (
    BLACK_RATIO,
    CLEAN_MAX_DT_MS,
    CLEAN_SHRINK,
    MIN_FRAMES,
    VERDICT_RANK,
    _classify,
    _is_episode_dir,
    _result_from_scan,
)

FRAME_NS = int(1e9 / 30)  # nominal 30 Hz capture period


def _ts(n_frames: int, gaps_ms: dict[int, float] | None = None) -> np.ndarray:
    """Timestamps for ``n_frames`` at a perfect 30 Hz.

    ``gaps_ms`` adds extra delay (in ms) after the given frame index, simulating
    a capture stutter.
    """
    dt = np.full(n_frames - 1, FRAME_NS, dtype=np.int64)
    for idx, extra_ms in (gaps_ms or {}).items():
        dt[idx] += int(extra_ms * 1e6)
    return np.concatenate([[0], np.cumsum(dt)]).astype(np.int64)


# ---------------------------------------------------------------------------
# _classify — the shrink / max-dt / frame-count verdict ladder
# ---------------------------------------------------------------------------


def test_classify_nominal_recording_is_clean():
    assert _classify(shrink=1.0, max_dt_ms=33.4, n_frames=1000) == "clean"


def test_classify_thresholds_are_exclusive():
    """Thresholds compare with a strict ``>``, so a value sitting exactly on the
    boundary stays in the better bucket."""
    assert _classify(CLEAN_SHRINK, 33.4, 1000) == "clean"
    assert _classify(1.0, CLEAN_MAX_DT_MS, 1000) == "clean"


@pytest.mark.parametrize(
    ("shrink", "expected"),
    [(1.0, "clean"), (1.06, "borderline"), (1.11, "bad"), (1.21, "broken")],
)
def test_classify_shrink_ladder(shrink, expected):
    assert _classify(shrink, 33.4, 1000) == expected


@pytest.mark.parametrize(
    ("max_dt_ms", "expected"),
    [(33.4, "clean"), (201.0, "borderline"), (1001.0, "bad"), (2001.0, "broken")],
)
def test_classify_max_dt_ladder(max_dt_ms, expected):
    assert _classify(1.0, max_dt_ms, 1000) == expected


def test_classify_too_few_frames_is_broken():
    """A near-empty recording is broken however good its timing looks."""
    assert _classify(1.0, 33.4, MIN_FRAMES - 1) == "broken"
    assert _classify(1.0, 33.4, MIN_FRAMES) == "clean"


def test_classify_worst_axis_wins():
    """Timing and frame-drop are orthogonal; the worse of the two decides."""
    assert _classify(shrink=1.0, max_dt_ms=1500.0, n_frames=1000) == "bad"
    assert _classify(shrink=1.15, max_dt_ms=33.4, n_frames=1000) == "bad"


def test_verdict_rank_orders_worst_last():
    ranked = sorted(VERDICT_RANK, key=lambda v: VERDICT_RANK[v])
    assert ranked == ["clean", "borderline", "bad", "broken", "black", "missing"]


# ---------------------------------------------------------------------------
# _result_from_scan — raw scan -> classified InspectResult
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_frames", [0, 1])
def test_result_degenerate_scan_is_broken(n_frames):
    """Fewer than two timestamps means no interval to measure — broken, and the
    metrics are zeroed rather than left undefined."""
    res = _result_from_scan(
        _ts(2)[:n_frames], sdk_total=10, n_errors=0, n_black=0, svo_mtime_ns=7
    )
    assert res.verdict == "broken"
    assert res.n_frames == n_frames
    assert res.shrink == 0.0
    assert res.max_dt_ms == 0.0
    assert res.svo2_mtime_ns == 7


def test_result_nominal_scan_is_clean():
    res = _result_from_scan(
        _ts(1000), sdk_total=1000, n_errors=0, n_black=0, svo_mtime_ns=0
    )
    assert res.verdict == "clean"
    assert res.n_frames == 1000
    assert res.shrink == pytest.approx(1.0, abs=1e-3)
    assert res.max_dt_ms == pytest.approx(1000 / 30, abs=1e-3)
    assert res.gaps_over_50ms == 0


def test_result_stutter_degrades_the_verdict_and_counts_gaps():
    """One long stall shows up in max_dt, the gap counters, and the verdict."""
    res = _result_from_scan(
        _ts(1000, {500: 600.0}), sdk_total=1000, n_errors=0, n_black=0, svo_mtime_ns=0
    )
    assert res.verdict == "borderline"
    assert res.max_dt_ms == pytest.approx(1000 / 30 + 600.0, abs=1e-3)
    assert res.gaps_over_500ms == 1
    assert res.gaps_over_50ms == 1


def test_result_dropped_frames_show_up_as_shrink():
    """Half the frames lost over the same wall-clock span doubles shrink, so the
    surviving frames would replay at 2x speed."""
    ts = _ts(1000)[::2]  # keep every other frame: same span, half the frames
    res = _result_from_scan(ts, sdk_total=1000, n_errors=0, n_black=0, svo_mtime_ns=0)
    assert res.shrink == pytest.approx(2.0, abs=1e-2)
    assert res.verdict == "broken"


def test_result_black_left_eye_overrides_timing():
    """A dead left eye corrupts the training RGB outright, so it beats an
    otherwise-clean timing verdict."""
    n_black = int(1000 * BLACK_RATIO) + 1
    res = _result_from_scan(
        _ts(1000), sdk_total=1000, n_errors=0, n_black=n_black, svo_mtime_ns=0
    )
    assert res.verdict == "black"
    assert res.black_ratio == pytest.approx(n_black / 1000)


def test_result_black_ratio_threshold_is_exclusive():
    """Exactly at BLACK_RATIO is not black — the check is a strict ``>``."""
    res = _result_from_scan(
        _ts(1000),
        sdk_total=1000,
        n_errors=0,
        n_black=int(1000 * BLACK_RATIO),
        svo_mtime_ns=0,
    )
    assert res.verdict == "clean"


def test_result_preserves_timing_metrics_under_a_black_verdict():
    """The timing numbers stay available for diagnosis even when black wins."""
    res = _result_from_scan(
        _ts(1000, {500: 600.0}), sdk_total=1000, n_errors=0, n_black=500, svo_mtime_ns=0
    )
    assert res.verdict == "black"
    assert res.gaps_over_500ms == 1


def test_result_reports_grab_errors_verbatim():
    res = _result_from_scan(
        _ts(1000), sdk_total=1010, n_errors=10, n_black=0, svo_mtime_ns=0
    )
    assert res.n_grab_errors == 10
    assert res.sdk_total_frames == 1010


# ---------------------------------------------------------------------------
# _is_episode_dir — what counts as an episode on disk
# ---------------------------------------------------------------------------


def test_is_episode_dir_requires_cameras_with_svo2(tmp_path):
    ep = tmp_path / "0000"
    (ep / "cameras").mkdir(parents=True)
    (ep / "cameras" / "scene_camera.svo2").touch()
    assert _is_episode_dir(ep) is True


def test_is_episode_dir_rejects_empty_cameras_dir(tmp_path):
    ep = tmp_path / "0000"
    (ep / "cameras").mkdir(parents=True)
    assert _is_episode_dir(ep) is False


def test_is_episode_dir_rejects_dir_without_cameras(tmp_path):
    ep = tmp_path / "0000"
    ep.mkdir()
    (ep / "robot_data.npz").touch()
    assert _is_episode_dir(ep) is False


def test_is_episode_dir_rejects_a_file(tmp_path):
    f = tmp_path / "notes.txt"
    f.touch()
    assert _is_episode_dir(f) is False
