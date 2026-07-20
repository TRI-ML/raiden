"""Tests for ``raiden.shardify`` ``--keep`` subsetting.

``_keep_anchor`` decides which anchor frames survive the keep filter, so a whole
recorded rollout can be subset to the human corrections (and the drift window
leading into them) at shard time rather than at record time.

Run with::

    uv run pytest tests/test_shardify.py -v
"""

from __future__ import annotations

import numpy as np

from raiden.shardify import _keep_anchor


def _keep_mask(control_source, keep, keep_context=15):
    """Apply the filter across a whole episode, returning the kept indices."""
    cs = np.asarray(control_source, dtype=np.int8)
    n = len(cs)
    return [t for t in range(n) if _keep_anchor(cs, t, keep, keep_context, n)]


# ---------------------------------------------------------------------------
# keep == "all" — the default, unchanged behaviour
# ---------------------------------------------------------------------------


def test_all_keeps_every_frame():
    cs = [0, 0, 1, 1, 0]
    assert _keep_mask(cs, "all") == [0, 1, 2, 3, 4]


def test_all_keeps_a_pure_teleop_episode():
    """Ordinary demos have control_source all-zero and must be untouched."""
    assert _keep_mask([0] * 6, "all") == list(range(6))


# ---------------------------------------------------------------------------
# keep == "interventions" — corrections only
# ---------------------------------------------------------------------------


def test_interventions_keeps_only_human_frames():
    cs = [0, 0, 1, 1, 0, 1]
    assert _keep_mask(cs, "interventions") == [2, 3, 5]


def test_interventions_on_a_teleop_episode_keeps_nothing():
    """The documented footgun: control_source is 0 for every ordinary teleop
    frame, so this mode empties a non-dagger dataset."""
    assert _keep_mask([0] * 10, "interventions") == []


def test_interventions_ignores_keep_context():
    """Lead-in frames belong to interventions_context only."""
    cs = [0, 0, 0, 1]
    assert _keep_mask(cs, "interventions", keep_context=2) == [3]


# ---------------------------------------------------------------------------
# keep == "interventions_context" — corrections plus the drift window
# ---------------------------------------------------------------------------


def test_context_keeps_the_lead_in_frames():
    """With keep_context=2, the two policy frames before the takeover survive
    and the earlier drift does not."""
    cs = [0, 0, 0, 1, 0]
    assert _keep_mask(cs, "interventions_context", keep_context=2) == [1, 2, 3]


def test_context_does_not_keep_frames_after_an_intervention():
    """The window is a forward look-ahead: it captures the approach to a
    takeover, not the recovery after it."""
    cs = [0, 1, 0, 0, 0]
    kept = _keep_mask(cs, "interventions_context", keep_context=2)
    assert kept == [0, 1]


def test_context_zero_still_keeps_the_frame_immediately_before():
    """max(1, keep_context) means 0 is not silently 'interventions only'."""
    cs = [0, 0, 1]
    assert _keep_mask(cs, "interventions_context", keep_context=0) == [1, 2]


def test_context_window_is_clamped_at_the_episode_end():
    """A window running past the last frame must not raise or wrap."""
    cs = [0, 0, 0]
    assert _keep_mask(cs, "interventions_context", keep_context=100) == []


def test_context_spans_two_separate_takeovers():
    cs = [0, 0, 1, 0, 0, 0, 1, 0]
    kept = _keep_mask(cs, "interventions_context", keep_context=1)
    assert kept == [1, 2, 5, 6]


def test_context_keeps_a_fully_human_episode_intact():
    assert _keep_mask([1] * 5, "interventions_context", keep_context=3) == list(
        range(5)
    )
