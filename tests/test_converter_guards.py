"""Guards against unreadable svo2 streams / zero aligned frames (the CleanUpSpill 'negative dimensions' crash)."""

import pytest

from raiden.converter import _check_aligned_frames, _check_initial_grabs


def test_initial_grabs_all_ok_pass():
    _check_initial_grabs(
        {"ego_camera": True, "scene_camera": True},
        {"ego_camera": 1274, "scene_camera": 1273},
    )


def test_initial_grab_failure_names_the_stream_and_index_total():
    with pytest.raises(RuntimeError) as e:
        _check_initial_grabs(
            {"ego_camera": True, "scene_camera": False},
            {"ego_camera": 1274, "scene_camera": 1273},
        )
    assert (
        "scene_camera" in str(e.value)
        and "1273" in str(e.value)
        and "ego_camera" not in str(e.value)
    )


def test_aligned_frames_returns_common_min_for_off_by_one_cameras():
    assert (
        _check_aligned_frames(
            {"ego_camera": 1273, "left_wrist_camera": 1272, "scene_camera": 1272}
        )
        == 1272
    )


@pytest.mark.parametrize("counts", [{"a": 0, "b": 5}, {"a": -1, "b": 3}, {}])
def test_aligned_frames_rejects_zero_or_negative(counts):
    with pytest.raises(RuntimeError):
        _check_aligned_frames(counts)
