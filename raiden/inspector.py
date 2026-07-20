"""Pre-flight quality check for raw SVO2 recordings — flag frame-stutter and
left-eye blackouts before ``rd convert``.

Reads each frame's capture timestamp from every ``cameras/*.svo2`` file and, for
the stereo LEFT view only, checks whether the frame is fully black.  The LEFT
view is the one ``rd convert`` turns into training RGB
(:meth:`raiden.cameras.zed.ZedCamera.get_frame` retrieves ``sl.VIEW.LEFT``), so a
black left eye corrupts the data even when the right eye and frame timing are
perfectly healthy — a failure the timestamp scan alone cannot see.

Stutter matters because :func:`raiden.robot.replay._solve_ik_sequence` treats
processed lowdim files as uniformly-spaced 30 Hz keyframes (``np.arange``-based
indexing).  Any real-time gap between consecutive surviving frames therefore
collapses to a single 1/30 s replay step, producing visible bursts of
unphysical speed during ``rd replay`` and silently distorting the action labels
a policy learns from.

All camera views (``cameras/*.svo2`` — e.g. ``scene_camera``,
``left_wrist_camera``, ``right_wrist_camera``) of each episode are scanned, since
wrist cameras sometimes drop or lose frames independently of the scene camera.
The views of one episode are scanned concurrently (one thread per view).  The
check is camera-name agnostic: any directory containing a ``cameras/`` folder
with at least one ``.svo2`` file is treated as an episode.

Caching
-------
Each scanned episode gets its own ``<episode_dir>/inspect.json`` cache file
holding one entry per camera, each keyed by that camera's SVO2 mtime.  Re-runs
only re-scan the views whose ``.svo2`` has changed since the cache was written
(or are missing from the cache entirely).  The task-level
``<task_dir>/inspect_report.json`` is a derived aggregate, rebuilt from the
per-episode files on every run.
"""

import concurrent.futures
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── verdict thresholds ────────────────────────────────────────────────────────
# ``shrink`` = real elapsed duration / ideal 30 Hz replay duration.  A value of
# 1.0 means the recording ran at exactly 30 Hz; larger means frames were dropped
# and the surviving ones will play back too fast.
CLEAN_SHRINK = 1.05
BORDERLINE_SHRINK = 1.10
BAD_SHRINK = 1.20
CLEAN_MAX_DT_MS = 200.0
BORDERLINE_MAX_DT_MS = 1000.0
BAD_MAX_DT_MS = 2000.0
MIN_FRAMES = 10

# A view whose frame count falls below this fraction of the episode's busiest
# view is flagged as having dropped/missing frames (cameras grab in lock-step at
# 30 Hz, so a large discrepancy means one view lost frames).
FRAME_DROP_RATIO = 0.95

# Left-eye blackout detection. A frame is "black" when its MEAN brightness across
# the RGB channels is at/below BLACK_MEAN_MAX. A dead left eye (sensor dropout)
# sits uniformly at the noise floor (~3/255), while even a dim real scene means in
# the tens-to-hundreds — a ~25x separation. We use the mean, not the per-pixel
# max: a visually-black frame still carries read noise and stray hot pixels whose
# raw channel values routinely exceed any low cutoff, so a max-based test silently
# misses real blackouts. A view is flagged once more than BLACK_RATIO of its
# frames are black.
BLACK_MEAN_MAX = 10.0
BLACK_RATIO = 0.10

# The LEFT view is retrieved downscaled to this resolution purely to measure mean
# brightness. A black frame is black at any resolution, so downscaling keeps the
# per-frame check on EVERY frame (no temporal sampling — a blackout starting
# anywhere is caught) while cutting its cost ~8x: full-res adds ~190% to the scan,
# this adds ~25%, since the unavoidable per-frame ``grab`` decode dominates.
BLACK_CHECK_WIDTH = 320
BLACK_CHECK_HEIGHT = 180

VERDICT_RANK = {
    "clean": 0,
    "borderline": 1,
    "bad": 2,
    "broken": 3,
    "black": 4,
    "missing": 5,
}

EPISODE_CACHE_NAME = "inspect.json"
TASK_REPORT_NAME = "inspect_report.json"

# Default concurrent SVO2 scans. Each scan is a decode loop that releases the GIL
# during ``grab``, but the ZED SDK saturates around 8 concurrent decodes (more
# threads only open extra cameras for no throughput gain), so cap the default
# there. Override with ``--workers`` for unusual hardware.
DEFAULT_SCAN_WORKERS = 8


def _default_workers() -> int:
    return min(os.cpu_count() or 4, DEFAULT_SCAN_WORKERS)


def _is_episode_dir(d: Path) -> bool:
    """An episode is any directory holding a ``cameras/`` folder with ≥1 ``.svo2``."""
    cam_dir = d / "cameras"
    return d.is_dir() and cam_dir.is_dir() and any(cam_dir.glob("*.svo2"))


@dataclass
class InspectResult:
    n_frames: int
    sdk_total_frames: int
    n_grab_errors: int
    n_black_frames: int
    black_ratio: float
    median_dt_ms: float
    mean_dt_ms: float
    max_dt_ms: float
    p99_dt_ms: float
    gaps_over_50ms: int
    gaps_over_100ms: int
    gaps_over_500ms: int
    real_duration_s: float
    replay_duration_s: float
    shrink: float
    worst_burst_at_replay_s: float
    verdict: str
    svo2_mtime_ns: int


def _scan_svo2(svo_path: Path) -> Tuple[np.ndarray, int, int, int]:
    """Walk an SVO2 file, collecting per-frame timestamps and left-eye blackouts.

    Returns ``(timestamps_ns, sdk_total_frames, n_grab_errors, n_black_frames)``.
    ``n_black_frames`` counts frames whose stereo LEFT view — the eye
    ``rd convert`` writes to training RGB — is black (mean RGB brightness
    ``<= BLACK_MEAN_MAX``).  ``pyzed`` is imported lazily so ``rd inspect
    --help`` works on machines without the SDK.
    """
    import pyzed.sl as sl

    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    # We read frame timestamps and the LEFT image only — skip depth (above),
    # self-calibration and sensor ingest to cut per-file overhead (matters when
    # scanning hundreds of episodes).
    init.camera_disable_self_calib = True
    init.sensors_required = False

    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open {svo_path}: {err}")

    sdk_total = cam.get_svo_number_of_frames()
    runtime = sl.RuntimeParameters()
    left = sl.Mat()
    low_res = sl.Resolution(BLACK_CHECK_WIDTH, BLACK_CHECK_HEIGHT)
    ts: List[int] = []
    n_errors = 0
    n_black = 0
    try:
        while True:
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                n_errors += 1
                continue
            ts.append(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            # Same eye convert uses (VIEW.LEFT), downscaled — RGB channels only
            # (get_data() is H x W x 4 BGRA; drop alpha).
            cam.retrieve_image(left, sl.VIEW.LEFT, sl.MEM.CPU, low_res)
            if left.get_data()[:, :, :3].mean() <= BLACK_MEAN_MAX:
                n_black += 1
    finally:
        left.free()
        cam.close()

    return np.asarray(ts, dtype=np.int64), int(sdk_total), n_errors, n_black


def _classify(shrink: float, max_dt_ms: float, n_frames: int) -> str:
    if n_frames < MIN_FRAMES or shrink > BAD_SHRINK or max_dt_ms > BAD_MAX_DT_MS:
        return "broken"
    if shrink > BORDERLINE_SHRINK or max_dt_ms > BORDERLINE_MAX_DT_MS:
        return "bad"
    if shrink > CLEAN_SHRINK or max_dt_ms > CLEAN_MAX_DT_MS:
        return "borderline"
    return "clean"


def _result_from_scan(
    ts: np.ndarray,
    sdk_total: int,
    n_errors: int,
    n_black: int,
    svo_mtime_ns: int,
) -> InspectResult:
    """Turn a raw SVO2 scan into a classified :class:`InspectResult`.

    Split out from :func:`_analyze_svo` so the (pure) classification maths can be
    unit-tested without the ZED SDK.
    """
    n = len(ts)
    if n < 2:
        return InspectResult(
            n_frames=n,
            sdk_total_frames=sdk_total,
            n_grab_errors=n_errors,
            n_black_frames=n_black,
            black_ratio=float(n_black / n) if n else 0.0,
            median_dt_ms=0.0,
            mean_dt_ms=0.0,
            max_dt_ms=0.0,
            p99_dt_ms=0.0,
            gaps_over_50ms=0,
            gaps_over_100ms=0,
            gaps_over_500ms=0,
            real_duration_s=0.0,
            replay_duration_s=0.0,
            shrink=0.0,
            worst_burst_at_replay_s=0.0,
            verdict="broken",
            svo2_mtime_ns=svo_mtime_ns,
        )

    dt_ms = np.diff(ts) / 1e6
    real_s = float((ts[-1] - ts[0]) / 1e9)
    replay_s = (n - 1) / 30.0
    shrink = real_s / replay_s if replay_s > 0 else 0.0
    max_dt = float(dt_ms.max())
    black_ratio = n_black / n

    # A black left eye corrupts the training data outright, so it overrides the
    # (orthogonal) timing verdict — the timing metrics stay in the result either
    # way for diagnosis.
    verdict = "black" if black_ratio > BLACK_RATIO else _classify(shrink, max_dt, n)

    return InspectResult(
        n_frames=n,
        sdk_total_frames=sdk_total,
        n_grab_errors=n_errors,
        n_black_frames=n_black,
        black_ratio=float(black_ratio),
        median_dt_ms=float(np.median(dt_ms)),
        mean_dt_ms=float(np.mean(dt_ms)),
        max_dt_ms=max_dt,
        p99_dt_ms=float(np.percentile(dt_ms, 99)),
        gaps_over_50ms=int(np.sum(dt_ms > 50)),
        gaps_over_100ms=int(np.sum(dt_ms > 100)),
        gaps_over_500ms=int(np.sum(dt_ms > 500)),
        real_duration_s=real_s,
        replay_duration_s=replay_s,
        shrink=float(shrink),
        worst_burst_at_replay_s=float(int(dt_ms.argmax())) / 30.0,
        verdict=verdict,
        svo2_mtime_ns=svo_mtime_ns,
    )


def _analyze_svo(svo_path: Path) -> InspectResult:
    svo_mtime_ns = svo_path.stat().st_mtime_ns
    ts, sdk_total, n_errors, n_black = _scan_svo2(svo_path)
    return _result_from_scan(ts, sdk_total, n_errors, n_black, svo_mtime_ns)


def _episode_svo_paths(episode_dir: Path) -> Dict[str, Path]:
    """Map ``camera_name -> svo2_path`` for every view present in the episode."""
    cam_dir = episode_dir / "cameras"
    return {p.stem: p for p in sorted(cam_dir.glob("*.svo2"))}


def _load_episode_cache(episode_dir: Path, force: bool) -> Dict[str, InspectResult]:
    """Return the per-camera cached results that are still fresh.

    A camera's entry is fresh when its ``svo2_mtime_ns`` matches the current
    ``.svo2`` mtime.  Cameras with stale/absent entries are omitted (forcing a
    rescan of just those views).  ``force``, a missing/corrupt cache, or an
    old flat-schema cache all yield ``{}`` (full rescan).
    """
    if force:
        return {}
    cache_path = episode_dir / EPISODE_CACHE_NAME
    if not cache_path.exists():
        return {}
    try:
        cached = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return {}

    svo_paths = _episode_svo_paths(episode_dir)
    fresh: Dict[str, InspectResult] = {}
    for cam, entry in cached.items():
        if not isinstance(entry, dict):
            return {}  # legacy flat-schema cache — rescan everything
        svo = svo_paths.get(cam)
        if svo is None:
            continue
        try:
            r = InspectResult(**entry)
        except (TypeError, KeyError):
            continue
        if r.svo2_mtime_ns == svo.stat().st_mtime_ns:
            fresh[cam] = r
    return fresh


def _save_episode_cache(episode_dir: Path, results: Dict[str, InspectResult]) -> None:
    cache_path = episode_dir / EPISODE_CACHE_NAME
    cache_path.write_text(
        json.dumps({cam: asdict(r) for cam, r in results.items()}, indent=2)
    )


def _frame_drops(views: Dict[str, InspectResult]) -> set:
    """Cameras whose frame count is anomalously low vs the busiest view."""
    if len(views) < 2:
        return set()
    ref = max(r.n_frames for r in views.values())
    if ref <= 0:
        return set()
    return {cam for cam, r in views.items() if r.n_frames < ref * FRAME_DROP_RATIO}


def _episode_verdict(
    views: Dict[str, InspectResult], drops: set, missing: List[str]
) -> str:
    """Worst per-view verdict, escalated for dropped or missing views."""
    verdicts = [r.verdict for r in views.values()]
    if drops:
        verdicts.append("bad")
    if missing:
        verdicts.append("missing")
    if not verdicts:
        return "missing"
    return max(verdicts, key=lambda v: VERDICT_RANK.get(v, 0))


def _print_table(items: List[Tuple[str, Dict]]) -> None:
    header = (
        f"{'ep':>5} {'cam':>18} {'n':>5} {'med':>6} {'max':>8} {'p99':>7} "
        f"{'g>50':>5} {'g>100':>6} {'g>500':>6} {'blk%':>6} {'shrink':>7} "
        f"{'verdict':>11} {'flag':>8}"
    )
    print(header)
    print("-" * len(header))
    for ep_id, ep in items:
        views: Dict[str, InspectResult] = ep["views"]
        drops: set = ep["drops"]
        missing: List[str] = ep["missing"]
        first = True
        for cam in sorted(views):
            r = views[cam]
            flag = "BLACK" if r.verdict == "black" else "DROP" if cam in drops else ""
            print(
                f"{ep_id if first else '':>5} {cam:>18} {r.n_frames:>5} "
                f"{r.median_dt_ms:>6.2f} {r.max_dt_ms:>8.1f} {r.p99_dt_ms:>7.1f} "
                f"{r.gaps_over_50ms:>5} {r.gaps_over_100ms:>6} {r.gaps_over_500ms:>6} "
                f"{r.black_ratio * 100:>5.1f}% {r.shrink:>6.2f}x {r.verdict:>11} {flag:>8}"
            )
            first = False
        for cam in missing:
            print(
                f"{ep_id if first else '':>5} {cam:>18} "
                f"{'-':>5} {'-':>6} {'-':>8} {'-':>7} "
                f"{'-':>5} {'-':>6} {'-':>6} {'-':>6} {'-':>7} "
                f"{'missing':>11} {'MISSING':>8}"
            )
            first = False


def inspect_task(
    task_dir: Path, force: bool = False, workers: Optional[int] = None
) -> Dict:
    """Inspect every episode in a task, print a per-episode table, and save report.

    Per-camera results are cached at ``<episode>/inspect.json`` (each entry
    mtime-keyed against its ``.svo2``); only changed/uncached views are
    rescanned.  Every view that needs a scan — across all episodes — runs in a
    single shared thread pool (``workers`` threads, default
    :func:`_default_workers`), so scans overlap across episodes rather than one
    episode at a time.  The task-level ``inspect_report.json`` aggregate is
    rebuilt from the per-episode files on every run.
    """
    task_dir = Path(task_dir)
    ep_dirs = sorted(d for d in task_dir.iterdir() if _is_episode_dir(d))
    if not ep_dirs:
        print(f"No episodes with cameras/*.svo2 found in {task_dir}")
        return {}

    print(f"Task: {task_dir.name}  ({len(ep_dirs)} episodes)\n")

    # Reuse fresh per-camera cache, then flatten every view still needing a scan
    # — across all episodes — into one job list so the whole task fills a single
    # shared thread pool instead of scanning one episode at a time.
    ep_by_id = {d.name: d for d in ep_dirs}
    raw: Dict[str, Dict[str, InspectResult]] = {}
    jobs: List[Tuple[str, str, Path]] = []
    for ep_dir in ep_dirs:
        cached = _load_episode_cache(ep_dir, force=force)
        raw[ep_dir.name] = dict(cached)
        for cam, svo in _episode_svo_paths(ep_dir).items():
            if cam not in cached:
                jobs.append((ep_dir.name, cam, svo))

    if jobs:
        n_workers = workers or _default_workers()
        print(
            f"Scanning {len(jobs)} view(s) across {len(ep_dirs)} episode(s) "
            f"with {n_workers} workers..."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {
                ex.submit(_analyze_svo, svo): (ep_id, cam) for ep_id, cam, svo in jobs
            }
            for fut in concurrent.futures.as_completed(futs):
                ep_id, cam = futs[fut]
                try:
                    raw[ep_id][cam] = fut.result()
                except Exception as e:
                    print(f"  {ep_id}/{cam}: scan failed ({e})")
        for ep_id in {ep_id for ep_id, _, _ in jobs}:
            if raw[ep_id]:
                _save_episode_cache(ep_by_id[ep_id], raw[ep_id])

    raw = {ep_id: views for ep_id, views in raw.items() if views}
    expected_cams = sorted({cam for views in raw.values() for cam in views})

    episodes: Dict[str, Dict] = {}
    for ep_id, views in raw.items():
        drops = _frame_drops(views)
        missing = [cam for cam in expected_cams if cam not in views]
        episodes[ep_id] = {
            "views": views,
            "drops": drops,
            "missing": missing,
            "verdict": _episode_verdict(views, drops, missing),
        }

    items = sorted(episodes.items())
    _print_table(items)

    counts = {
        "clean": 0,
        "borderline": 0,
        "bad": 0,
        "broken": 0,
        "black": 0,
        "missing": 0,
    }
    for ep in episodes.values():
        counts[ep["verdict"]] = counts.get(ep["verdict"], 0) + 1
    print(
        f"\nSummary (per-episode, worst view): clean={counts['clean']}  "
        f"borderline={counts['borderline']}  bad={counts['bad']}  "
        f"broken={counts['broken']}  black={counts['black']}  "
        f"missing={counts['missing']}  (scanned {len(episodes)})"
    )

    report = {
        "task": task_dir.name,
        "total_episodes": len(ep_dirs),
        "scanned_episodes": len(episodes),
        "expected_cameras": expected_cams,
        "verdicts": counts,
        "thresholds": {
            "clean_shrink_max": CLEAN_SHRINK,
            "borderline_shrink_max": BORDERLINE_SHRINK,
            "bad_shrink_max": BAD_SHRINK,
            "clean_max_dt_ms": CLEAN_MAX_DT_MS,
            "borderline_max_dt_ms": BORDERLINE_MAX_DT_MS,
            "bad_max_dt_ms": BAD_MAX_DT_MS,
            "min_frames": MIN_FRAMES,
            "frame_drop_ratio": FRAME_DROP_RATIO,
            "black_mean_max": BLACK_MEAN_MAX,
            "black_ratio": BLACK_RATIO,
        },
        "episodes": {
            ep_id: {
                "verdict": ep["verdict"],
                "drops": sorted(ep["drops"]),
                "missing": ep["missing"],
                "views": {cam: asdict(r) for cam, r in ep["views"].items()},
            }
            for ep_id, ep in items
        },
    }
    report_path = task_dir / TASK_REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report saved to: {report_path}")

    return report


def select_inspect_tasks(data_dir: str = "data/raw") -> List[str]:
    """Use fzf to select one or more raw recording task directories (Tab to multi-select)."""
    base = Path(data_dir)
    if not base.exists():
        print(f"No recordings found in {base}")
        sys.exit(1)

    task_dirs = sorted(
        d
        for d in base.iterdir()
        if d.is_dir()
        and any(_is_episode_dir(sub) for sub in d.iterdir() if sub.is_dir())
    )

    if not task_dirs:
        print(f"No tasks with cameras/*.svo2 recordings found in {base}")
        sys.exit(1)

    labels = {
        f"{d.name}  ({sum(1 for s in d.iterdir() if _is_episode_dir(s))} recording(s))": d
        for d in task_dirs
    }

    from raiden.utils import fzf_select

    selected = fzf_select(list(labels), prompt="Inspect task(s)> ", multi=True)
    if not selected:
        return []
    if isinstance(selected, str):
        selected = [selected]
    return [str(labels[s]) for s in selected]


def run_inspect(
    recording_dir: Optional[str] = None,
    data_dir: str = "data",
    force: bool = False,
    workers: Optional[int] = None,
) -> None:
    """Entry point for ``rd inspect``.

    With ``recording_dir`` set, inspect exactly that task directory; otherwise
    open an fzf selector over ``<data_dir>/raw`` (Tab to multi-select tasks).
    """
    if recording_dir is not None:
        inspect_task(Path(recording_dir), force=force, workers=workers)
        return
    for task_dir in select_inspect_tasks(str(Path(data_dir) / "raw")):
        inspect_task(Path(task_dir), force=force, workers=workers)
        print()
