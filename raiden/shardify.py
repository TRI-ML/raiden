"""Export converted Raiden episodes to WebDataset sharded .tar format.

Each sample in a shard contains:
  {uuid}.{cam}_t{idx}.png           — camera images at specified time indices (lossless PNG)
  {uuid}.lowdim.npz                 — windowed lowdim arrays  (T × D each key)
  {uuid}.metadata.json              — episode / sample metadata
  {uuid}.language_instructions.json — language annotations

Alongside the shards the following files are written:
  preprocessing_config.yaml — full config snapshot
  manifest.jsonl            — one JSON line per shard: {"shard": ..., "num_sequences": N}
  stats.json                — per-key statistics (mean/std/min/max + percentiles)
  processing_metadata.json
"""

import dataclasses
import io
import json
import pickle
import platform
import random
import socket
import sys
import tarfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ShardifyConfig:
    """Parameters controlling the shardification process."""

    # Required
    output_dir: Path

    # Window parameters
    past_lowdim_steps: int = 1
    future_lowdim_steps: int = 19
    #: Step spacing (in raw frames) between consecutive timesteps in the lowdim
    #: action/proprio window.  Does NOT affect anchor frame sampling (every raw
    #: frame is an anchor → 30 Hz sample density) or image offsets.
    #: Default 3 = 10 Hz action window from 30 Hz recordings.
    #: Set to 1 for native 30 Hz action resolution.
    stride: int = 3
    #: Image time indices in raw frame units relative to the anchor frame
    #: (negative = past).  [-1, 0] fetches the previous raw frame and the
    #: anchor itself — two consecutive 30 Hz frames (1/30 s apart).
    image_indices: List[int] = dataclasses.field(default_factory=lambda: [-1, 0])
    max_padding_left: int = 3
    max_padding_right: int = 15
    padding_strategy: str = "copy"

    # Camera selection and naming
    #: Ordered list of camera names to include.  None = all cameras in data order.
    camera_names: Optional[List[str]] = None
    #: Rename cameras in the output.  Keys = our names, values = desired output names.
    camera_name_map: Dict[str, str] = dataclasses.field(default_factory=dict)

    # Image output
    #: Resize images to (H, W) before storing.  None = no resize.
    resize_images_size: Optional[Tuple[int, int]] = None
    jpeg_quality: int = 95
    use_depth: bool = False

    # Sample filtering
    filter_still_samples: bool = False
    still_threshold: float = 0.05
    fail_on_nan: bool = True

    max_episodes_to_process: int = -1

    # Output
    samples_per_shard: int = 100
    num_workers: int = 8

    #: Maximum samples kept per stat key for percentile estimation.
    stats_reservoir_size: int = 50_000

    #: Only feed every N-th sample into the stats accumulators to save memory.
    stats_stride: int = 10


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _rot6d_to_mat(v: np.ndarray) -> np.ndarray:
    """Convert a (6,) rot6d vector to a (3, 3) rotation matrix via Gram-Schmidt.

    Inverse of ``_rot9_to_rot6d``: reconstructs the full rotation matrix from
    its first two rows.

    Args:
        v: (6,) float array — [R[0,:], R[1,:]].

    Returns:
        (3, 3) float64 rotation matrix.
    """
    a1, a2 = v[:3].astype(np.float64), v[3:6].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def _build_transform(xyz: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
    """Build a 4×4 rigid-body transform from a (3,) position and (6,) rot6d."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rot6d_to_mat(rot6d)
    T[:3, 3] = xyz.astype(np.float64)
    return T


def _rot9_to_rot6d(rot9: np.ndarray) -> np.ndarray:
    """Convert a row-major flattened 3×3 rotation matrix to the 6D representation.

    Uses the first two rows of R: [R[0,:], R[1,:]] = [R00,R01,R02, R10,R11,R12] → (6,).
    Matches vla_foundry's ``matrix_to_rot_6d``: ``R[:2, :].flatten()``.

    Args:
        rot9: (..., 9) float array — row-major flattened 3×3.

    Returns:
        (..., 6) float array.
    """
    mat = rot9.reshape(rot9.shape[:-1] + (3, 3))  # (..., 3, 3)
    return mat[..., :2, :].reshape(rot9.shape[:-1] + (6,))  # first 2 rows


# ---------------------------------------------------------------------------
# Online statistics accumulator  (Welford + reservoir sampling)
# ---------------------------------------------------------------------------


class _StatsAccumulator:
    """Incrementally tracks mean/std/min/max and a reservoir for percentiles.

    Handles data of shape (T, D) per sample.
    """

    def __init__(self, T: int, D: int, reservoir_size: int = 50_000):
        self.T = T
        self.D = D
        self.n = 0
        # Welford's online algorithm per (t, d)
        self._mean = np.zeros((T, D), dtype=np.float64)
        self._M2 = np.zeros((T, D), dtype=np.float64)
        self._min = np.full((T, D), np.inf, dtype=np.float64)
        self._max = np.full((T, D), -np.inf, dtype=np.float64)
        # Reservoir sampling for percentiles
        self._res = np.zeros((reservoir_size, T, D), dtype=np.float32)
        self._res_n = 0
        self._res_size = reservoir_size

    def update(self, sample: np.ndarray) -> None:
        """Add one (T, D) sample."""
        x = sample.astype(np.float64)
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        self._M2 += delta * (x - self._mean)
        self._min = np.minimum(self._min, x)
        self._max = np.maximum(self._max, x)
        # Reservoir sampling
        if self._res_n < self._res_size:
            self._res[self._res_n] = sample.astype(np.float32)
            self._res_n += 1
        else:
            j = int(np.random.randint(0, self.n))
            if j < self._res_size:
                self._res[j] = sample.astype(np.float32)

    def finalize(self) -> Dict[str, Any]:
        """Return a stats dict compatible with the reference stats.json format."""
        if self.n == 0:
            return {}
        # Per-timestep std from Welford M2
        std_per_ts = np.sqrt(self._M2 / max(self.n - 1, 1))  # (T, D)

        # Global mean: since every timestep has the same sample count n, this is exact.
        global_mean = self._mean.mean(axis=0)  # (D,)

        # Global std via parallel Welford combine across T timesteps:
        #   M2_combined[d] = sum_t( M2[t,d] + n * (mean[t,d] - global_mean[d])^2 )
        delta = self._mean - global_mean  # (T, D)
        M2_combined = (self._M2 + self.n * delta**2).sum(axis=0)  # (D,)
        global_std = np.sqrt(M2_combined / max(self.n * self.T - 1, 1))  # (D,)

        global_min = self._min.min(axis=0)  # (D,)
        global_max = self._max.max(axis=0)  # (D,)

        res = self._res[: self._res_n]  # (R, T, D)
        flat = res.reshape(-1, self.D)  # (R*T, D) — all observations flattened
        pcts_global = np.percentile(flat, [1, 2, 5, 95, 98, 99], axis=0)  # (6, D)
        pcts_per_ts = np.percentile(res, [1, 2, 5, 95, 98, 99], axis=0)  # (6, T, D)

        def _to_list(arr: np.ndarray) -> Any:
            if arr.ndim == 1:
                return arr.tolist()
            return [_to_list(row) for row in arr]

        return {
            "mean": _to_list(global_mean),
            "std": _to_list(global_std),
            "min": _to_list(global_min),
            "max": _to_list(global_max),
            "mean_per_timestep": _to_list(self._mean),
            "std_per_timestep": _to_list(std_per_ts),
            "min_per_timestep": _to_list(self._min),
            "max_per_timestep": _to_list(self._max),
            "percentile_1": _to_list(pcts_global[0]),
            "percentile_2": _to_list(pcts_global[1]),
            "percentile_5": _to_list(pcts_global[2]),
            "percentile_95": _to_list(pcts_global[3]),
            "percentile_98": _to_list(pcts_global[4]),
            "percentile_99": _to_list(pcts_global[5]),
            "percentile_1_per_timestep": _to_list(pcts_per_ts[0]),
            "percentile_2_per_timestep": _to_list(pcts_per_ts[1]),
            "percentile_5_per_timestep": _to_list(pcts_per_ts[2]),
            "percentile_95_per_timestep": _to_list(pcts_per_ts[3]),
            "percentile_98_per_timestep": _to_list(pcts_per_ts[4]),
            "percentile_99_per_timestep": _to_list(pcts_per_ts[5]),
            "count": self.n,
            "percentile_sample_count": int(self._res_n),
        }


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------


def _load_episode_frames(ep_dir: Path) -> List[Dict[str, Any]]:
    """Load all per-frame lowdim pkl files from an episode directory."""
    lowdim_dir = ep_dir / "lowdim"
    pkl_files = sorted(lowdim_dir.glob("??????????.pkl"))
    if not pkl_files:
        pkl_files = sorted(lowdim_dir.glob("?????????.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No lowdim .pkl files in {lowdim_dir}")
    frames = []
    for p in pkl_files:
        with open(p, "rb") as f:
            frames.append(pickle.load(f))
    return frames


def _load_rgb(
    ep_dir: Path,
    camera_name: str,
    frame_idx: int,
    resize: Optional[Tuple[int, int]],
) -> Optional[tuple[bytes, str]]:
    """Return (image_bytes, ext) for a frame, optionally resizing.

    When resize is None the raw JPEG bytes from the converter are returned
    unchanged (single lossy pass, quality set by the converter).  When resize
    is specified the image is resized with cv2.INTER_LANCZOS4 (matching the
    server's preprocessing) and encoded as lossless PNG to avoid a second
    lossy JPEG pass.

    Returns:
        (bytes, ext) where ext is "jpg" or "png", or None if the file is absent.
    """
    path = ep_dir / "rgb" / camera_name / f"{frame_idx:010d}.png"
    if not path.exists():
        return None
    if resize is None:
        return path.read_bytes(), "png"
    # Load as RGB numpy array (PIL JPEG decode), resize with cv2.INTER_LANCZOS4
    # (matching the server's preprocessing), then encode as lossless PNG.
    # PNG has no colour-space tag so bytes are written/read as-is; passing RGB
    # directly means PIL.Image.open on the stored PNG returns correct RGB.
    img_np = np.array(Image.open(path))  # H W 3, RGB uint8
    h_out, w_out = resize
    img_np = cv2.resize(img_np, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)
    _, buf = cv2.imencode(".png", img_np)
    return bytes(buf), "png"


def _load_depth_png(ep_dir: Path, camera_name: str, frame_idx: int) -> Optional[bytes]:
    """Return 16-bit PNG bytes for a depth frame, or None if absent.

    Loads the uint16 mm depth stored by the converter as a ``.npz`` and
    re-encodes it as a 16-bit greyscale PNG (lossless, widely supported).
    """
    path = ep_dir / "depth" / camera_name / f"{frame_idx:010d}.npz"
    if not path.exists():
        return None
    depth_mm = np.load(path)["depth"]  # uint16, millimetres
    _, buf = cv2.imencode(".png", depth_mm)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Window / sample building
# ---------------------------------------------------------------------------


def _clamp_frame(idx: int, n_frames: int) -> int:
    return max(0, min(idx, n_frames - 1))


_ROBOT = "yam"


def _build_window_arrays(
    frames: List[Dict[str, Any]],
    anchor_idx: int,
    config: ShardifyConfig,
    output_cam_names: List[str],
) -> Dict[str, np.ndarray]:
    """Build windowed (T, D) arrays for one anchor frame."""
    T = config.past_lowdim_steps + 1 + config.future_lowdim_steps
    n = len(frames)
    R = _ROBOT

    s = config.stride
    window_idx = [
        _clamp_frame(anchor_idx + offset * s, n)
        for offset in range(-config.past_lowdim_steps, config.future_lowdim_steps + 1)
    ]

    def _collect(field: str, start: int, length: int) -> Optional[np.ndarray]:
        rows = []
        for fi in window_idx:
            val = frames[fi].get(field)
            if val is None:
                return None
            rows.append(np.asarray(val[start : start + length], dtype=np.float32))
        return np.stack(rows)  # (T, length)

    out: Dict[str, np.ndarray] = {}

    # ── action: (T, 13) single-arm or (T, 26) bimanual EE poses ──────────
    action_seq = _collect("action", 0, 26)
    if action_seq is not None:
        out[f"robot__action__poses__left::{R}__xyz"] = action_seq[:, 0:3]
        out[f"robot__action__poses__left::{R}__rot_6d"] = _rot9_to_rot6d(
            action_seq[:, 3:12]
        )
        out[f"robot__action__grippers__left::{R}_hand"] = action_seq[:, 12:13]
        if action_seq.shape[1] >= 26:
            out[f"robot__action__poses__right::{R}__xyz"] = action_seq[:, 13:16]
            out[f"robot__action__poses__right::{R}__rot_6d"] = _rot9_to_rot6d(
                action_seq[:, 16:25]
            )
            out[f"robot__action__grippers__right::{R}_hand"] = action_seq[:, 25:26]

    # ── actual poses: FK(actual joints) — same layout as action ─────────
    actual_poses_seq = _collect("actual_poses", 0, 26)
    if actual_poses_seq is not None:
        out[f"robot__actual__poses__left::{R}__xyz"] = actual_poses_seq[:, 0:3]
        out[f"robot__actual__poses__left::{R}__rot_6d"] = _rot9_to_rot6d(
            actual_poses_seq[:, 3:12]
        )
        out[f"robot__actual__grippers__left::{R}_hand"] = actual_poses_seq[:, 12:13]
        if actual_poses_seq.shape[1] >= 26:
            out[f"robot__actual__poses__right::{R}__xyz"] = actual_poses_seq[:, 13:16]
            out[f"robot__actual__poses__right::{R}__rot_6d"] = _rot9_to_rot6d(
                actual_poses_seq[:, 16:25]
            )
            out[f"robot__actual__grippers__right::{R}_hand"] = actual_poses_seq[
                :, 25:26
            ]

    # ── joints: (T, 7) single-arm or (T, 14) bimanual joint positions ────
    joints_seq = _collect("joints", 0, 14)
    if joints_seq is not None:
        out[f"robot__actual__joint_position__left::{R}"] = joints_seq[:, 0:7]
        if joints_seq.shape[1] >= 14:
            out[f"robot__actual__joint_position__right::{R}"] = joints_seq[:, 7:14]

    # ── action_joints: (T, 7) single-arm or (T, 14) bimanual commanded ───
    act_joints_seq = _collect("action_joints", 0, 14)
    if act_joints_seq is not None:
        out[f"robot__desired__joint_position__left::{R}"] = act_joints_seq[:, 0:7]
        if act_joints_seq.shape[1] >= 14:
            out[f"robot__desired__joint_position__right::{R}"] = act_joints_seq[:, 7:14]

    # ── intrinsics / extrinsics ───────────────────────────────────────────
    # Stored at the image timesteps (raw frame offsets, 30 Hz) so they align
    # with the RGB and depth images.  Shape: (len(image_indices), 3, 3) and
    # (len(image_indices), 4, 4) respectively.
    anchor_frame = frames[anchor_idx]
    img_frame_indices = [_clamp_frame(anchor_idx + i, n) for i in config.image_indices]
    for cam_name in output_cam_names:
        src_cam = _reverse_map(config.camera_name_map, cam_name)
        K = anchor_frame.get("intrinsics", {}).get(src_cam)
        if K is not None:
            K_arr = np.asarray(K, dtype=np.float32)
            out[f"intrinsics.{cam_name}"] = np.tile(
                K_arr[None], (len(img_frame_indices), 1, 1)
            )

        ext_rows = []
        for fi in img_frame_indices:
            ext = frames[fi].get("extrinsics", {}).get(src_cam)
            if ext is None:
                ext = np.eye(4, dtype=np.float32)
            ext_rows.append(np.asarray(ext, dtype=np.float64))
        out[f"extrinsics.{cam_name}"] = np.stack(ext_rows)

    # ── relative poses (relative to anchor actual pose) ───────────────────
    # For each arm side that has actual pose data, compute xyz/rot6d/gripper
    # offsets relative to the anchor frame's actual pose.  Both action and
    # actual pose sequences get a relative variant.  The reference is always
    # the anchor actual pose so the policy sees displacements from "where the
    # robot is now".
    anchor_i = config.past_lowdim_steps
    for side in ("left", "right"):
        anc_xyz_key = f"robot__actual__poses__{side}::{R}__xyz"
        anc_rot_key = f"robot__actual__poses__{side}::{R}__rot_6d"
        if anc_xyz_key not in out:
            continue  # arm not present (single-arm episode)

        anc_xyz = out[anc_xyz_key][anchor_i]  # (3,)
        anc_rot6d = out[anc_rot_key][anchor_i]  # (6,)
        T_anc_inv = np.linalg.inv(_build_transform(anc_xyz, anc_rot6d))  # (4, 4)

        for src in ("action", "actual"):
            if src == "action":
                xyz_key = f"robot__action__poses__{side}::{R}__xyz"
                rot_key = f"robot__action__poses__{side}::{R}__rot_6d"
                out_xyz = f"robot__action__poses__{side}::{R}__xyz_relative"
                out_rot = f"robot__action__poses__{side}::{R}__rot_6d_relative"
            else:
                xyz_key = f"robot__actual__poses__{side}::{R}__xyz"
                rot_key = f"robot__actual__poses__{side}::{R}__rot_6d"
                out_xyz = f"robot__actual__poses__{side}::{R}__xyz_relative"
                out_rot = f"robot__actual__poses__{side}::{R}__rot_6d_relative"

            if xyz_key not in out:
                continue

            xyz_seq = out[xyz_key]  # (T, 3)
            rot6d_seq = out[rot_key]  # (T, 6)

            rel_xyz = np.empty_like(xyz_seq)
            rel_rot6d = np.empty_like(rot6d_seq)
            for i in range(xyz_seq.shape[0]):
                T_t = _build_transform(xyz_seq[i], rot6d_seq[i])
                T_rel = T_anc_inv @ T_t
                rel_xyz[i] = T_rel[:3, 3].astype(np.float32)
                rel_rot6d[i] = T_rel[:2, :3].flatten().astype(np.float32)

            out[out_xyz] = rel_xyz
            out[out_rot] = rel_rot6d

    # ── masks ─────────────────────────────────────────────────────────────
    past_mask = np.zeros(T, dtype=bool)
    past_mask[: config.past_lowdim_steps] = True
    future_mask = np.zeros(T, dtype=bool)
    future_mask[config.past_lowdim_steps + 1 :] = True
    out["past_mask"] = past_mask
    out["future_mask"] = future_mask

    return out


def _reverse_map(camera_name_map: Dict[str, str], out_name: str) -> str:
    """Return the source camera name for a given output camera name."""
    for src, dst in camera_name_map.items():
        if dst == out_name:
            return src
    return out_name  # not in map → same name


def _is_still(
    action_seq: Optional[np.ndarray], anchor_idx: int, threshold: float
) -> bool:
    """Return True if the future EE trajectory barely moves (below threshold)."""
    if action_seq is None:
        return False
    future = action_seq[anchor_idx + 1 :]
    if len(future) == 0:
        return False
    anchor_xyz_l = action_seq[anchor_idx, 0:3]
    anchor_xyz_r = action_seq[anchor_idx, 13:16]
    max_move = max(
        float(np.linalg.norm(future[:, 0:3] - anchor_xyz_l, axis=1).max()),
        float(np.linalg.norm(future[:, 13:16] - anchor_xyz_r, axis=1).max()),
    )
    return max_move < threshold


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


class _ShardWriter:
    """Writes samples into sequential tar shards."""

    def __init__(self, shard_dir: Path, samples_per_shard: int):
        self._shard_dir = shard_dir
        self._sps = samples_per_shard
        self._shard_idx = 0
        self._buf: List[Dict[str, bytes]] = []
        self._shard_counts: List[int] = []

    def add(self, files: Dict[str, bytes]) -> None:
        self._buf.append(files)
        if len(self._buf) >= self._sps:
            self._flush()

    def close(self) -> None:
        if self._buf:
            self._flush()

    def _flush(self) -> None:
        name = f"shard_{self._shard_idx:06d}.tar"
        path = self._shard_dir / name
        with tarfile.open(path, "w") as tf:
            for sample in self._buf:
                for fname, data in sample.items():
                    info = tarfile.TarInfo(name=fname)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
        self._shard_counts.append(len(self._buf))
        self._shard_idx += 1
        self._buf = []

    def manifest_lines(self) -> List[str]:
        return [
            json.dumps({"shard": f"shard_{i:06d}", "num_sequences": n})
            for i, n in enumerate(self._shard_counts)
        ]


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------


def _s3_prefix_exists(s3_client, bucket: str, prefix: str) -> int:
    """Return the number of objects under s3://bucket/prefix/, or 0 if none."""
    prefix_with_slash = prefix.rstrip("/") + "/"
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix_with_slash, MaxKeys=1)
    return resp.get("KeyCount", 0)


def _s3_delete_prefix(s3_client, bucket: str, prefix: str) -> int:
    """Delete all objects under s3://bucket/prefix/. Returns count deleted."""
    prefix_with_slash = prefix.rstrip("/") + "/"
    deleted = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_with_slash):
        objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objects:
            s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            deleted += len(objects)
    return deleted


def _s3_backup_prefix(s3_client, bucket: str, prefix: str) -> str:
    """Copy all objects under prefix/ to prefix_backup_<timestamp>/ and return the backup prefix."""
    import datetime  # noqa: PLC0415

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_prefix = f"{prefix.rstrip('/')}_backup_{ts}"
    prefix_with_slash = prefix.rstrip("/") + "/"
    backup_with_slash = backup_prefix + "/"
    paginator = s3_client.get_paginator("list_objects_v2")
    copied = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_with_slash):
        for obj in page.get("Contents", []):
            src_key = obj["Key"]
            dst_key = backup_with_slash + src_key[len(prefix_with_slash) :]
            s3_client.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": src_key},
                Key=dst_key,
            )
            copied += 1
    print(f"  Backed up {copied} object(s) to s3://{bucket}/{backup_prefix}/")
    return backup_prefix


def _prompt_s3_overwrite(bucket: str, prefix: str) -> str:
    """Prompt the user what to do when the S3 destination already has data.

    Returns one of: ``"delete"``, ``"backup"``, ``"cancel"``.
    """
    print("\nS3 destination already contains data:")
    print(f"  s3://{bucket}/{prefix}/")
    print()
    print("  [d] Delete existing shards and overwrite")
    print("  [b] Back up existing shards then overwrite")
    print("  [c] Cancel")
    print()
    while True:
        choice = input("Choice [d/b/c]: ").strip().lower()
        if choice in ("d", "b", "c"):
            return {"d": "delete", "b": "backup", "c": "cancel"}[choice]
        print("  Please enter d, b, or c.")


def upload_to_s3(local_dir: Path, s3_bucket: str, s3_prefix: str) -> None:
    """Upload all files in local_dir to s3://s3_bucket/s3_prefix/."""
    import boto3  # noqa: PLC0415

    s3 = boto3.client("s3")
    files = sorted(local_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"Uploading {len(files)} files to s3://{s3_bucket}/{s3_prefix}/")
    for f in files:
        key = f"{s3_prefix}/{f.relative_to(local_dir)}"
        s3.upload_file(str(f), s3_bucket, key)
        print(f"  {key}")
    print("Upload complete.")


# ---------------------------------------------------------------------------
# Preprocessing config writer
# ---------------------------------------------------------------------------


def _write_preprocessing_config(
    shard_dir: Path,
    config: ShardifyConfig,
    output_dir_str: str,
    episode_dirs: List[Path],
) -> None:
    """Write preprocessing_config.yaml alongside the shards."""
    import yaml  # noqa: PLC0415

    output_cam_names = _resolve_output_cam_names(config, episode_dirs)
    cfg = {
        "camera_names": output_cam_names,
        "compute_statistics": True,
        "fail_on_nan": config.fail_on_nan,
        "filter_still_samples": config.filter_still_samples,
        "future_lowdim_steps": config.future_lowdim_steps,
        "image_indices": list(config.image_indices),
        "image_format": "png",
        "max_episodes_to_process": config.max_episodes_to_process,
        "max_padding_left": config.max_padding_left,
        "max_padding_right": config.max_padding_right,
        "num_workers": config.num_workers,
        "output_dir": output_dir_str,
        "padding_strategy": config.padding_strategy,
        "past_lowdim_steps": config.past_lowdim_steps,
        "resize_images_size": list(config.resize_images_size)
        if config.resize_images_size
        else None,
        "samples_per_shard": config.samples_per_shard,
        "still_threshold": config.still_threshold,
        "stride": config.stride,
        "validation_episodes_path": None,
    }
    with open(shard_dir / "preprocessing_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)


def _resolve_output_cam_names(
    config: ShardifyConfig, episode_dirs: List[Path]
) -> List[str]:
    """Return the ordered list of output camera names."""
    if config.camera_names is not None:
        return list(config.camera_names)
    # Infer from first available episode
    for ep_dir in episode_dirs:
        pkl_files = sorted((ep_dir / "lowdim").glob("??????????.pkl"))
        if not pkl_files:
            continue
        with open(pkl_files[0], "rb") as f:
            frame = pickle.load(f)
        src_cams = list(frame.get("extrinsics", {}).keys())
        return [config.camera_name_map.get(c, c) for c in src_cams]
    return []


# ---------------------------------------------------------------------------
# Interactive task selection
# ---------------------------------------------------------------------------


def select_processed_task(data_dir: str = "data") -> List[Tuple[Path, List[Path]]]:
    """Use fzf to select one or more converted tasks.

    Returns a list of ``(task_dir, episode_dirs)`` tuples for each selected task.
    """
    import sys  # noqa: PLC0415

    from raiden.utils import fzf_select  # noqa: PLC0415

    base = Path(data_dir) / "processed"
    task_dirs = sorted(
        (
            d
            for d in base.iterdir()
            if d.is_dir()
            and any(
                (ep / "metadata.json").exists() for ep in d.iterdir() if ep.is_dir()
            )
        ),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not task_dirs:
        print(f"No converted tasks found in {base}")
        sys.exit(1)

    _ALL_LABEL = "*** ALL TASKS ***"
    labels = {
        f"{d.name}  ({sum(1 for ep in d.iterdir() if ep.is_dir() and (ep / 'metadata.json').exists())} episode(s))": d
        for d in task_dirs
    }
    choices = [_ALL_LABEL] + list(labels)
    selected = fzf_select(
        choices,
        prompt="Shardify task(s)> ",
        multi=True,
        header="Tab: toggle select  |  Enter: confirm  |  Select '*** ALL TASKS ***' to shardify everything",
    )

    chosen_dirs = task_dirs if _ALL_LABEL in selected else [labels[s] for s in selected]

    result = []
    for task_dir in chosen_dirs:
        episode_dirs = sorted(
            ep
            for ep in task_dir.iterdir()
            if ep.is_dir() and (ep / "metadata.json").exists()
        )
        result.append((task_dir, episode_dirs))
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _build_sample(
    args: tuple,
) -> Dict[str, Any]:
    """Build one sample for the shard writer (runs in a worker thread).

    Args:
        args: (ctx, t, config, output_cam_names) — packed as a tuple so
            ThreadPoolExecutor.map can call this as a plain function.

    Returns:
        A dict with key ``"filtered"`` set to ``"padding"``, ``"still"``, or
        ``"nan"`` when the sample is dropped, or ``None`` when it should be
        written.  Kept samples also carry ``"sample_files"``, ``"lowdim"``,
        ``"episode_id"``, and ``"lang"`` keys.
    """
    ctx, t, config, output_cam_names = args
    frames = ctx["frames"]
    n_frames = ctx["n_frames"]
    src_cam_names = ctx["src_cam_names"]
    episode_id = ctx["episode_id"]
    language_task = ctx["language_task"]
    language_prompt = ctx["language_prompt"]
    control = ctx["control"]
    ep_dir = ctx["ep_dir"]
    s = config.stride

    # ── padding filter ────────────────────────────────────────────────
    left_frames_needed = config.past_lowdim_steps * s
    right_frames_needed = config.future_lowdim_steps * s
    left_pad = max(0, -(-max(0, left_frames_needed - t) // s))
    right_pad = max(0, -(-max(0, right_frames_needed - (n_frames - 1 - t)) // s))
    if left_pad > config.max_padding_left or right_pad > config.max_padding_right:
        return {"filtered": "padding"}

    # ── still-sample filter ───────────────────────────────────────────
    if config.filter_still_samples:
        window_actions = np.stack(
            [
                np.asarray(
                    frames[_clamp_frame(t + o, n_frames)].get("action", np.zeros(26)),
                    dtype=np.float32,
                )
                for o in range(
                    -config.past_lowdim_steps, config.future_lowdim_steps + 1
                )
            ]
        )
        if _is_still(window_actions, config.past_lowdim_steps, config.still_threshold):
            return {"filtered": "still"}

    # ── build lowdim arrays ───────────────────────────────────────────
    lowdim = _build_window_arrays(frames, t, config, output_cam_names)

    # ── NaN check ─────────────────────────────────────────────────────
    if config.fail_on_nan:
        for key, arr in lowdim.items():
            if isinstance(arr, np.ndarray) and arr.dtype.kind == "f":
                if np.isnan(arr).any():
                    raise ValueError(
                        f"NaN in key '{key}' at episode {episode_id} frame {t}"
                    )
    else:
        if any(
            isinstance(arr, np.ndarray)
            and arr.dtype.kind == "f"
            and np.isnan(arr).any()
            for arr in lowdim.values()
        ):
            return {"filtered": "nan"}

    # ── images ────────────────────────────────────────────────────────
    sample_files: Dict[str, bytes] = {}
    sample_uuid = str(uuid.uuid4())

    for img_idx in config.image_indices:
        abs_frame = _clamp_frame(t + img_idx, n_frames)
        suffix = f"t{img_idx}"
        for src_cam, out_cam in zip(src_cam_names, output_cam_names):
            result = _load_rgb(ep_dir, src_cam, abs_frame, config.resize_images_size)
            if result is not None:
                rgb, img_ext = result
                sample_files[f"{sample_uuid}.{out_cam}_{suffix}.{img_ext}"] = rgb
            if config.use_depth:
                depth = _load_depth_png(ep_dir, src_cam, abs_frame)
                if depth is not None:
                    sample_files[f"{sample_uuid}.{out_cam}_{suffix}.depth.png"] = depth

    # ── serialize lowdim ──────────────────────────────────────────────
    buf = io.BytesIO()
    np.savez_compressed(buf, **lowdim)
    sample_files[f"{sample_uuid}.lowdim.npz"] = buf.getvalue()

    # ── metadata.json ─────────────────────────────────────────────────
    img_ts = [_clamp_frame(t + i, n_frames) for i in config.image_indices]
    sample_meta = {
        "episode_id": episode_id,
        "sample_id": f"{sample_uuid}_{episode_id}_t{t:04d}",
        "anchor_timestep": t,
        "anchor_relative_idx": config.past_lowdim_steps,
        "image_timesteps": img_ts,
        "lowdim_start_timestep": max(0, t - config.past_lowdim_steps),
        "lowdim_end_timestep": min(n_frames - 1, t + config.future_lowdim_steps),
        "past_padding": int(left_pad),
        "future_padding": int(right_pad),
        "camera_names": output_cam_names,
        "original_episode_length": n_frames,
        "is_padded": left_pad > 0 or right_pad > 0,
        "control": control,
    }
    sample_files[f"{sample_uuid}.metadata.json"] = json.dumps(sample_meta).encode()

    lang = {"original": [language_prompt or language_task]}
    sample_files[f"{sample_uuid}.language_instructions.json"] = json.dumps(
        lang
    ).encode()

    return {
        "filtered": None,
        "sample_files": sample_files,
        "lowdim": lowdim,
        "episode_id": episode_id,
    }


def run_shardify(
    episode_dirs: List[Path],
    config: ShardifyConfig,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
) -> None:
    """Convert a list of converted episode directories to sharded WebDataset format.

    Args:
        episode_dirs: Paths to episode directories, each containing ``lowdim/``,
            ``rgb/``, and ``metadata.json``.
        config: Shardification parameters.
        s3_bucket: If set, upload the shards directory to this S3 bucket.
        s3_prefix: S3 key prefix for upload (e.g. ``yam_datasets/task_name``).
    """
    t_start = time.time()
    shard_dir = config.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # ── S3 pre-check: prompt before overwriting existing data ─────────────
    if s3_bucket and s3_prefix:
        import boto3 as _boto3  # noqa: PLC0415

        _s3 = _boto3.client("s3")
        if _s3_prefix_exists(_s3, s3_bucket, s3_prefix):
            action = _prompt_s3_overwrite(s3_bucket, s3_prefix)
            if action == "cancel":
                print("Cancelled.")
                return
            elif action == "backup":
                print("Backing up existing shards...")
                _s3_backup_prefix(_s3, s3_bucket, s3_prefix)
                print("Deleting existing shards...")
                _s3_delete_prefix(_s3, s3_bucket, s3_prefix)
            elif action == "delete":
                print("Deleting existing shards...")
                n = _s3_delete_prefix(_s3, s3_bucket, s3_prefix)
                print(f"  Deleted {n} object(s).")

    # Resolve which episodes to process
    eps = list(episode_dirs)
    random.shuffle(eps)
    if config.max_episodes_to_process > 0:
        eps = eps[: config.max_episodes_to_process]

    # Resolve output camera name ordering
    output_cam_names = _resolve_output_cam_names(config, eps)

    # Write preprocessing config
    output_dir_str = (
        f"s3://{s3_bucket}/{s3_prefix}"
        if s3_bucket and s3_prefix
        else str(config.output_dir)
    )
    _write_preprocessing_config(shard_dir, config, output_dir_str, eps)

    T = config.past_lowdim_steps + 1 + config.future_lowdim_steps
    writer = _ShardWriter(shard_dir, config.samples_per_shard)
    stats_accumulators: Dict[str, _StatsAccumulator] = {}

    total_samples = 0
    filtered_padding = 0
    filtered_still = 0
    filtered_nan = 0
    stats_counter = 0

    # ── Phase 1: load all episodes ────────────────────────────────────────
    print(f"Loading {len(eps)} episode(s)...")
    ep_contexts: List[dict] = []
    skipped = 0
    total_frames = 0
    for i, ep_dir in enumerate(eps, 1):
        if not (ep_dir / "metadata.json").exists():
            print(f"  [{i}/{len(eps)}] SKIP {ep_dir.name}: no metadata.json")
            skipped += 1
            continue
        try:
            frames = _load_episode_frames(ep_dir)
        except FileNotFoundError as e:
            print(f"  [{i}/{len(eps)}] SKIP {ep_dir.name}: {e}")
            skipped += 1
            continue
        anchor_frame = frames[0]
        with open(ep_dir / "metadata.json") as _mf:
            _ep_meta = json.load(_mf)
        ep_contexts.append(
            {
                "ep_dir": ep_dir,
                "frames": frames,
                "n_frames": len(frames),
                "src_cam_names": [
                    _reverse_map(config.camera_name_map, out_cam)
                    for out_cam in output_cam_names
                ],
                "language_task": str(anchor_frame.get("language_task", "")),
                "language_prompt": str(anchor_frame.get("language_prompt", "")),
                "episode_id": ep_dir.name,
                "control": _ep_meta.get("control", "leader"),
            }
        )
        total_frames += len(frames)
        print(f"  [{i}/{len(eps)}] {ep_dir.name}  ({len(frames)} frames)")
    skip_msg = f", {skipped} skipped" if skipped else ""
    print(
        f"Loaded {len(ep_contexts)} episodes{skip_msg}, {total_frames} frames total\n"
    )

    # ── Phase 2: global shuffle across all episodes ───────────────────────
    all_work: List[tuple] = [
        (ctx, t) for ctx in ep_contexts for t in range(ctx["n_frames"])
    ]
    random.shuffle(all_work)
    print(
        f"Processing {len(all_work)} anchors from {len(ep_contexts)} episodes "
        f"(globally shuffled, stride={config.stride})..."
    )

    ep_sample_counts: Dict[str, int] = {}

    work_args = [(ctx, t, config, output_cam_names) for ctx, t in all_work]
    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        pbar = tqdm(
            executor.map(_build_sample, work_args),
            total=len(work_args),
            unit="anchor",
            dynamic_ncols=True,
        )
        for result in pbar:
            if result["filtered"] == "padding":
                filtered_padding += 1
            elif result["filtered"] == "still":
                filtered_still += 1
            elif result["filtered"] == "nan":
                filtered_nan += 1
            else:
                writer.add(result["sample_files"])
                total_samples += 1
                episode_id = result["episode_id"]
                ep_sample_counts[episode_id] = ep_sample_counts.get(episode_id, 0) + 1
                stats_counter += 1
                pbar.set_postfix(
                    filtered=filtered_padding + filtered_still + filtered_nan,
                    shard=writer._shard_idx,
                )

                # ── accumulate stats (every stats_stride samples) ──────
                if stats_counter % config.stats_stride == 0:
                    lowdim = result["lowdim"]
                    for key, arr in lowdim.items():
                        if not isinstance(arr, np.ndarray):
                            continue
                        if arr.dtype == bool or arr.ndim < 2:
                            continue
                        if key.startswith("intrinsics.") or key.startswith(
                            "extrinsics."
                        ):
                            continue
                        sample_arr = arr.reshape(T, -1).astype(np.float32)
                        D = sample_arr.shape[1]
                        if key not in stats_accumulators:
                            stats_accumulators[key] = _StatsAccumulator(
                                T, D, config.stats_reservoir_size
                            )
                        stats_accumulators[key].update(sample_arr)

    print("\nSamples per episode:")
    for ep_id, count in sorted(ep_sample_counts.items()):
        print(f"  {ep_id}: {count}")

    writer.close()

    # ── write manifest.jsonl ─────────────────────────────────────────────
    (shard_dir / "manifest.jsonl").write_text("\n".join(writer.manifest_lines()) + "\n")

    # ── compute and write stats.json ─────────────────────────────────────
    print("\nComputing statistics...")
    stats: Dict[str, Any] = {
        key: acc.finalize() for key, acc in stats_accumulators.items()
    }
    for key, s in stats.items():
        count = s.get("count", 0)
        mn = s.get("min", [])
        mx = s.get("max", [])
        same = mn == mx if isinstance(mn, list) else False
        flag = "  [!] min==max" if same else ""
        print(f"  {key}: n={count}{flag}")
    with open(shard_dir / "stats.json", "w") as f:
        json.dump(stats, f)

    # ── write processing_metadata.json ───────────────────────────────────
    elapsed = time.time() - t_start
    proc_meta = {
        "metadata_version": "1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_data": {
            "episode_dirs": [str(d) for d in eps],
            "num_episodes": len(eps),
        },
        "config": dataclasses.asdict(config),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
        "processing": {
            "total_samples_created": total_samples,
            "padding_samples_filtered": filtered_padding,
            "still_samples_filtered": filtered_still,
            "nan_samples_filtered": filtered_nan,
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    # Make Path objects JSON-serialisable
    proc_meta["config"]["output_dir"] = str(config.output_dir)
    with open(shard_dir / "processing_metadata.json", "w") as f:
        json.dump(proc_meta, f, indent=2)

    print(
        f"\nDone: {total_samples} samples → {writer._shard_idx} shards  "
        f"filtered: {filtered_padding + filtered_still + filtered_nan} "
        f"(pad={filtered_padding} still={filtered_still} nan={filtered_nan})  "
        f"elapsed: {elapsed:.0f}s"
    )
    print(f"Output: {shard_dir}")

    # ── optional S3 upload ────────────────────────────────────────────────
    if s3_bucket and s3_prefix:
        upload_to_s3(shard_dir, s3_bucket, s3_prefix)
