"""Fast Foundation Stereo (FFS) depth estimation wrapper.

Wraps the NVlabs/Fast-FoundationStereo model as an alternative stereo depth
backend for ZED cameras in ``rd convert``.

Installation
------------
Run the install script once to clone FFS and make it pip-installable::

    uv run python scripts/install_ffs.py [--download-weights]

Then activate the optional extra::

    uv sync --extra ffs

Checkpoint
----------
The install script downloads ``model_best_bp2_serialize.pth`` when called with
``--download-weights``, or you can place any ``*.pth`` checkpoint in
``~/.config/raiden/weights/`` manually.
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from raiden._config import WEIGHTS_DIR as _WEIGHTS_DIR_CFG


# Path to the FFS clone (set by install_ffs.py).
_FFS_DIR = Path(__file__).parent.parent / "third_party" / "Fast-FoundationStereo"

# Directory where pretrained checkpoints and configs are stored.
_WEIGHTS_DIR = _WEIGHTS_DIR_CFG

# Directory where ONNX files and TensorRT engines are stored.
_ONNX_DIR = _WEIGHTS_DIR / "onnx"


class FFSDepthPredictor:
    """Depth predictor backed by Fast Foundation Stereo.

    Parameters
    ----------
    scale : float
        Input resize factor (default 1.0). Use 0.5 to halve the resolution
        for faster inference at the cost of depth precision.
    iters : int
        Number of update iterations (default 8, range 4–32).
    device : str
        PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        scale: float = 1.0,
        iters: int = 8,
        device: str = "cuda",
    ) -> None:
        self._scale = scale
        self._iters = iters
        self._device = device
        self._model = None
        self._InputPadder = None
        # Timing accumulators for profiling.
        self._t_inference: float = 0.0
        self._n_calls: int = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch

        try:
            from core.utils.utils import InputPadder  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "foundation-stereo is not installed. "
                "Run: uv run python scripts/install_ffs.py && uv sync --extra ffs"
            ) from exc

        # The FFS checkpoint is a fully serialized model (torch.save(model)),
        # not a state dict — load it directly.
        ckpts = sorted(
            _WEIGHTS_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not ckpts:
            raise RuntimeError(
                f"No *.pth checkpoint found in '{_WEIGHTS_DIR}'. "
                "Run: uv run python scripts/install_ffs.py --download-weights"
            )
        ckpt_path = ckpts[0]

        # Free any residual GPU allocations (e.g. from ZED SDK) before loading.
        if self._device != "cpu":
            torch.cuda.empty_cache()

        model = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = model.to(self._device)
        model.eval()
        self._model = model
        self._InputPadder = InputPadder
        print(f"[FFS] Loaded model: {ckpt_path.name}")

    def predict(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        fx: float,
        baseline: float,
    ) -> np.ndarray:
        """Run Fast Foundation Stereo and return a depth map in metres.

        Parameters
        ----------
        left_bgr : np.ndarray
            Left camera image (H, W, 3), BGR uint8.
        right_bgr : np.ndarray
            Right camera image (H, W, 3), BGR uint8.
        fx : float
            Left-camera focal length in pixels (at the *original* resolution).
        baseline : float
            Stereo baseline in metres.

        Returns
        -------
        np.ndarray
            float32 (H, W) depth map in metres. 0 = invalid.
        """
        import torch

        self._ensure_loaded()

        H, W = left_bgr.shape[:2]
        sH = int(H * self._scale)
        sW = int(W * self._scale)

        def to_tensor(bgr: np.ndarray) -> "torch.Tensor":
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            if self._scale != 1.0:
                rgb = cv2.resize(rgb, (sW, sH), interpolation=cv2.INTER_LINEAR)
            return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)

        left_t = to_tensor(left_bgr)
        right_t = to_tensor(right_bgr)

        padder = self._InputPadder(left_t.shape, divis_by=32, force_square=False)
        left_t, right_t = padder.pad(left_t, right_t)

        amp_enabled = self._device != "cpu"
        t0 = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16),
        ):
            disp = self._model.forward(
                left_t,
                right_t,
                iters=self._iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
        if self._device != "cpu":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        disp = padder.unpad(disp.float())
        disp_np = (
            disp.data.cpu().numpy().reshape(sH, sW).clip(0, None).astype(np.float32)
        )
        del disp, left_t, right_t
        if self._device != "cpu":
            torch.cuda.empty_cache()

        # remove_invisible: invalidate pixels where the right-image projection
        # goes out of frame (occluded regions). Mirrors FFS run_demo.py.
        xx = np.arange(sW, dtype=np.float32)[None, :]  # (1, sW)
        us_right = xx - disp_np
        disp_np[us_right < 0] = 0.0

        # Upsample disparity back to original resolution if input was scaled.
        if self._scale != 1.0:
            disp_np = cv2.resize(disp_np, (W, H), interpolation=cv2.INTER_LINEAR)

        # depth = (fx * scale) * baseline / disp_scaled
        fx_eff = fx * self._scale
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(disp_np > 0.5, fx_eff * baseline / disp_np, 0.0).astype(
                np.float32
            )

        self._t_inference += t1 - t0
        self._n_calls += 1

        return depth

    def timing_summary(self) -> str:
        """Return a one-line timing summary string."""
        if self._n_calls == 0:
            return "no calls yet"
        avg_inf = self._t_inference / self._n_calls * 1000
        return f"inference={avg_inf:.0f}ms  (avg over {self._n_calls} calls)"


class FFSTrtDepthPredictor:
    """TensorRT-accelerated Fast Foundation Stereo depth predictor.

    Uses pre-compiled TensorRT engines for faster inference than the PyTorch
    path. Requires engine files produced by scripts/make_onnx.py and compiled
    with trtexec::

        trtexec --onnx=~/.config/raiden/weights/onnx/feature_runner.onnx \\
                --saveEngine=~/.config/raiden/weights/onnx/feature_runner.engine --fp16
        trtexec --onnx=~/.config/raiden/weights/onnx/post_runner.onnx \\
                --saveEngine=~/.config/raiden/weights/onnx/post_runner.engine --fp16

    Parameters
    ----------
    onnx_dir : str, optional
        Directory containing feature_runner.engine, post_runner.engine, and
        onnx.yaml. Defaults to ~/.config/raiden/weights/onnx/.
    """

    def __init__(self, onnx_dir: Optional[str] = None) -> None:
        self._onnx_dir = Path(onnx_dir) if onnx_dir is not None else _ONNX_DIR
        self._runner = None
        self._image_h: int = 448
        self._image_w: int = 640
        self._t_inference: float = 0.0
        self._n_calls: int = 0

    @staticmethod
    def engines_available(onnx_dir: Optional[str] = None) -> bool:
        """Return True if TensorRT engine files are present."""
        d = Path(onnx_dir) if onnx_dir is not None else _ONNX_DIR
        return (d / "feature_runner.engine").exists() and (
            d / "post_runner.engine"
        ).exists()

    def _ensure_loaded(self) -> None:
        if self._runner is not None:
            return

        import sys
        import yaml
        from omegaconf import OmegaConf

        sys.path.insert(0, str(_FFS_DIR))
        from core.foundation_stereo import TrtRunner  # noqa: PLC0415

        feature_engine = self._onnx_dir / "feature_runner.engine"
        post_engine = self._onnx_dir / "post_runner.engine"
        if not feature_engine.exists() or not post_engine.exists():
            raise RuntimeError(
                f"TensorRT engines not found in '{self._onnx_dir}'. "
                "Run: python scripts/make_onnx.py, then compile with trtexec."
            )

        with open(self._onnx_dir / "onnx.yaml") as f:
            cfg = yaml.safe_load(f)

        self._image_h = cfg.get("image_h", 448)
        self._image_w = cfg.get("image_w", 640)

        args = OmegaConf.create(cfg)
        self._runner = TrtRunner(args, str(feature_engine), str(post_engine))
        print(
            f"[FFS-TRT] Loaded engines from {self._onnx_dir}"
            f" (input: {self._image_h}x{self._image_w})"
        )

    def predict(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        fx: float,
        baseline: float,
    ) -> np.ndarray:
        """Run TRT inference and return a depth map in metres.

        Parameters
        ----------
        left_bgr : np.ndarray
            Left camera image (H, W, 3), BGR uint8.
        right_bgr : np.ndarray
            Right camera image (H, W, 3), BGR uint8.
        fx : float
            Left-camera focal length in pixels at the *original* resolution.
        baseline : float
            Stereo baseline in metres.

        Returns
        -------
        np.ndarray
            float32 (H, W) depth map in metres at the original resolution.
            0 = invalid.
        """
        import torch

        self._ensure_loaded()

        H, W = left_bgr.shape[:2]
        sx = self._image_w / W  # horizontal scale factor

        def to_tensor(bgr: np.ndarray) -> "torch.Tensor":
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = cv2.resize(
                rgb, (self._image_w, self._image_h), interpolation=cv2.INTER_LINEAR
            )
            return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda()

        left_t = to_tensor(left_bgr)
        right_t = to_tensor(right_bgr)

        t0 = time.perf_counter()
        disp = self._runner.forward(left_t, right_t)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        disp_np = (
            disp.cpu()
            .numpy()
            .reshape(self._image_h, self._image_w)
            .clip(0, None)
            .astype(np.float32)
        )
        del disp, left_t, right_t

        # Invalidate pixels whose right-image projection is out of frame.
        xx = np.arange(self._image_w, dtype=np.float32)[None, :]
        disp_np[xx - disp_np < 0] = 0.0

        # Upsample disparity to original resolution.
        # Values are still in scaled-resolution pixel units, so divide by sx
        # to convert to original-resolution pixel units before computing depth.
        disp_full = cv2.resize(disp_np, (W, H), interpolation=cv2.INTER_LINEAR)
        disp_full /= sx

        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(disp_full > 0.5, fx * baseline / disp_full, 0.0).astype(
                np.float32
            )

        self._t_inference += t1 - t0
        self._n_calls += 1
        return depth

    def timing_summary(self) -> str:
        """Return a one-line timing summary string."""
        if self._n_calls == 0:
            return "no calls yet"
        avg_inf = self._t_inference / self._n_calls * 1000
        return f"inference={avg_inf:.0f}ms  (avg over {self._n_calls} calls)"
