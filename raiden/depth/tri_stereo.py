"""TRI Stereo depth estimation wrapper.

Wraps TRI's learned stereo depth model as an alternative stereo depth
backend for ZED cameras in ``rd convert``.

Two variants are supported: ``c32`` (lighter) and ``c64`` (higher quality).
Two inference backends are supported in order of preference:

1. **TensorRT** — fastest; requires a pre-compiled ``.engine`` file.
2. **ONNX Runtime** — no compilation needed; requires a ``.onnx`` file.

Model files
-----------
Model files are searched in order:

1. ``<repo>/weights/tri_stereo/`` — tracked via git-lfs (canonical)
2. ``~/.config/raiden/weights/tri_stereo/`` — user config fallback

=============  ===========================
Backend        Filename
=============  ===========================
ONNX c32       ``stereo_c32.onnx``
ONNX c64       ``stereo_c64.onnx``
TensorRT c32   ``stereo_c32.engine``
TensorRT c64   ``stereo_c64.engine``
=============  ===========================
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Search order for model files: repo weights/ (git-lfs) first, then ~/.config/raiden.
_SEARCH_DIRS = [
    Path(__file__).parent.parent.parent / "weights" / "tri_stereo",
    Path.home() / ".config" / "raiden" / "weights" / "tri_stereo",
]


def _resolve_checkpoint(variant: str, ext: str) -> Path:
    """Return the path for a given variant and file extension.

    Searches ``weights/tri_stereo/`` in the repo first, then
    ``~/.config/raiden/weights/tri_stereo/``.  Returns the first path that
    exists, or the repo path (as a canonical default) if neither does.

    Parameters
    ----------
    variant : str
        ``"c32"`` or ``"c64"``.
    ext : str
        File extension without leading dot, e.g. ``"onnx"`` or ``"engine"``.
    """
    filename = f"stereo_{variant}.{ext}"
    for d in _SEARCH_DIRS:
        p = d / filename
        if p.exists():
            return p
    return _SEARCH_DIRS[0] / filename


def _disp_to_depth(
    disp: np.ndarray,
    fx: float,
    baseline: float,
) -> np.ndarray:
    """Convert a disparity map (pixels) to a depth map (meters).

    Parameters
    ----------
    disp : np.ndarray
        Disparity in pixels at the *model input* resolution.
    fx : float
        Focal length in pixels at the *model input* resolution.
    baseline : float
        Stereo baseline in meters.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disp > 1e-3, fx * baseline / disp, 0.0).astype(np.float32)
    return depth


# ---------------------------------------------------------------------------
# ONNX Runtime backend
# ---------------------------------------------------------------------------


class TRIStereoOnnxDepthPredictor:
    """Depth predictor backed by the TRI Stereo model (ONNX Runtime).

    The ONNX model has a fixed input resolution baked in at export time.
    Input images are resized to that resolution before inference.

    Parameters
    ----------
    variant : str
        Model variant: ``"c32"`` or ``"c64"``.
    onnx_path : str or None
        Path to the ``.onnx`` file.  Defaults to
        ``~/.config/raiden/weights/tri_stereo/stereo_{variant}.onnx``.
    use_cuda : bool
        Use CUDA execution provider when available (default True).
    """

    def __init__(
        self,
        variant: str = "c64",
        onnx_path: Optional[str] = None,
        use_cuda: bool = True,
    ) -> None:
        self._variant = variant
        self._onnx_path = (
            Path(onnx_path) if onnx_path else _resolve_checkpoint(variant, "onnx")
        )
        self._use_cuda = use_cuda
        self._session = None
        self._model_h: int = 0
        self._model_w: int = 0
        self._t_inference: float = 0.0
        self._n_calls: int = 0

    @staticmethod
    def model_available(variant: str = "c64", onnx_path: Optional[str] = None) -> bool:
        """Return True if the ONNX model file exists."""
        p = Path(onnx_path) if onnx_path else _resolve_checkpoint(variant, "onnx")
        return p.exists()

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        # Import torch first so its bundled cuDNN is loaded into the process
        # before onnxruntime tries to find libcudnn.so.9 via dlopen.
        import torch  # noqa: PLC0415, F401

        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed.  Run: pip install onnxruntime-gpu"
            ) from exc

        if not self._onnx_path.exists():
            raise RuntimeError(
                f"TRI Stereo ONNX model not found: {self._onnx_path}. Run: git lfs pull"
            )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._use_cuda
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(self._onnx_path), providers=providers)

        # Read the fixed input shape from the model.
        shape = session.get_inputs()[0].shape  # [1, 3, H, W]
        self._model_h = int(shape[2])
        self._model_w = int(shape[3])
        self._session = session

        active_provider = session.get_providers()[0]
        print(
            f"[TRIStereo-{self._variant.upper()}] Loaded ONNX model: {self._onnx_path.name}"
            f" (input: {self._model_h}x{self._model_w}, provider: {active_provider})"
        )

    def predict(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        fx: float,
        baseline: float,
    ) -> np.ndarray:
        """Run TRI Stereo ONNX inference and return a depth map in meters.

        Parameters
        ----------
        left_bgr, right_bgr : np.ndarray
            BGR uint8 images (H, W, 3).
        fx : float
            Left-camera focal length in pixels at the *original* resolution.
        baseline : float
            Stereo baseline in meters.

        Returns
        -------
        np.ndarray
            float32 (H, W) depth map in meters.  0 = invalid.
        """
        self._ensure_loaded()

        H, W = left_bgr.shape[:2]
        mH, mW = self._model_h, self._model_w

        def prepare(bgr: np.ndarray) -> np.ndarray:
            resized = cv2.resize(bgr, (mW, mH), interpolation=cv2.INTER_LINEAR)
            return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        left_np = prepare(left_bgr)
        right_np = prepare(right_bgr)

        t0 = time.perf_counter()
        disparity, _disp_sparse, _confidence = self._session.run(
            None, {"left_image": left_np, "right_image": right_np}
        )
        t1 = time.perf_counter()

        # disparity shape: (1, 1, mH, mW)
        disp_np = disparity.reshape(mH, mW).clip(0, None)

        # Focal length at model resolution.
        fx_scaled = fx * (mW / W)
        depth_model = _disp_to_depth(disp_np, fx_scaled, baseline)

        # Upsample to original resolution.
        if (mH, mW) != (H, W):
            depth = cv2.resize(depth_model, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            depth = depth_model

        self._t_inference += t1 - t0
        self._n_calls += 1
        return depth

    def timing_summary(self) -> str:
        if self._n_calls == 0:
            return "no calls yet"
        avg_ms = self._t_inference / self._n_calls * 1000
        return f"inference={avg_ms:.0f}ms  (avg over {self._n_calls} calls)"


# ---------------------------------------------------------------------------
# TensorRT backend
# ---------------------------------------------------------------------------


class TRIStereoTrtDepthPredictor:
    """Depth predictor backed by the TRI Stereo model (TensorRT).

    Requires a pre-compiled TensorRT ``.engine`` file.  The engine is compiled
    for a fixed input resolution.

    Parameters
    ----------
    variant : str
        Model variant: ``"c32"`` or ``"c64"``.
    engine_path : str or None
        Path to the ``.engine`` file.  Defaults to
        ``~/.config/raiden/weights/tri_stereo/stereo_{variant}.engine``.
    """

    def __init__(
        self,
        variant: str = "c64",
        engine_path: Optional[str] = None,
    ) -> None:
        self._variant = variant
        self._engine_path = (
            Path(engine_path) if engine_path else _resolve_checkpoint(variant, "engine")
        )
        self._context = None
        self._model_h: int = 0
        self._model_w: int = 0
        self._t_inference: float = 0.0
        self._n_calls: int = 0

    @staticmethod
    def engine_available(
        variant: str = "c64", engine_path: Optional[str] = None
    ) -> bool:
        """Return True if the TensorRT engine file exists."""
        p = Path(engine_path) if engine_path else _resolve_checkpoint(variant, "engine")
        return p.exists()

    def _ensure_loaded(self) -> None:
        if self._context is not None:
            return

        try:
            import tensorrt as trt  # noqa: PLC0415
            import torch  # noqa: PLC0415, F401
        except ImportError as exc:
            raise RuntimeError(
                "tensorrt or torch is not installed. "
                "See docs/guide/tensorrt.md for setup instructions."
            ) from exc

        if not self._engine_path.exists():
            raise RuntimeError(
                f"TRI Stereo TensorRT engine not found: {self._engine_path}. "
                "Compile it with trtexec after exporting the ONNX model."
            )

        trt_logger = trt.Logger(trt.Logger.ERROR)
        trt_runtime = trt.Runtime(trt_logger)
        with open(self._engine_path, "rb") as f:
            engine = trt_runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError(
                f"Failed to deserialize TRI Stereo TRT engine: {self._engine_path}. "
                "The engine was likely compiled with a different TensorRT version. "
                "Recompile it with: uv run python scripts/tri_stereo_tensorrt_build.py"
            )

        # Infer input H×W from tensor dimensions.
        context = engine.create_execution_context()
        # Tensor 0 is the left image: (1, 3, H, W)
        input_name = engine.get_tensor_name(0)
        binding_shape = engine.get_tensor_shape(input_name)
        self._model_h = int(binding_shape[2])
        self._model_w = int(binding_shape[3])
        self._engine = engine
        self._context = context

        print(
            f"[TRIStereo-{self._variant.upper()}] Loaded TRT engine: {self._engine_path.name}"
            f" (input: {self._model_h}x{self._model_w})"
        )

    def predict(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        fx: float,
        baseline: float,
    ) -> np.ndarray:
        """Run TRI Stereo TensorRT inference and return a depth map in meters.

        Parameters
        ----------
        left_bgr, right_bgr : np.ndarray
            BGR uint8 images (H, W, 3).
        fx : float
            Left-camera focal length in pixels at the *original* resolution.
        baseline : float
            Stereo baseline in meters.

        Returns
        -------
        np.ndarray
            float32 (H, W) depth map in meters.  0 = invalid.
        """
        import torch

        self._ensure_loaded()

        H, W = left_bgr.shape[:2]
        mH, mW = self._model_h, self._model_w

        def prepare(bgr: np.ndarray) -> "torch.Tensor":
            resized = cv2.resize(bgr, (mW, mH), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(resized.astype(np.float32) / 255.0)
            return t.permute(2, 0, 1).unsqueeze(0).cuda().contiguous()

        left_t = prepare(left_bgr)
        right_t = prepare(right_bgr)

        disparity = torch.zeros((1, 1, mH, mW), dtype=torch.float32, device="cuda")
        disparity_sparse = torch.zeros(
            (1, 1, mH, mW), dtype=torch.float32, device="cuda"
        )
        confidence = torch.zeros(
            (1, 1, mH // 4, mW // 4), dtype=torch.float32, device="cuda"
        )

        trt_buffers = [
            left_t.data_ptr(),
            right_t.data_ptr(),
            disparity.data_ptr(),
            disparity_sparse.data_ptr(),
            confidence.data_ptr(),
        ]

        t0 = time.perf_counter()
        self._context.execute_v2(trt_buffers)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        disp_np = disparity.cpu().numpy().reshape(mH, mW).clip(0, None)
        del left_t, right_t, disparity, disparity_sparse, confidence

        fx_scaled = fx * (mW / W)
        depth_model = _disp_to_depth(disp_np, fx_scaled, baseline)

        if (mH, mW) != (H, W):
            depth = cv2.resize(depth_model, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            depth = depth_model

        self._t_inference += t1 - t0
        self._n_calls += 1
        return depth

    def timing_summary(self) -> str:
        if self._n_calls == 0:
            return "no calls yet"
        avg_ms = self._t_inference / self._n_calls * 1000
        return f"inference={avg_ms:.0f}ms  (avg over {self._n_calls} calls)"
