# Fast Foundation Stereo

TensorRT-compiled engines run Fast Foundation Stereo significantly faster than
the PyTorch model. Once compiled, `rd convert` automatically detects and uses
the engines.

## Prerequisites

**1. Install TensorRT**

Add the TensorRT apt repository following the
[local repo installation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#installation-steps-local-repo-method),
then install:

```bash
sudo apt-get install --no-install-recommends libnvinfer-bin libnvinfer-dev libnvinfer-headers-dev
uv tool install --reinstall -e ".[zed,ffs,ffs-trt-cu12]"    # CUDA 12
uv tool install --reinstall -e ".[zed,ffs,ffs-trt-cu13]"    # CUDA 13
```

!!! note
    `sudo apt-get install tensorrt` fails on Python 3.11+ systems due to a
    dependency conflict. See [FAQ](faq.md#sudo-apt-get-install-tensorrt-fails-with-python-version-conflict)
    for details.

Verify both are available:

```bash
trtexec --help
uv run python -c "import tensorrt; print(tensorrt.__version__)"
```

**2. Install Fast Foundation Stereo**

Follow the [Fast Foundation Stereo installation](installation.md#fast-foundation-stereo-optional)
steps first (clone repo, sync extra, download weights).

## Step 1 - Export to ONNX

Export the pretrained model to ONNX format. The default input size (448×640)
is chosen to fit ZED HD720 images efficiently:

```bash
uv run python scripts/make_onnx.py
```

This writes `feature_runner.onnx`, `post_runner.onnx`, and `onnx.yaml` to
`~/.config/raiden/weights/onnx/`.

Options:

| Flag | Default | Description |
|---|---|---|
| `--model_dir` | most recent `~/.config/raiden/weights/*.pth` | Path to the `.pth` checkpoint |
| `--save_path` | `~/.config/raiden/weights/onnx/` | Output directory |
| `--height` | `448` | ONNX input height (must be divisible by 32) |
| `--width` | `640` | ONNX input width (must be divisible by 32) |
| `--valid_iters` | `8` | Update iterations during forward pass |
| `--max_disp` | `192` | Max disparity for the geometry encoding volume |
| `--build-engines` | off | Also compile TensorRT engines after export |

## Step 2 - Compile to TensorRT engines

Compile both ONNX models to TensorRT FP16 engines. This step runs once per
machine and takes a few minutes. Pass `--build-engines` to `make_onnx.py` to
do it in one shot:

```bash
uv run python scripts/make_onnx.py --build-engines
```

This exports the ONNX files and immediately compiles the engines using the
TensorRT Python API.

Alternatively, compile manually with `trtexec` (delete any existing `.engine`
files first to avoid stale-file issues):

```bash
rm -f ~/.config/raiden/weights/onnx/feature_runner.engine ~/.config/raiden/weights/onnx/post_runner.engine

trtexec --onnx=~/.config/raiden/weights/onnx/feature_runner.onnx \
        --saveEngine=~/.config/raiden/weights/onnx/feature_runner.engine \
        --fp16

trtexec --onnx=~/.config/raiden/weights/onnx/post_runner.onnx \
        --saveEngine=~/.config/raiden/weights/onnx/post_runner.engine \
        --fp16
```

## Step 3 - Use

No changes to your workflow are needed. When `rd convert` is run with
`--stereo-method ffs`, Raiden automatically detects the engine files and uses
TensorRT. If the engines are not present it falls back to PyTorch.

```bash
rd convert --stereo-method ffs
```

You will see `[FFS-TRT] Loaded engines from ~/.config/raiden/weights/onnx` in the output
when TensorRT is active.

!!! note "Engines are machine-specific"
    TensorRT engines are compiled for the specific GPU and driver version on
    the build machine. Recompile after upgrading drivers, CUDA, or TensorRT,
    or when moving to a different GPU.

## Citation

If you use Fast Foundation Stereo, please cite:

```bibtex
@article{wen2026fastfoundationstereo,
  title={{Fast-FoundationStereo}: Real-Time Zero-Shot Stereo Matching},
  author={Bowen Wen and Shaurya Dewan and Stan Birchfield},
  journal={CVPR},
  year={2026}
}
```
