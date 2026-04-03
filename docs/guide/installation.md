# Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ZED SDK](https://www.stereolabs.com/developers/) (if using ZED cameras)

## Install

Clone the repository with submodules:

```bash
git clone --recurse-submodules git@github.com:TRI-ML/raiden.git
cd raiden
```

If you already cloned without `--recurse-submodules`, initialize the submodule manually:

```bash
git submodule update --init
```

## Hardware SDKs

**ZED cameras** - after installing the ZED SDK, run the helper script to download
the matching Python wheel:

```bash
uv run python scripts/install_pyzed.py
```

This downloads `pyzed-*.whl` into `packages/` and updates `pyproject.toml` to
reference it.

**Intel RealSense** (not recommended) - no additional SDK installation is required
for most distributions; `pyrealsense2` is installed via the `realsense` extra.

## Install `rd`

Install `rd` as a shell command, picking the extras that match your hardware and
depth backend:

```bash
uv tool install -e .                                                    # base install
uv tool install -e ".[zed]"                                             # + ZED cameras
uv tool install -e ".[zed,tri-stereo]"                                  # + TRI Stereo depth (ONNX)
uv tool install -e ".[zed,tri-stereo,tri-stereo-trt-cu12]"              # + TensorRT (CUDA 12)
uv tool install -e ".[zed,tri-stereo,tri-stereo-trt-cu13]"              # + TensorRT (CUDA 13)
uv tool install -e ".[realsense]"                                       # + RealSense cameras (not recommended)
rd --help
```

!!! note
    Re-run `uv tool install --reinstall -e ".[<extras>]"` whenever you add
    or change extras, or after pulling updates from the repository.

## Optional depth backends

- **[TRI Stereo Depth](tri_stereo.md)** — TRI's learned stereo depth model tailored for robot manipulation scenes. Supports `c32` and `c64` variants with ONNX and TensorRT backends.
- **[Fast Foundation Stereo](tensorrt.md)** — foundation model stereo depth; higher quality at object boundaries and thin structures.
