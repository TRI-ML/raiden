# Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Install

Clone the repository with submodules and sync dependencies with `uv`:

```bash
git clone --recurse-submodules git@github.com:TRI-ML/raiden.git
cd raiden
uv sync
```

If you already cloned without `--recurse-submodules`, initialize the submodule manually:

```bash
git submodule update --init
```

Install `rd` as a shell command:

```bash
uv tool install -e .                                                    # base install
uv tool install -e ".[zed]"                                             # + ZED cameras
uv tool install -e ".[zed,tri-stereo]"                                  # + TRI Stereo depth (ONNX)
uv tool install -e ".[zed,tri-stereo,tri-stereo-trt-cu12]"              # + TensorRT (CUDA 12)
uv tool install -e ".[zed,tri-stereo,tri-stereo-trt-cu13]"              # + TensorRT (CUDA 13)
rd --help
```

!!! note
    Re-run `uv tool install --reinstall -e ".[<extras>]"` whenever you add
    or change extras, or after pulling updates from the repository.

## Hardware SDKs

**ZED cameras** - install the [ZED SDK](https://www.stereolabs.com/developers/),
then run the helper script to download the matching Python wheel:

```bash
uv run python scripts/install_pyzed.py
uv sync --extra zed
```

This downloads `pyzed-*.whl` into `packages/` and updates `pyproject.toml` to
reference it. `uv sync --extra zed` then installs it into the project environment.

**Intel RealSense** - `pyrealsense2` is included as a core dependency and will be
installed by `uv sync`. No additional SDK installation is required for most
distributions.

## Optional depth backends

- **[TRI Stereo Depth](tri_stereo.md)** — TRI's learned stereo depth model tailored for robot manipulation scenes. Supports `c32` and `c64` variants with ONNX and TensorRT backends.
- **[Fast Foundation Stereo](tensorrt.md)** — foundation model stereo depth; higher quality at object boundaries and thin structures.
