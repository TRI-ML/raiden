# Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- hidapi system library (required for SpaceMouse support):

```bash
sudo apt install libhidapi-hidraw0 libhidapi-libusb0
```

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

Install `rd` as a shell command:

```bash
uv tool install .
rd --help
```

## Hardware SDKs

**ZED cameras** - install the [ZED SDK](https://www.stereolabs.com/developers/),
then run the helper script to download the matching Python wheel:

```bash
uv run python scripts/install_pyzed.py
uv tool install ".[zed]"
```

This downloads `pyzed-*.whl` into `packages/` and updates `pyproject.toml` to
reference it. `uv tool install ".[zed]"` then reinstalls `rd` with ZED support.

**SpaceMouse** - install a udev rule so the device is accessible without sudo:

```bash
sudo bash scripts/install_spacemouse_udev.sh
sudo usermod -aG plugdev $USER
```

Log out and back in, then replug the SpaceMouse.

**Intel RealSense** - `pyrealsense2` is included as a core dependency and installed
automatically. No additional SDK installation is required for most distributions.

## Fast Foundation Stereo (optional)

[Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo) is
an optional depth backend for ZED cameras that produces higher-quality depth
maps than the ZED SDK. It requires a CUDA GPU.

**1. Populate source and install:**

```bash
uv run python scripts/install_ffs.py
```

This clones the model source into `third_party/Fast-FoundationStereo`,
creates a packaging shim, and installs it into the active environment.

**2. Reinstall `rd` with the `ffs` extra (TensorRT / ONNX dependencies):**

```bash
uv tool install ".[ffs]"
```

**3. Download the pretrained weights:**

Download the checkpoint from the
[Fast Foundation Stereo repository](https://github.com/NVlabs/Fast-FoundationStereo)
(see the releases or the README for the link) and place the `*.pth` file in
`~/.config/raiden/weights/`. Raiden picks the most recently modified checkpoint in that
directory automatically.

After installation, pass `--stereo-method ffs` to `rd convert`
to use Fast Foundation Stereo depth. See [Conversion](conversion.md#depth-backends-for-zed-cameras) for details.

For faster inference, compile the model to TensorRT engines. See
[TensorRT Acceleration](tensorrt.md) for the full guide.
