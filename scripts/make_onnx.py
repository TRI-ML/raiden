#!/usr/bin/env python3
"""Export Fast Foundation Stereo to ONNX for ZED HD720 stereo images.

Outputs feature_runner.onnx, post_runner.onnx, and onnx.yaml to
~/.config/raiden/weights/onnx/ (or --save_path).

After running this, compile to TensorRT engines.  Pass --build-engines to do
this automatically via the TensorRT Python API (recommended):

    uv run python scripts/make_onnx.py --build-engines

Or compile manually with trtexec (delete any existing .engine files first):

    trtexec --onnx=~/.config/raiden/weights/onnx/feature_runner.onnx \\
            --saveEngine=~/.config/raiden/weights/onnx/feature_runner.engine --fp16

    trtexec --onnx=~/.config/raiden/weights/onnx/post_runner.onnx \\
            --saveEngine=~/.config/raiden/weights/onnx/post_runner.engine --fp16

Once the .engine files are present, rd convert will automatically
use TensorRT instead of PyTorch for Fast Foundation Stereo inference.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

_RAIDEN_DIR = Path(__file__).parent.parent
_FFS_DIR = _RAIDEN_DIR / "third_party" / "Fast-FoundationStereo"
_WEIGHTS_DIR = Path.home() / ".config" / "raiden" / "weights"

sys.path.insert(0, str(_FFS_DIR))


def _build_trt_engines(save_path: Path) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    for name in ("feature_runner", "post_runner"):
        onnx_path = save_path / f"{name}.onnx"
        engine_path = save_path / f"{name}.engine"
        print(f"\nBuilding {name}.engine (may take several minutes)...")
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            ok = parser.parse(f.read())
        if not ok:
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse {onnx_path}")
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        engine_mem = builder.build_serialized_network(network, config)
        if engine_mem is None:
            raise RuntimeError(f"TensorRT failed to build engine for {name}")
        engine_bytes = bytes(engine_mem)
        print(f"  Engine size: {len(engine_bytes) / 1024 / 1024:.1f} MiB")
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"  Saved {engine_path}")

    print("\nTensorRT engines ready. rd convert will use them automatically.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Path to .pth checkpoint (default: most recently modified in ~/.config/raiden/weights/)",
    )
    parser.add_argument(
        "--save_path",
        default=str(_WEIGHTS_DIR / "onnx"),
        help="Directory to write ONNX files and onnx.yaml (default: ~/.config/raiden/weights/onnx/)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=448,
        help="ONNX input height in pixels, must be divisible by 32 (default: 448)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="ONNX input width in pixels, must be divisible by 32 (default: 640)",
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=8,
        help="Number of update iterations during forward pass (default: 8)",
    )
    parser.add_argument(
        "--max_disp",
        type=int,
        default=192,
        help="Max disparity for the geometry encoding volume (default: 192)",
    )
    parser.add_argument(
        "--build-engines",
        action="store_true",
        help="After exporting ONNX, compile TensorRT engines via the Python API",
    )
    args = parser.parse_args()

    assert args.height % 32 == 0 and args.width % 32 == 0, (
        f"height ({args.height}) and width ({args.width}) must both be divisible by 32"
    )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if args.model_dir is None:
        ckpts = sorted(
            _WEIGHTS_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not ckpts:
            raise RuntimeError(f"No *.pth checkpoint found in '{_WEIGHTS_DIR}'")
        args.model_dir = str(ckpts[0])

    print(f"Checkpoint : {args.model_dir}")
    print(f"Input size : {args.height} x {args.width}")
    print(f"Output dir : {save_path}")

    import torch
    import yaml
    from omegaconf import OmegaConf
    from core.foundation_stereo import (
        TrtFeatureRunner,
        TrtPostRunner,
        build_gwc_volume_triton,
    )

    torch.autograd.set_grad_enabled(False)

    print("\nLoading model...")
    model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.cuda().eval()

    feature_runner = TrtFeatureRunner(model).cuda().eval()
    post_runner = TrtPostRunner(model).cuda().eval()

    left_img = torch.randn(1, 3, args.height, args.width).cuda().float() * 255
    right_img = torch.randn(1, 3, args.height, args.width).cuda().float() * 255

    import onnx

    print("\nExporting feature_runner.onnx ...")
    feature_onnx_path = save_path / "feature_runner.onnx"
    torch.onnx.export(
        feature_runner,
        (left_img, right_img),
        str(feature_onnx_path),
        opset_version=17,
        input_names=["left", "right"],
        output_names=[
            "features_left_04",
            "features_left_08",
            "features_left_16",
            "features_left_32",
            "features_right_04",
            "stem_2x",
        ],
        do_constant_folding=True,
        # Use the dynamo exporter so torchvision ops (roi_align) are decomposed
        # into standard ONNX ops that TensorRT can compile.
    )
    # The dynamo exporter may write weights to a separate .onnx.data file.
    # trtexec requires a single self-contained file, so merge if needed.
    _data_file = feature_onnx_path.with_suffix(".onnx.data")
    if _data_file.exists():
        print("  Merging external data into single file ...")
        _m = onnx.load(str(feature_onnx_path))
        onnx.save(_m, str(feature_onnx_path))
        _data_file.unlink(missing_ok=True)

    print("Exporting post_runner.onnx ...")
    (
        features_left_04,
        features_left_08,
        features_left_16,
        features_left_32,
        features_right_04,
        stem_2x,
    ) = feature_runner(left_img, right_img)
    gwc_volume = build_gwc_volume_triton(
        features_left_04.half(),
        features_right_04.half(),
        args.max_disp // 4,
        model.cv_group,
    )
    torch.onnx.export(
        post_runner,
        (
            features_left_04,
            features_left_08,
            features_left_16,
            features_left_32,
            features_right_04,
            stem_2x,
            gwc_volume,
        ),
        str(save_path / "post_runner.onnx"),
        opset_version=17,
        input_names=[
            "features_left_04",
            "features_left_08",
            "features_left_16",
            "features_left_32",
            "features_right_04",
            "stem_2x",
            "gwc_volume",
        ],
        output_names=["disp"],
        do_constant_folding=True,
        dynamo=False,
    )

    # Save model config + image dimensions for inference.
    cfg = OmegaConf.to_container(model.args)
    cfg["image_h"] = args.height
    cfg["image_w"] = args.width
    cfg["cv_group"] = model.cv_group
    with open(save_path / "onnx.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"\nDone. Files written to {save_path}/")

    if args.build_engines:
        _build_trt_engines(save_path)
    else:
        print("\nNext - compile to TensorRT engines (run once per machine):")
        print("  uv run python scripts/make_onnx.py --build-engines")
        print("Or manually with trtexec (delete existing .engine files first):")
        print(
            f"  trtexec --onnx={save_path}/feature_runner.onnx"
            f" --saveEngine={save_path}/feature_runner.engine --fp16"
        )
        print(
            f"  trtexec --onnx={save_path}/post_runner.onnx"
            f" --saveEngine={save_path}/post_runner.engine --fp16"
        )


if __name__ == "__main__":
    main()
