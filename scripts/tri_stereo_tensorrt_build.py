# Copyright 2024 Toyota Research Institute.  All rights reserved.

import argparse
from pathlib import Path

import tensorrt as trt

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights" / "tri_stereo"


def main(args):
    onnx_path = str(Path(args.onnx).expanduser().resolve()) if args.onnx else None
    engine_path = str(Path(args.engine).expanduser().resolve()) if args.engine else None

    trt_logger = trt.Logger(trt.Logger.ERROR)
    trt_builder = trt.Builder(trt_logger)
    trt_network = trt_builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    trt_parser = trt.OnnxParser(trt_network, trt_logger)
    trt_parse_success = trt_parser.parse_from_file(onnx_path)
    assert trt_parse_success, f"Failed to parse ONNX model: {onnx_path}"

    trt_config = trt_builder.create_builder_config()
    if args.fp16:
        trt_config.set_flag(trt.BuilderFlag.FP16)
    trt_serialized_engine_out = trt_builder.build_serialized_network(
        trt_network, trt_config
    )
    with open(engine_path, "wb") as out_file:
        out_file.write(trt_serialized_engine_out)
    print(f"Saved TensorRT engine to {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["c32", "c64"],
        default="c64",
        help="Model variant (default: c64)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Input .onnx file (default: weights/tri_stereo/stereo_<variant>.onnx)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Output .engine file (default: weights/tri_stereo/stereo_<variant>.engine)",
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")

    args = parser.parse_args()

    if args.onnx is None:
        args.onnx = str(_WEIGHTS_DIR / f"stereo_{args.variant}.onnx")
    if args.engine is None:
        args.engine = str(_WEIGHTS_DIR / f"stereo_{args.variant}.engine")

    main(args)
