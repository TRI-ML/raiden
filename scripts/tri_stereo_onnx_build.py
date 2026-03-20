# Copyright 2024 Toyota Research Institute.  All rights reserved.

import argparse
import io
import sys
from pathlib import Path
from typing import Tuple
import warnings

# Add MMT Python source to path (requires third_party/mmt_stereo_inference).
_MMT_PYTHON_DIR = (
    Path(__file__).parent.parent
    / "third_party"
    / "mmt_stereo_inference"
    / "mmt"
    / "stereo_inference"
    / "python"
)
if not _MMT_PYTHON_DIR.exists():
    raise SystemExit(
        f"MMT source not found at {_MMT_PYTHON_DIR}.\n"
        "Clone the MMT repository into third_party/ first:\n"
        "  uv run python scripts/install_tri_stereo.py"
    )
sys.path.insert(0, str(_MMT_PYTHON_DIR))

import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from stereo_utils import (  # noqa: E402
    add_disparity_args,
    add_post_processing_args,
    get_settings,
    load_checkpoint,
)

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights" / "tri_stereo"


class ExportableStereo(nn.Module):
    def __init__(self, args, model: nn.Module):
        super().__init__()

        self.settings = get_settings(args)
        self.model = model

    def forward(
        self, left_image: torch.Tensor, right_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(left_image, right_image, settings=self.settings)
        disparity = output["disparity"]
        disparity_sparse = output["disparity_sparse"]
        confidence = output["confidence"]
        return disparity, disparity_sparse, confidence


def main(args):
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else _WEIGHTS_DIR / f"stereo_{args.variant}.pth"
    )
    onnx_out = (
        Path(args.onnx) if args.onnx else _WEIGHTS_DIR / f"stereo_{args.variant}.onnx"
    )

    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(args, str(checkpoint))
    model.eval()

    exportable_model = ExportableStereo(args, model).eval()

    dummy_input = (
        torch.zeros((1, 3, args.height, args.width), dtype=torch.float32),
        torch.zeros((1, 3, args.height, args.width), dtype=torch.float32),
    )

    input_names = ["left_image", "right_image"]
    output_names = ["disparity", "disparity_sparse", "confidence"]

    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="Constant folding - Only steps=1",
            category=UserWarning,
        )
        torch.onnx.export(
            exportable_model,
            dummy_input,
            buf,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            opset_version=18,
            dynamo=False,
        )
    buf.seek(0)
    model_proto = onnx.load(buf)
    onnx.save_model(model_proto, str(onnx_out), save_as_external_data=False)
    print(f"Saved ONNX model to {onnx_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["c32", "c64"],
        default="c64",
        help="Model variant (default: c64)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pth file (default: weights/tri_stereo/stereo_<variant>.pth)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=448,
        help="ONNX input height, must be divisible by 32 (default: 448)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="ONNX input width, must be divisible by 32 (default: 640)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Output .onnx file (default: weights/tri_stereo/stereo_<variant>.onnx)",
    )
    add_disparity_args(parser)
    add_post_processing_args(parser)

    args = parser.parse_args()

    main(args)
