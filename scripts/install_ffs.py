#!/usr/bin/env python3
"""Populate third_party/Fast-FoundationStereo with the real FFS source.

Fast-FoundationStereo has no setup.py/pyproject.toml of its own, so raiden
ships a packaging shim at third_party/Fast-FoundationStereo/ that is already
part of the uv workspace.  This script fetches the actual model code from
GitHub into that directory so the installed package becomes functional.

Usage::

    uv run python scripts/install_ffs.py [--download-weights]

The ``--download-weights`` flag downloads the default pretrained checkpoint
(requires ``gdown``).  You can also place any ``*.pth`` file in
``~/.config/raiden/weights/`` manually.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/NVlabs/Fast-FoundationStereo.git"
FFS_DIR = Path(__file__).parent.parent / "third_party" / "Fast-FoundationStereo"

# Files/dirs to copy from the FFS repo into our stub directory.
# We preserve our own pyproject.toml and only bring in the model source.
_COPY_ITEMS = [
    "core",
    "Utils.py",
    "scripts",
]

# Sentinel: placeholder core/__init__.py written by raiden (not real FFS code).
_PLACEHOLDER_MARKER = "Placeholder — replaced by the real FFS source"


def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def _is_placeholder() -> bool:
    """Return True if core/__init__.py is still the raiden placeholder."""
    init = FFS_DIR / "core" / "__init__.py"
    if not init.exists():
        return True
    return _PLACEHOLDER_MARKER in init.read_text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download the default pretrained weights after populating (requires gdown)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch source even if already populated",
    )
    args = parser.parse_args()

    # ── 1. Fetch FFS source ───────────────────────────────────────────────
    if not _is_placeholder() and not args.force:
        print(f"FFS source already populated at {FFS_DIR} — skipping fetch.")
        print("Use --force to re-fetch.")
    else:
        print(f"\nFetching Fast-FoundationStereo source into {FFS_DIR} …")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "ffs"
            run(["git", "clone", "--depth", "1", REPO_URL, str(tmp_path)])

            for item in _COPY_ITEMS:
                src = tmp_path / item
                dst = FFS_DIR / item
                if not src.exists():
                    print(f"  Warning: {item} not found in cloned repo — skipping.")
                    continue
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                print(f"  Copied: {item}")

            # Copy config files (needed for model loading).
            for cfg in tmp_path.glob("*.yaml"):
                shutil.copy2(cfg, FFS_DIR / cfg.name)
                print(f"  Copied: {cfg.name}")

        print("  Source populated.")

    # ── 2. Ensure pyproject.toml exists (FFS ships none) ─────────────────
    pyproject = FFS_DIR / "pyproject.toml"
    if not pyproject.exists():
        pyproject.write_text(
            "[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n'
            "\n"
            "[project]\n"
            'name = "foundation-stereo"\n'
            'version = "0.1.0"\n'
            'description = "Fast-FoundationStereo stereo depth estimation"\n'
            'requires-python = ">=3.11"\n'
            "dependencies = []\n"
            "\n"
            "[tool.hatch.build.targets.wheel]\n"
            'packages = ["core"]\n'
        )
        print("  Created pyproject.toml shim.")

    # ── 3. Re-install so the new core/ is picked up ───────────────────────
    print("\nRe-installing foundation-stereo into the active environment …")
    run(["uv", "pip", "install", "-e", str(FFS_DIR)])

    # ── 3. Optional weight download ───────────────────────────────────────
    weights_dir = Path.home() / ".config" / "raiden" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    existing = list(weights_dir.glob("*.pth"))

    if existing:
        print(f"\nCheckpoint already present: {existing[0].name}")
    elif args.download_weights:
        print("\nDownloading default checkpoint (model_best_bp2.pth) …")
        try:
            import gdown  # type: ignore[import]
        except ImportError:
            print("  gdown not found — installing …")
            run(["uv", "pip", "install", "gdown"])
            import gdown  # type: ignore[import]

        # File ID from the Fast-FoundationStereo README.
        file_id = "1mQKCStBHN9OQYNtOGJTVplv5HjVfqnjH"
        out_path = weights_dir / "model_best_bp2.pth"
        gdown.download(id=file_id, output=str(out_path), quiet=False)
        print(f"  Saved: {out_path}")
    else:
        print(
            "\nNo checkpoint found in ~/.config/raiden/weights/. "
            "Download one from https://github.com/NVlabs/Fast-FoundationStereo "
            "and place it there, or rerun with --download-weights."
        )

    print("\nDone. FFS is ready — use --stereo-method ffs with rd convert.\n")


if __name__ == "__main__":
    main()
