#!/usr/bin/env python3
"""Download the pyzed wheel for the installed ZED SDK and update pyproject.toml.

Usage:
    uv run python scripts/install_pyzed.py

This script:
  1. Detects the installed ZED SDK version, Python version, and platform.
  2. Downloads the matching pyzed wheel into packages/.
  3. Updates the [tool.uv.sources] entry in pyproject.toml to point at the
     new wheel so that `uv sync` installs it.
"""

import platform
import re
import sys
import urllib.request
from pathlib import Path

_RAIDEN_DIR = Path(__file__).parent.parent
_PACKAGES_DIR = _RAIDEN_DIR / "packages"
_PYPROJECT = _RAIDEN_DIR / "pyproject.toml"
_ZED_CMAKE = Path("/usr/local/zed/zed-config-version.cmake")


def _zed_sdk_version() -> str:
    text = _ZED_CMAKE.read_text()
    m = re.search(r'set\(PACKAGE_VERSION\s+"([^"]+)"\)', text)
    if not m:
        raise RuntimeError(f"Could not parse ZED SDK version from {_ZED_CMAKE}")
    # Use only major.minor (e.g. "5.2" from "5.2.0")
    parts = m.group(1).split(".")
    return f"{parts[0]}.{parts[1]}"


def _platform_tag() -> str:
    machine = platform.machine()
    if machine == "x86_64":
        return "linux_x86_64"
    if machine == "aarch64":
        return "linux_aarch64"
    raise RuntimeError(f"Unsupported platform: {machine}")


def _python_tag() -> str:
    v = sys.version_info
    return f"cp{v.major}{v.minor}"


def main() -> None:
    sdk = _zed_sdk_version()
    plat = _platform_tag()
    py = _python_tag()

    wheel_name = f"pyzed-{sdk}-{py}-{py}-{plat}.whl"
    url = f"https://download.stereolabs.com/zedsdk/{sdk}/whl/{plat}/{wheel_name}"

    _PACKAGES_DIR.mkdir(exist_ok=True)
    dest = _PACKAGES_DIR / wheel_name

    if dest.exists():
        print(f"Already present: packages/{wheel_name}")
    else:
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to packages/{wheel_name}")

    # Update pyproject.toml: replace the pyzed path source with the new wheel.
    text = _PYPROJECT.read_text()
    new_source = f'pyzed = {{ path = "packages/{wheel_name}" }}'
    updated = re.sub(
        r"pyzed\s*=\s*\{[^}]*\}",
        new_source,
        text,
    )
    if updated == text:
        print("pyproject.toml already up to date.")
    else:
        _PYPROJECT.write_text(updated)
        print(f"Updated pyproject.toml: pyzed → packages/{wheel_name}")

    print(
        "\nRun `uv sync --extra zed` to install the wheel into the project environment."
    )


if __name__ == "__main__":
    main()
