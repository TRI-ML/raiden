"""Shared utilities for Raiden CLI."""

import platform
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Optional

_FZF_MIN_VERSION = (0, 70, 0)
_FZF_INSTALL_DIR = Path.home() / ".local" / "bin"


def _parse_fzf_version(version_str: str) -> tuple[int, ...]:
    # version_str is like "0.70.0" (may have a leading "v")
    return tuple(int(x) for x in version_str.lstrip("v").split(".")[:3])


def _ensure_fzf() -> str:
    """Return the fzf binary path, installing to ~/.local/bin if missing or too old."""
    fzf_candidates = [_FZF_INSTALL_DIR / "fzf", Path("fzf")]
    for candidate in fzf_candidates:
        try:
            result = subprocess.run(
                [str(candidate), "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                version = _parse_fzf_version(result.stdout.split()[0])
                if version >= _FZF_MIN_VERSION:
                    return str(candidate)
        except FileNotFoundError:
            continue

    min_ver = ".".join(str(x) for x in _FZF_MIN_VERSION)
    print(f"fzf >= {min_ver} not found — installing to {_FZF_INSTALL_DIR} ...")

    machine = platform.machine().lower()
    arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"
    # Fetch the latest release tag from GitHub
    api_url = "https://api.github.com/repos/junegunn/fzf/releases/latest"
    with urllib.request.urlopen(api_url) as resp:
        import json

        tag = json.loads(resp.read())["tag_name"]  # e.g. "v0.70.0"
    ver = tag.lstrip("v")
    tarball_url = f"https://github.com/junegunn/fzf/releases/download/{tag}/fzf-{ver}-linux_{arch}.tar.gz"

    _FZF_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(tarball_url) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            member = tf.next()
            while member is not None:
                if member.name == "fzf" or member.name.endswith("/fzf"):
                    member.name = "fzf"
                    tf.extract(member, path=_FZF_INSTALL_DIR)
                    break
                member = tf.next()

    fzf_bin = _FZF_INSTALL_DIR / "fzf"
    fzf_bin.chmod(fzf_bin.stat().st_mode | 0o111)
    print(f"fzf {ver} installed to {fzf_bin}")
    return str(fzf_bin)


def fzf_select(
    items: List[str],
    prompt: str,
    multi: bool = False,
    header: Optional[str] = None,
) -> List[str]:
    """Pipe *items* to fzf and return the selected entries.

    Uses ``--multi`` when *multi=True* (Tab to toggle items).
    Exits cleanly if the user cancels (Esc / Ctrl-C).
    """
    fzf_bin = _ensure_fzf()
    args = [
        fzf_bin,
        f"--prompt={prompt}",
        "--height=40%",
        "--layout=reverse",
        "--border",
    ]
    if multi:
        args += [
            "--multi",
            "--bind=tab:toggle",
            "--marker=● ",
            "--color=marker:#ffffff",
        ]
        default_header = "Tab: toggle  |  Enter: confirm  |  Esc: cancel"
        args += [f"--header={header or default_header}", "--header-first"]
    elif header:
        args += [f"--header={header}", "--header-first"]
    result = subprocess.run(
        args,
        input="\n".join(items),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        sys.exit(0)
    return [line for line in result.stdout.strip().splitlines() if line]


def select_recording(data_dir: str = "data/raw") -> Optional[Path]:
    """Interactively select a single recording episode using fzf.

    Returns the Path to the selected episode directory, or None if the user
    cancels.  An episode directory is a timestamped subdirectory of a task
    folder that contains a ``cameras/`` subdirectory.
    """
    base = Path(data_dir)
    if not base.exists():
        print(f"No recordings found in {base}")
        sys.exit(1)

    episodes: dict[str, Path] = {}
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "cameras").exists():
                label = f"{task_dir.name} / {ep_dir.name}"
                episodes[label] = ep_dir

    if not episodes:
        print(f"No recordings found in {base}")
        sys.exit(1)

    selected = fzf_select(list(episodes), prompt="Select recording> ")
    if not selected:
        return None
    return episodes[selected[0]]


def select_processed_recording(data_dir: str = "data/processed") -> Optional[Path]:
    """Interactively select a single converted episode using fzf.

    Returns the Path to the selected episode directory, or None if the user
    cancels.  An episode directory contains a ``metadata.json`` file.
    """
    base = Path(data_dir)
    if not base.exists():
        print(f"No processed recordings found in {base}")
        sys.exit(1)

    episodes: dict[str, Path] = {}
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if ep_dir.is_dir() and (ep_dir / "metadata.json").exists():
                label = f"{task_dir.name} / {ep_dir.name}"
                episodes[label] = ep_dir

    if not episodes:
        print(f"No processed recordings found in {base}")
        sys.exit(1)

    selected = fzf_select(list(episodes), prompt="Select recording> ")
    if not selected:
        return None
    return episodes[selected[0]]
