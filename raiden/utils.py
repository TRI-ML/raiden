"""Shared utilities for Raiden CLI."""

import sys
from pathlib import Path
from typing import List, Optional

import iterfzf


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
    extra = ["--height=40%", "--layout=reverse", "--border"]
    bind: dict = {}
    color: dict = {}
    hdr = ""

    if multi:
        extra += ["--marker=● ", "--header-first"]
        bind["tab"] = "toggle"
        color["marker"] = "#ffffff"
        hdr = header or "Tab: toggle  |  Enter: confirm  |  Esc: cancel"
    elif header:
        extra.append("--header-first")
        hdr = header

    try:
        result = iterfzf.iterfzf(
            items,
            prompt=prompt,
            multi=multi,
            bind=bind or None,
            color=color or None,
            header=hdr,
            __extra__=extra,
        )
    except KeyboardInterrupt:
        sys.exit(0)
    if result is None:
        sys.exit(0)
    # multi=False returns a str; multi=True returns a list[str]
    return result if isinstance(result, list) else [result]


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
