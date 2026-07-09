"""Run-folder management, seeding, and small IO helpers.

Every notebook execution — interactive or via run.py — gets its own results
folder runs/<notebook>/<timestamp>/ holding config.json, metrics.json and all
figures, so runs are comparable and never overwrite each other.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def new_run_dir(notebook_name: str) -> Path:
    """Create runs/<notebook>/<timestamp>/ — or reuse the dir the CLI runner
    (run.py) pre-created and exported via TRIMODAL_RUN_DIR."""
    env = os.environ.get("TRIMODAL_RUN_DIR")
    if env:
        run_dir = Path(env)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = PROJECT_ROOT / "runs" / notebook_name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default))


def collect_runs() -> list[dict]:
    """Scan runs/*/*/metrics.json into a flat list (newest first per
    notebook) for cross-notebook comparison tables."""
    out = []
    root = PROJECT_ROOT / "runs"
    if not root.exists():
        return out
    for mfile in sorted(root.glob("*/*/metrics.json")):
        try:
            rec = json.loads(mfile.read_text())
            rec["_run_dir"] = str(mfile.parent)
            out.append(rec)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def latest_metrics_per_notebook() -> dict[str, dict]:
    latest = {}
    for rec in collect_runs():  # sorted by path => timestamp order per notebook
        latest[rec.get("notebook", "?")] = rec
    return latest
