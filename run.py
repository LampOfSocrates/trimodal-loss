"""CLI notebook runner: every run gets its own results folder.

Usage:
    python run.py 00                    # run one notebook
    python run.py all                   # run 00..03 in order
    python run.py 03 --set steps=200 dataset=sketchy backbone=clip
    python run.py 00 --smoke            # toy backbone, tiny subset, seconds

Creates runs/<notebook>/<timestamp>/ BEFORE execution, exports it as
TRIMODAL_RUN_DIR (the notebooks' new_run_dir() picks it up), executes the
notebook headlessly with nbclient, then saves the executed notebook (with all
outputs/figures) into the same folder next to config.json / metrics.json.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NOTEBOOKS = {
    "EDA": "00_EDA.ipynb",
    "00": "00_data_and_adaptation.ipynb",
    "01": "01_frechet_mean_loss.ipynb",
    "02": "02_spherical_triangle_loss.ipynb",
    "03": "03_vmf_loss.ipynb",
}


def run_one(key: str, env: dict) -> Path:
    nb_file = HERE / NOTEBOOKS[key]
    name = nb_file.stem
    run_dir = HERE / "runs" / name / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(env, TRIMODAL_RUN_DIR=str(run_dir))
    print(f"=== {nb_file.name}  ->  {run_dir}")
    nb = nbformat.read(nb_file, as_version=4)
    client = NotebookClient(nb, timeout=3600, kernel_name="trimodal",
                            resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:  # save even on failure so partial outputs are inspectable
        nbformat.write(nb, run_dir / f"executed_{nb_file.name}")
    print(f"=== finished {nb_file.name}; executed copy + results in {run_dir}\n")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=[*NOTEBOOKS, "all"])
    ap.add_argument("--smoke", action="store_true",
                    help="toy backbone + tiny subset; runs in seconds")
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value",
                    help=f"config overrides, e.g. steps=200 dataset=mock")
    args = ap.parse_args()

    env = {}
    if args.smoke:
        env["TRIMODAL_SMOKE"] = "1"
    for kv in args.set:
        k, _, v = kv.partition("=")
        env["TRIMODAL_" + k.upper()] = v

    keys = list(NOTEBOOKS) if args.which == "all" else [args.which]
    for key in keys:
        run_one(key, env)


if __name__ == "__main__":
    sys.exit(main())
