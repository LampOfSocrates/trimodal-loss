"""Aggregate runs/*/*/metrics.json into a per-config table for experiment
observations. Groups runs by (notebook, dataset, steps, key overrides) and
reports mean +/- std over seeds for the headline retrieval metrics.

    python docs/_aggregate.py                 # all runs
    python docs/_aggregate.py --since 20260709-1900   # only runs after a stamp
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["sketch2photo_R@1_instance", "sketch2photo_R@5_instance",
        "sketch2photo_R@1_category", "sketchtext2photo_R@1_instance"]
SHORT = ["R@1i", "R@5i", "R@1c", "s+t R@1i"]


def mean_std(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="")
    ap.add_argument("--notebook", default="")
    args = ap.parse_args()

    groups = defaultdict(lambda: defaultdict(list))
    for mfile in sorted((ROOT / "runs").glob("*/*/metrics.json")):
        stamp = mfile.parent.name
        if args.since and stamp < args.since:
            continue
        rec = json.loads(mfile.read_text())
        nb = rec.get("notebook", "?")
        if args.notebook and args.notebook not in nb:
            continue
        cfg = json.loads((mfile.parent / "config.json").read_text())
        # group key = notebook + the config knobs that define a "condition"
        key = (nb, cfg.get("dataset"), cfg.get("steps"), cfg.get("num_classes"),
               cfg.get("lambda_frechet"), cfg.get("lambda_triangle"),
               cfg.get("lambda_vmf"), cfg.get("vmf_kappa_mode", "learned"))
        for k in KEYS:
            groups[key][k].append(rec.get("metrics", {}).get(k))
        groups[key]["_seeds"].append(cfg.get("seed"))

    hdr = f"{'notebook':26s}{'ds':10s}{'steps':>6s}{'ncls':>5s}{'lamF':>6s}{'lamT':>6s}{'lamV':>7s}{'kappa':>8s}{'n':>3s}"
    hdr += "".join(f"{s:>16s}" for s in SHORT)
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(groups):
        nb, ds, steps, ncls, lf, lt, lv, km = key
        g = groups[key]
        n = len(g["_seeds"])
        row = f"{nb:26s}{str(ds):10s}{str(steps):>6s}{str(ncls):>5s}{str(lf):>6s}{str(lt):>6s}{str(lv):>7s}{km:>8s}{n:>3d}"
        for k in KEYS:
            m, s = mean_std(g[k])
            row += f"   {m:6.3f}±{s:5.3f}"
        print(row)


if __name__ == "__main__":
    main()
