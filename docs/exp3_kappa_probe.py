"""EXP3 · KAPPA probe — honest test of kappa_sketch < kappa_photo.

Part A: on held-out test embeddings, per-modality Banerjee kappa-hat BEFORE
(frozen CLIP sketch path) vs AFTER (LoRA-trained), plus mean per-instance
kappa-hat over each instance's real sketches. This measures concentration as a
STATISTIC of the data, sidestepping the training confound (learned kappa on the
train set measures post-adaptation fit, not intrinsic modality noise).

Part B: train the vMF loss with kappa_mode=learned vs ema and compare both the
kappa ordering and R@1.

    python docs/exp3_kappa_probe.py --set steps=1200 num_classes=16 max_instances_per_class=80 sketches_per_photo=5 batch_size=48
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from lib import sphere
from lib.config import get_config
from lib.losses import build_loss
from lib.run_utils import save_json
from lib.train import (embed_test_set, evaluate_retrieval, setup_experiment,
                       train)


def per_modality_kappa(emb):
    """Global Banerjee kappa-hat per modality, estimated around each concept's
    own mean direction (so it measures within-concept concentration)."""
    out = {}
    # group query indices by instance to form per-concept means
    by_inst = defaultdict(list)
    for i, m in enumerate(emb["meta"]):
        by_inst[m["instance_id"]].append(i)
    d = emb["sketch"].shape[1]
    for name, key in [("sketch", "sketch"), ("text", "text")]:
        X = emb[key]
        aligns = []
        for inst, idx in by_inst.items():
            mu = sphere.normalize(X[idx].mean(0, keepdim=True))
            aligns.append((X[idx] * mu).sum(-1))
        r = torch.cat(aligns).mean().clamp(1e-4, 1 - 1e-4)
        out[name] = (r * (d - r ** 2) / (1 - r ** 2)).item()
    # photo: one embedding per instance -> concentration across instances of a
    # class around the class mean (photos have no within-instance spread)
    gal = emb["gallery"]
    gc = torch.tensor(emb["gallery_classes"])
    aligns = []
    for c in gc.unique():
        sub = gal[gc == c]
        if len(sub) < 2:
            continue
        mu = sphere.normalize(sub.mean(0, keepdim=True))
        aligns.append((sub * mu).sum(-1))
    if aligns:
        r = torch.cat(aligns).mean().clamp(1e-4, 1 - 1e-4)
        out["photo_classlevel"] = (r * (d - r ** 2) / (1 - r ** 2)).item()
    return out


def per_instance_sketch_kappa(emb, min_sk=3):
    groups = defaultdict(list)
    for i, m in enumerate(emb["meta"]):
        if m["source"] == "real":
            groups[m["instance_id"]].append(i)
    ks = []
    for inst, idx in groups.items():
        if len(idx) >= min_sk:
            k, _ = sphere.kappa_mle(emb["sketch"][idx])
            ks.append(k.item())
    return (sum(ks) / len(ks) if ks else float("nan")), len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v")
    args = ap.parse_args()
    for kv in args.set:
        k, _, v = kv.partition("=")
        os.environ["TRIMODAL_" + k.upper()] = v

    cfg = get_config()
    result = {}

    # -------- Part A: before vs after adaptation (baseline training) --------
    exp = setup_experiment(cfg)
    emb_before = embed_test_set(exp)            # frozen CLIP + zero-init LoRA
    result["A_before"] = per_modality_kappa(emb_before)
    kb, nb = per_instance_sketch_kappa(emb_before)
    result["A_before"]["per_instance_sketch_kappa"] = kb
    result["A_before"]["n_instances"] = nb

    print("[exp3] training baseline for the AFTER snapshot…")
    train(exp, build_loss("baseline", cfg, exp["model"].embed_dim))
    _, emb_after = evaluate_retrieval(exp)
    result["A_after"] = per_modality_kappa(emb_after)
    ka, na = per_instance_sketch_kappa(emb_after)
    result["A_after"]["per_instance_sketch_kappa"] = ka

    # -------- Part B: learned vs EMA kappa (fresh models) -------------------
    result["B"] = {}
    for mode in ("learned", "ema"):
        os.environ["TRIMODAL_VMF_KAPPA_MODE"] = mode
        cfg_m = get_config()
        exp_m = setup_experiment(cfg_m)
        lm = build_loss("vmf", cfg_m, exp_m["model"].embed_dim)
        print(f"[exp3] training vMF kappa_mode={mode}…")
        train(exp_m, lm, grouped=True)
        metrics_m, _ = evaluate_retrieval(exp_m)
        result["B"][mode] = dict(
            kappa={n: k.item() for n, k in zip(lm.MODALITIES, lm.kappas())},
            R1_instance=metrics_m["sketch2photo_R@1_instance"],
            R5_instance=metrics_m["sketch2photo_R@5_instance"],
        )

    print("\n=== EXP3 KAPPA result ===")
    import json
    print(json.dumps(result, indent=2))
    save_json(result, ROOT / "docs" / "exp3_kappa_result.json")
    # quick honest-test readout
    a = result["A_after"]
    print(f"\nHeld-out AFTER: kappa_sketch(per-instance)={a['per_instance_sketch_kappa']:.1f} "
          f"vs kappa_photo(class)={a.get('photo_classlevel', float('nan')):.1f}  "
          f"-> H1 {'HOLDS' if a['per_instance_sketch_kappa'] < a.get('photo_classlevel', 1e9) else 'FAILS'}")


if __name__ == "__main__":
    main()
