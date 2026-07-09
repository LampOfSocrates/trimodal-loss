"""EXP4 · SHAPE probe — does spherical-triangle excess E predict retrieval?

Trains a BASELINE model (no triangle term — so the test isn't circular), then
per held-out triple computes E(s,v,t) and the sketch->photo rank of the correct
photo, and reports whether E separates hits from misses and correlates with rank
beyond the pairwise sketch-photo similarity.

    python docs/exp4_shape_probe.py            # default scale
    python docs/exp4_shape_probe.py --set steps=1500 num_classes=16
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from scipy import stats

from lib import sphere
from lib.config import get_config
from lib.losses import build_loss
from lib.run_utils import save_json
from lib.train import (embed_test_set, evaluate_retrieval, rank_test_queries,
                       setup_experiment, train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v")
    args = ap.parse_args()
    for kv in args.set:
        k, _, v = kv.partition("=")
        os.environ["TRIMODAL_" + k.upper()] = v

    cfg = get_config()
    exp = setup_experiment(cfg)
    print("[exp4] training BASELINE (no triangle term) so E-vs-rank is not circular")
    train(exp, build_loss("baseline", cfg, exp["model"].embed_dim))
    metrics, emb = evaluate_retrieval(exp)
    rows = rank_test_queries(exp, emb)

    # per-query sketch/photo/text embeddings aligned to emb["meta"] / rows
    s = emb["sketch"]
    # gallery photo & text per query instance
    inst_to_gpos = {inst: p for p, inst in enumerate(emb["gallery_instances"])}
    E, ranks, svsim = [], [], []
    for r in rows:
        q = r["query"]
        gpos = inst_to_gpos.get(r["instance_id"])
        if gpos is None:
            continue
        v = emb["gallery"][gpos]
        t = emb["text"][q]
        E.append(sphere.spherical_triangle_excess(
            s[q:q + 1], v.unsqueeze(0), t.unsqueeze(0)).item())
        ranks.append(r["rank"])
        svsim.append((s[q] * v).sum().item())

    E = torch.tensor(E); ranks = torch.tensor(ranks, dtype=torch.float)
    svsim = torch.tensor(svsim)
    hit = ranks == 0
    mean_E_hit = E[hit].mean().item() if hit.any() else float("nan")
    mean_E_miss = E[~hit].mean().item() if (~hit).any() else float("nan")
    rho, p = stats.spearmanr(E.numpy(), ranks.numpy())
    # partial Spearman of E vs rank controlling for s.v: residualize both on
    # rank(s.v) via rank-transform, then correlate residuals
    def rankt(x):
        return torch.tensor(stats.rankdata(x.numpy()), dtype=torch.float)
    rE, rR, rSV = rankt(E), rankt(ranks), rankt(svsim)

    def resid(y, x):
        x1 = torch.stack([torch.ones_like(x), x], 1)
        beta = torch.linalg.lstsq(x1, y.unsqueeze(1)).solution
        return y - (x1 @ beta).squeeze(1)
    partial_rho = stats.pearsonr(resid(rE, rSV).numpy(),
                                 resid(rR, rSV).numpy())[0]

    out = dict(
        n_queries=len(E),
        top1_hit_rate=hit.float().mean().item(),
        mean_excess_hits=mean_E_hit,
        mean_excess_misses=mean_E_miss,
        spearman_E_rank=float(rho), spearman_p=float(p),
        partial_spearman_E_rank_given_sv=float(partial_rho),
        metrics_R1_instance=metrics["sketch2photo_R@1_instance"],
    )
    print("\n=== EXP4 SHAPE result ===")
    for k, v in out.items():
        print(f"  {k:38s} {v}")
    verdict = ("E PREDICTS rank (sweep triangle in EXP2)"
               if (rho > 0.1 and p < 0.05 and abs(partial_rho) > 0.05)
               else "E carries little/no signal beyond pairwise sim -> de-prioritize")
    out["verdict"] = verdict
    print("\n  VERDICT:", verdict)
    save_json(out, ROOT / "docs" / "exp4_shape_result.json")
    print("  saved -> docs/exp4_shape_result.json")


if __name__ == "__main__":
    main()
