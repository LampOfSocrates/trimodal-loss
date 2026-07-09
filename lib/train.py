"""Shared experiment setup, training loop, retrieval evaluation, diagnostics.

All four notebooks call setup_experiment() with the same config so their
metrics.json files are directly comparable (same subset, seed, backbone,
retrieval protocol).
"""

import itertools
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from . import sphere
from .data import (GroupedInstanceSampler, TriDataset, collate,
                   prepare_triples, split_triples)
from .model import TriEncoder
from .run_utils import resolve_device, save_json, set_seed


def setup_experiment(cfg):
    """seed -> data -> split -> model. Returns a dict of everything."""
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print(f"[setup] device={device}")
    triples, ds_name = prepare_triples(cfg)
    train_t, test_t = split_triples(triples)
    model = TriEncoder(cfg, device)
    ds_train = TriDataset(train_t, model.preprocess)
    ds_test = TriDataset(test_t, model.preprocess)
    n_real = sum(t.source == "real" for t in triples)
    print(f"[setup] dataset={ds_name}  triples={len(triples)} "
          f"(real={n_real}, synthetic={len(triples) - n_real})  "
          f"train={len(train_t)} test={len(test_t)}")
    return dict(cfg=cfg, device=device, dataset_name=ds_name, model=model,
                ds_train=ds_train, ds_test=ds_test)


def make_loader(cfg, ds_train, grouped=False):
    if grouped:  # vMF needs same-instance members in-batch
        sampler = GroupedInstanceSampler(ds_train,
                                         cfg["vmf_instances_per_batch"],
                                         cfg["vmf_triples_per_instance"],
                                         seed=cfg["seed"])
        return DataLoader(ds_train, batch_sampler=sampler, collate_fn=collate)
    return DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True,
                      drop_last=True, collate_fn=collate)


def train(exp, loss_module, grouped=False):
    """Train LoRA/adapter (+ any loss-module params, e.g. vMF kappas)."""
    cfg, model = exp["cfg"], exp["model"]
    loss_module = loss_module.to(exp["device"])
    loss_params = list(loss_module.parameters())  # e.g. vMF raw_kappa
    n_par = sum(p.numel() for p in model.trainable_parameters() + loss_params)
    print(f"[train] {n_par:,} trainable params, {cfg['steps']} steps")
    groups = [dict(params=model.trainable_parameters(), lr=cfg["lr"])]
    if loss_params:
        groups.append(dict(params=loss_params, lr=cfg["kappa_lr"]))
    opt = torch.optim.Adam(groups)
    loader = make_loader(cfg, exp["ds_train"], grouped=grouped)
    it = itertools.cycle(loader)
    history = []
    t0 = time.time()
    model.train()
    for step in range(cfg["steps"]):
        batch = next(it)
        s = model.encode_sketch(batch["sketch"])          # grad via LoRA
        v = model.encode_photo(batch["photo"])            # frozen
        t = model.encode_text(batch["caption"])           # frozen
        ids = batch["instance_id"].to(exp["device"])
        loss, comp = loss_module(s, v, t, ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec = dict(step=step, loss=loss.item(), **comp)
        history.append(rec)
        if step % max(1, cfg["steps"] // 10) == 0 or step == cfg["steps"] - 1:
            extras = {k: v for k, v in comp.items() if k.startswith("kappa")}
            print(f"  step {step:4d}  loss {loss.item():.4f}  "
                  + " ".join(f"{k}={v:.1f}" for k, v in extras.items())
                  + f"  ({time.time() - t0:.0f}s)")
    model.eval()
    return history


# ------------------------------------------------------------- eval --------

@torch.no_grad()
def embed_test_set(exp, max_items=None):
    """Embed the test split. Photos deduplicated by instance -> gallery.
    Returns dict of tensors + aligned metadata lists."""
    cfg, model, ds = exp["cfg"], exp["model"], exp["ds_test"]
    idxs = list(range(len(ds)))[: (max_items or cfg["eval_max_queries"])]
    loader = DataLoader(Subset(ds, idxs),
                        batch_size=cfg["batch_size"], collate_fn=collate)
    S, T, meta = [], [], []
    photo_by_inst = {}
    for batch in loader:
        S.append(model.encode_sketch(batch["sketch"], grad=False).cpu())
        T.append(model.encode_text(batch["caption"]).cpu())
        V = model.encode_photo(batch["photo"]).cpu()
        for j, inst in enumerate(batch["instance_id"].tolist()):
            meta.append(dict(instance_id=inst,
                             class_id=batch["class_id"][j].item(),
                             source=batch["source"][j]))
            photo_by_inst.setdefault(inst, (V[j], batch["class_id"][j].item()))
    gal_insts = sorted(photo_by_inst)
    gallery = torch.stack([photo_by_inst[i][0] for i in gal_insts])
    return dict(sketch=torch.cat(S), text=torch.cat(T), meta=meta,
                gallery=gallery, gallery_instances=gal_insts,
                gallery_classes=[photo_by_inst[i][1] for i in gal_insts])


def recall_at_k(sim, meta, gal_insts, gal_classes, ks):
    """Per-instance (exact paired photo) and per-category (any same-class
    photo) Recall@K — both are needed, they change character dramatically."""
    gal_i = torch.tensor(gal_insts)
    gal_c = torch.tensor(gal_classes)
    ranks = sim.argsort(dim=1, descending=True)
    out = {}
    for k in ks:
        top = ranks[:, :k]
        inst_hit = cat_hit = 0
        for q, m in enumerate(meta):
            inst_hit += (gal_i[top[q]] == m["instance_id"]).any().item()
            cat_hit += (gal_c[top[q]] == m["class_id"]).any().item()
        out[f"R@{k}_instance"] = inst_hit / len(meta)
        out[f"R@{k}_category"] = cat_hit / len(meta)
    return out


def evaluate_retrieval(exp):
    """sketch->photo and composite (sketch+text)->photo retrieval metrics."""
    emb = embed_test_set(exp)
    ks = exp["cfg"]["recall_ks"]
    metrics = {}
    sim_s = emb["sketch"] @ emb["gallery"].t()
    for k, val in recall_at_k(sim_s, emb["meta"], emb["gallery_instances"],
                              emb["gallery_classes"], ks).items():
        metrics["sketch2photo_" + k] = val
    # composite query: normalized midpoint of sketch and text on the sphere
    q = sphere.normalize(emb["sketch"] + emb["text"])
    sim_q = q @ emb["gallery"].t()
    for k, val in recall_at_k(sim_q, emb["meta"], emb["gallery_instances"],
                              emb["gallery_classes"], ks).items():
        metrics["sketchtext2photo_" + k] = val
    metrics["n_queries"] = len(emb["meta"])
    metrics["n_gallery"] = len(emb["gallery_instances"])
    return metrics, emb


# -------------------------------------------------------- diagnostics ------

def cap_diagnostic(emb):
    """Synthetic-vs-real sketch cap diagnostic (paper.md §3.6): per source,
    Banerjee kappa-hat + mean direction; report the angle between cap means.
    Guards the augmentation rule — synthetic caps are tighter & offset."""
    out = {}
    dirs = {}
    for source in ("real", "synthetic"):
        idx = [i for i, m in enumerate(emb["meta"]) if m["source"] == source]
        if len(idx) < 5:
            out[source] = None
            continue
        kap, mu = sphere.kappa_mle(emb["sketch"][idx])
        out[source] = dict(kappa_hat=kap.item(), n=len(idx))
        dirs[source] = mu
    if len(dirs) == 2:
        ang = torch.rad2deg(sphere.geodesic_distance(dirs["real"],
                                                     dirs["synthetic"]))
        out["cap_angle_deg"] = ang.item()
    return out


def per_instance_kappa_sketch(emb, min_sketches=3, real_only=True):
    """Per-instance kappa_hat over an instance's sketches — the quantity
    Sketchy's >=5 sketches/photo makes estimable. real_only enforces the
    'kappa from REAL sketches only' rule (paper.md §3.4)."""
    groups = {}
    for i, m in enumerate(emb["meta"]):
        if real_only and m["source"] != "real":
            continue
        groups.setdefault(m["instance_id"], []).append(i)
    kappas = []
    for inst, idx in groups.items():
        if len(idx) >= min_sketches:
            kap, _ = sphere.kappa_mle(emb["sketch"][idx])
            kappas.append(kap.item())
    return kappas


# ------------------------------------------------- qualitative triads ------

def rank_test_queries(exp, emb):
    """For every sketch query, the rank (0-based) at which its correct photo
    instance appears in the sketch->photo gallery ranking. Lower is better;
    rank 0 = top-1 hit. Returns a list of dicts sorted worst-first."""
    sim = emb["sketch"] @ emb["gallery"].t()
    order = sim.argsort(dim=1, descending=True)
    gal_inst = emb["gallery_instances"]
    inst_to_pos = {inst: p for p, inst in enumerate(gal_inst)}
    rows = []
    for q, m in enumerate(emb["meta"]):
        ranking = order[q].tolist()
        gt_gallery = inst_to_pos.get(m["instance_id"])
        rank = ranking.index(gt_gallery) if gt_gallery is not None else len(ranking)
        rows.append(dict(query=q, rank=rank, instance_id=m["instance_id"],
                         class_id=m["class_id"], source=m["source"],
                         top1_instance=gal_inst[ranking[0]],
                         top1_sim=sim[q, ranking[0]].item(),
                         gt_sim=(sim[q, gt_gallery].item()
                                 if gt_gallery is not None else float("nan"))))
    rows.sort(key=lambda r: (-r["rank"], -(r["top1_sim"] - r["gt_sim"])))
    return rows


def show_good_bad_triads(exp, emb, run_dir, k_show=10):
    """Render the k_show WORST and k_show BEST sketch->photo queries as triads
    [query sketch | ground-truth photo | top-1 retrieved photo], annotated with
    rank and the class/source. Worst performers are drawn first and flagged, so
    failure modes (which classes/sources the model misses) are visible at a
    glance. Saves worst_triads.png / best_triads.png and returns the ranked
    rows so notebooks can tabulate the hardest cases."""
    from PIL import Image

    rows = rank_test_queries(exp, emb)
    triples = exp["ds_test"].triples
    photo_of, class_name = {}, {}
    for t in triples:
        photo_of.setdefault(t.instance_id, t.photo_path)
        class_name.setdefault(t.class_id, t.class_name)

    def render(subset, title, fname):
        n = len(subset)
        fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.3 * n))
        if n == 1:
            axes = axes.reshape(1, 3)
        for r, row in enumerate(subset):
            q = row["query"]
            hit = row["rank"] == 0
            imgs = [(triples[q].sketch_path, f"sketch [{row['source']}]"),
                    (photo_of[row["instance_id"]], "ground-truth photo"),
                    (photo_of[row["top1_instance"]],
                     f"top-1 (rank of GT: {row['rank']})")]
            for c, (path, cap) in enumerate(imgs):
                axes[r, c].imshow(Image.open(path))
                axes[r, c].axis("off")
                color = "green" if (c == 2 and hit) else (
                    "red" if c == 2 else "black")
                axes[r, c].set_title(cap, fontsize=8, color=color)
            axes[r, 0].set_ylabel(class_name.get(row["class_id"], "?"),
                                  fontsize=8, rotation=0, ha="right",
                                  labelpad=28)
            axes[r, 0].axis("on")
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        fig.suptitle(title, fontsize=11, y=1.0)
        fig.tight_layout()
        fig.savefig(run_dir / fname, dpi=110, bbox_inches="tight")
        plt.show()

    worst = rows[:k_show]
    best = [r for r in rows if r["rank"] == 0][:k_show] or rows[-k_show:]
    render(worst, f"{k_show} WORST sketch->photo queries (hardest first)",
           "worst_triads.png")
    render(best, f"{k_show} BEST sketch->photo queries (top-1 hits)",
           "best_triads.png")

    from collections import Counter
    miss = Counter(class_name.get(r["class_id"], "?")
                   for r in rows if r["rank"] > 0)
    print(f"\nWorst-performing classes (by miss count, rank>0): "
          f"{miss.most_common(8)}")
    print(f"Mean rank {sum(r['rank'] for r in rows) / len(rows):.2f}  |  "
          f"top-1 hit rate {sum(r['rank'] == 0 for r in rows) / len(rows):.3f}")
    return rows


# ------------------------------------------------------------ report -------

def finish_run(run_dir, notebook, cfg, history, metrics, extra=None):
    save_json({k: v for k, v in cfg.items()}, run_dir / "config.json")
    save_json(history, run_dir / "history.json")
    rec = dict(notebook=notebook, dataset=cfg["dataset"],
               backbone=cfg["backbone"], steps=cfg["steps"],
               seed=cfg["seed"], metrics=metrics, **(extra or {}))
    save_json(rec, run_dir / "metrics.json")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["step"] for h in history], [h["loss"] for h in history])
    ax.set(xlabel="step", ylabel="loss", title=f"{notebook} training loss")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close(fig)
    print(f"[done] results in {run_dir}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:38s} {v:.4f}")
    return rec
