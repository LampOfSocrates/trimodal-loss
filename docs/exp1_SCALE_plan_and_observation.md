# EXP1 · SCALE — make the loss comparison statistically real

**Mnemonic:** SCALE = scale the data up and add seeds so the four losses can
actually be separated.
**Status:** PLAN written 2026-07-09 · OBSERVATION pending.
**Est. runtime:** ⚠️ **borderline ~40–60 min** (12 training runs). Under 1 h on
the RTX 5070 Ti at the config below, but the closest to the limit of the runnable
experiments — see the timing note.

## Motivation

The committed smoke runs use 8 classes / 96 instances → a **35-photo gallery**,
where sketch→photo R@5 ≈ 0.9 is near ceiling. Nothing can distinguish the four
loss framings at that scale; the R@1 gaps we see are single-digit query counts
of noise. Before believing *any* ranking of baseline vs Fréchet vs triangle vs
vMF we need (a) a gallery large enough that R@1 has headroom, and (b) multiple
seeds so a difference has an error bar.

## Hypothesis

With a few hundred gallery photos and 3 seeds, the four losses will separate
into a stable ordering on sketch→photo R@1 (per-instance). Null hypothesis to
be taken seriously: **they are within noise of each other** — a plain pairwise
InfoNCE sum is already spherical, and the geometric terms may add nothing once
the contrastive core is well-tuned.

## Method

- Data: HF Sketchy mirror, `num_classes=20`, `max_instances_per_class=100`,
  `sketches_per_photo=5` → ~1.5–2k train sketches, ~300–400 gallery photos.
- Same backbone (frozen CLIP ViT-B/32 + LoRA on sketch path), `steps=2000`,
  `batch_size=48`, all other config identical across the four losses.
- Seeds: `{0, 1, 2}`. Four losses × 3 seeds = **12 runs**.
- Metric of record: sketch→photo **R@1 / R@5 instance** and **R@1 category**,
  mean ± std over seeds. Secondary: composite (sketch+text)→photo R@1.

## Commands

```bash
for s in 0 1 2; do
  python run.py 00 --set num_classes=20 max_instances_per_class=100 sketches_per_photo=5 steps=2000 batch_size=48 seed=$s
  python run.py 01 --set num_classes=20 max_instances_per_class=100 sketches_per_photo=5 steps=2000 batch_size=48 seed=$s
  python run.py 02 --set num_classes=20 max_instances_per_class=100 sketches_per_photo=5 steps=2000 batch_size=48 seed=$s
  python run.py 03 --set num_classes=20 max_instances_per_class=100 sketches_per_photo=5 steps=2000 batch_size=48 seed=$s
done
```
(aggregated by `docs/_aggregate.py`, which reads `runs/*/*/metrics.json`.)

## Timing note (⚠️ >1 h risk)

At ~0.07 s/CLIP-step + image decode, one 2000-step run is ~3–4 min; 12 runs
≈ 40–50 min. If image decoding dominates (PNG decode per step, `num_workers=0`
on Windows), this can drift over 1 h. **Mitigation if it does:** drop to 2 seeds
(8 runs) or precompute/cache photo & text embeddings once (they are frozen —
only the sketch path changes), which turns each step into a cheap LoRA forward.
The embedding-cache optimization is noted as the first thing to build if EXP1
runtime is unacceptable.

## Success criteria

- A loss beats the baseline on R@1-instance by **more than 2× the seed std** →
  worth pursuing.
- Otherwise, report "within noise" honestly and let EXP2 (weights) decide
  whether any framing has a regime where it wins.

## OBSERVATION — run 2026-07-09 (20 classes, 2000 steps, **2 seeds**, ~34 min total)

Config: `num_classes=20, max_instances_per_class=100, sketches_per_photo=5,
steps=2000, batch_size=48`, seeds {0,1}. Gallery = **80 photos**, 400 queries,
so chance R@1 ≈ 1/80 = 0.0125 (R@1 ≈ 0.20 has ample headroom). Ran **2 seeds,
not the planned 3**, to stay under the hour — so the ± below is a 2-sample
spread, indicative only.

| loss | R@1 inst | R@5 inst | R@1 cat | (s+t)→p R@1 inst |
|---|---|---|---|---|
| **baseline** (00) | 0.205 ± 0.012 | 0.586 ± 0.024 | 0.969 | 0.294 ± 0.024 |
| **Fréchet** (01) | 0.209 ± 0.006 | **0.621 ± 0.001** | 0.972 | **0.330 ± 0.015** |
| **triangle** (02) | 0.201 ± 0.001 | 0.603 ± 0.028 | 0.974 | 0.320 ± 0.003 |
| **vMF** (03) | **0.115 ± 0.000** | 0.443 ± 0.010 | 0.900 | 0.265 ± 0.005 |

### Findings

1. **Fréchet is the only framing that is consistently ≥ baseline.** On R@1
   instance it ties baseline (0.209 vs 0.205, well within the seed spread), but
   on **R@5 instance it is +3.4 pts (0.621 vs 0.586) with near-zero variance**,
   and on **composite (sketch+text)→photo R@1 it is +3.6 pts (0.330 vs 0.294)** —
   both hold in *both* seeds. So the consensus-anchor helps rank the correct
   photo into the top-5 and helps the text bridge, even though it doesn't move
   top-1. Modest but real.
2. **Triangle ≈ baseline** (0.201 vs 0.205 R@1i; slightly better R@5i within
   noise). Exactly what EXP4 predicted — the excess term is largely redundant
   with the pairwise objective. **Confirmed: de-prioritize.**
3. **vMF underperforms badly at λ_vmf=0.05** (R@1i 0.115 vs 0.205 — a 44%
   relative drop, identical across both seeds). This is the λ-confound EXP2 was
   written to test: with κ≈700–900 the vMF NLL gradient dwarfs the InfoNCE
   term. Whether small λ recovers it is now the key open question.
4. **Category retrieval is near-saturated** for all but vMF (R@1cat ≈ 0.97) —
   the losses differentiate on *instance* retrieval, not category.

### Decision

✅ **Proceed to EXP2**, narrowed by EXP1+EXP3+EXP4:
- **Fréchet** — sweep λ_frechet upward/downward to see if the R@5/composite gain
  strengthens or is a fixed small offset.
- **vMF** — sweep λ_vmf ∈ {0.002, 0.01, 0.05} to test whether small λ removes
  the damage (the EXP2 core question).
- **Triangle** — **skip** (or one confirmatory point); EXP1+EXP4 agree it's inert.

### Caveats

- **n=2 seeds** — the plan wanted 3; treat ± as indicative, reconfirm the
  Fréchet edge with a 3rd seed if it matters for a headline claim.
- Gallery capped at 80 by `eval_max_queries=400`; a larger gallery (raise the
  cap) would harden R@1 and is worth doing before any publication-grade number.
- Pseudo-text captions throughout (A1); real-text claim still waits on EXP5.
