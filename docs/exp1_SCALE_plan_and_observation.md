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

## OBSERVATION (fill after run)

- [ ] Results table (mean ± std over seeds), per loss.
- [ ] Which loss (if any) clears the 2×-std bar vs baseline.
- [ ] Gallery size / ceiling check: is R@1-instance now well below 1.0?
- [ ] Worst-class carryover: are jellyfish/duck/snail still the hardest at
      scale (from the notebooks' worst-triads), or does scale fix them?
- [ ] Decision: proceed to EXP2 weight sweep on which loss(es)?
