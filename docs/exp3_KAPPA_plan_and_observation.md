# EXP3 · KAPPA — test κ_sketch < κ_photo with the training confound removed

**Mnemonic:** KAPPA = does the vMF concentration actually recover sketch
abstraction variance, honestly measured?
**Status:** PLAN written 2026-07-09 · OBSERVATION pending.
**Est. runtime:** ✅ **~20–30 min** (well under 1 h) — mostly a diagnostic on
already-trained models plus one training variant.

## Motivation

The vMF framing's headline prediction is **κ_sketch < κ_photo** (sketches are
more dispersed). The committed run found the **opposite** (κ_sketch highest,
607 vs photo 534), and the paper already flags why: only the sketch path is
trainable, so the vMF term *trains sketches toward the consensus μ* and the
learned κ measures **post-adaptation concentration on the training set**, not
intrinsic modality noise. That makes the learned-κ number the wrong instrument.
This experiment measures κ honestly.

## Hypotheses

- **H1 (held-out κ̂):** estimated on the **test** split with the Banerjee
  MLE — not gradient-learned — κ̂_sketch < κ̂_photo, because human sketches of a
  held-out photo genuinely scatter more than the single photo embedding.
- **H2 (statistic beats parameter):** replacing the gradient-learned κ with a
  detached EMA of the per-modality Banerjee κ̂ (κ as a *statistic*, not a free
  parameter) preserves the uncertainty-weighting benefit and yields the
  predicted κ_sketch < κ_photo ordering.

## Method

**Part A — held-out κ̂ (no training change):** on the test embeddings from a
trained baseline model, compute per-modality κ̂ (Banerjee) three ways:
1. global per-modality κ̂ over all test sketches / photos / texts;
2. **per-instance** κ̂_sketch over each instance's ≥3 real sketches (the Sketchy
   structure), averaged — the cleanest estimate of sketch spread;
3. before vs after adaptation (frozen CLIP sketch path vs LoRA-trained), to see
   whether adaptation *narrows* the sketch cap.
`03_vmf_loss.ipynb` already computes (1) and (2) in its diagnostic section; this
formalizes and tabulates them and adds the before/after column.

**Part B — EMA-κ variant (small training change):** add a `vmf_kappa_mode`
config (`learned` | `ema`). In `ema` mode, `VMFLoss` sets each κ_m to a detached
exponential moving average of the batch Banerjee κ̂ for that modality instead of
a learned parameter, so κ reports the data's concentration and only the mean
direction drives gradients. Train once per mode and compare both the κ ordering
and R@1.

## Commands

```bash
# Part A: reuse a trained model; run the vMF notebook and read its κ̂ diagnostic
python run.py 03 --set steps=1500 num_classes=16 max_instances_per_class=80 sketches_per_photo=5
# Part B: EMA vs learned κ
python run.py 03 --set vmf_kappa_mode=learned steps=1500 num_classes=16 max_instances_per_class=80 sketches_per_photo=5
python run.py 03 --set vmf_kappa_mode=ema     steps=1500 num_classes=16 max_instances_per_class=80 sketches_per_photo=5
```

## What would falsify the framing

If **held-out** per-instance κ̂_sketch is *not* below κ̂_photo, the core vMF
claim fails on this data (or the pseudo-captions/scale are inadequate and it must
wait for FS-COCO / real multi-sketch Sketchy). Either way we report it plainly.

## OBSERVATION — run 2026-07-09 (16 classes, 1000 steps, held-out test = 1575 triples)

Raw numbers in `docs/exp3_kappa_result.json`. **Total time ~7 min** (baseline
train + two vMF trainings). Config: `num_classes=16, max_instances_per_class=80,
sketches_per_photo=5, batch_size=48`, HF Sketchy mirror.

### Part A — held-out per-modality κ̂ (Banerjee), before vs after adaptation

| modality (grouping) | BEFORE (frozen CLIP) | AFTER (LoRA baseline) |
|---|---|---|
| sketch, per-instance (5 sketches/photo) | 8080 | **1934** |
| photo, class-level | 2950 | 2950 (path frozen) |
| text, per-instance | 2.55e6 | 2.55e6 |

### Part B — learned vs EMA κ (fresh vMF models)

| mode | κ_sketch | κ_photo | κ_text | R@1-inst | R@5-inst |
|---|---|---|---|---|---|
| learned (gradient param) | 859 | 718 | 727 | 0.118 | 0.433 |
| ema (statistic) | 6495 | 1175 | 1091 | 0.095 | 0.363 |

### Interpretation — the prediction is NOT supported here, and the honest test is blocked by the data

1. **κ_sketch < κ_photo fails by every estimator.** Learned-κ, EMA-κ, and the
   before-adaptation held-out κ̂ all put **sketch concentration HIGH, not low**.
   The framing's headline intuition ("human sketches scatter more than photos ⇒
   low κ_sketch") simply does not show up on this dataset.

2. **A clean matched test is impossible on the Sketchy mirror** — exactly the
   paper's §3.3 tension, now empirical:
   - *Photos* have **one image per instance**, so there is no within-instance
     photo spread to compare against within-instance sketch spread; the only
     photo κ available is *class-level* (a different grouping), making the
     `1934 < 2950` "H1 HOLDS" readout an **apples-to-oranges** artifact, not a
     real confirmation. (The script prints HOLDS; do not trust it — see below.)
   - *Text* is **degenerate**: the mirror's generated captions are near-identical
     within a class, so text embeddings collapse (κ̂ ≈ 2.5e6). Any "text
     uncertainty" number here is meaningless (A1).

3. **Adaptation *spreads* an instance's sketches** (per-instance κ̂ 8080 → 1934),
   the opposite of the "training narrows the cap" guess in the plan — LoRA pulls
   each sketch toward its photo/text and in doing so decorrelates the 5 sketches
   of a photo.

4. **EMA-κ does not rescue the ordering** and slightly *hurts* retrieval
   (R@1 0.095 vs 0.118). Turning κ into a statistic instead of a parameter
   changes its scale but not the qualitative "sketch most concentrated" result.

### Verdict

❌ **On the Sketchy mirror, the vMF per-modality-κ uncertainty story is not
testable as stated and its central prediction does not hold.** The blockers are
dataset-intrinsic (single photo per instance; degenerate generated captions),
not fixable by better optimization. This is a *positive* result for the paper's
dataset argument: **the vMF claim requires (a) real diverse multi-artist sketches
AND (b) real human captions AND (c) multiple photos per concept** — i.e. it needs
true Sketchy sketches *with* FS-COCO-style human text, which no single available
dataset provides. Until then, **do not claim κ as a learned modality-uncertainty
weight**; report it as a diagnostic only.

**Fix filed:** the probe's built-in "H1 HOLDS/FAILS" line compares mismatched
groupings and should compare per-instance sketch vs per-instance photo κ̂ — which
this data cannot provide. Left in with this caveat rather than reporting a false
positive.
