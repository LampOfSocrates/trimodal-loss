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

## OBSERVATION (fill after run)

- [ ] Table: global κ̂ and mean per-instance κ̂ per modality (held-out).
- [ ] Is κ̂_sketch < κ̂_photo on held-out data? (the honest H1 test)
- [ ] Before/after adaptation: does LoRA training narrow the sketch cap?
- [ ] EMA-κ vs learned-κ: κ ordering and R@1 for each.
- [ ] Verdict on whether learned-κ is a usable uncertainty weight here.
