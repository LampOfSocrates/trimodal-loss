# EXP2 · WEIGH — sweep the geometric-term weight before trusting any ranking

**Mnemonic:** WEIGH = weight the geometric term (λ) correctly; a loss judged at
one λ is not judged at all.
**Status:** PLAN written 2026-07-09 · OBSERVATION pending.
**Est. runtime:** ⚠️ **borderline ~50–60 min** (up to 18 runs). See timing note.

## Motivation

Each geometric loss is `L = L_pairwise-InfoNCE + λ · L_geo`. The committed runs
use a single hand-picked λ per loss (`lambda_frechet=1`, `lambda_triangle=1`,
`lambda_vmf=0.05`). That is a confound: vMF's apparent R@1 drop may simply be λ
too large — with κ≈600 the vMF gradient can be ~30× the InfoNCE gradient scale,
drowning the contrastive term. A loss that only helps at one λ, or only hurts at
one λ, has not been fairly evaluated.

## Hypothesis

Each geometric term has a non-trivial λ* > 0 where it beats λ=0 (pure baseline)
on R@1-instance. If the best any loss can do is λ→0 (i.e. "add nothing"), that is
itself the finding — the explicit geometry is inert once InfoNCE is tuned.

## Method

- Fix the best data scale from EXP1 (or `num_classes=16, instances=80,
  steps=1500` if EXP1 not yet run).
- Sweep, per loss:
  - Fréchet: `lambda_frechet ∈ {0.1, 0.3, 1.0}`
  - Triangle: `lambda_triangle ∈ {0.1, 0.3, 1.0}`
  - vMF: `lambda_vmf ∈ {0.002, 0.01, 0.05}`
- Include the **λ=0 baseline point** once (that is just notebook 00).
- Seeds `{0, 1}` to keep the count at 3 losses × 3 λ × 2 seeds = **18 runs**.
- Plot R@1-instance vs λ per loss, with the baseline as a horizontal band
  (mean ± std). A loss "wins" only if its best λ clears the baseline band.

## Commands

```bash
for s in 0 1; do
  for L in 0.1 0.3 1.0; do python run.py 01 --set lambda_frechet=$L  steps=1500 seed=$s num_classes=16 max_instances_per_class=80; done
  for L in 0.1 0.3 1.0; do python run.py 02 --set lambda_triangle=$L steps=1500 seed=$s num_classes=16 max_instances_per_class=80; done
  for L in 0.002 0.01 0.05; do python run.py 03 --set lambda_vmf=$L   steps=1500 seed=$s num_classes=16 max_instances_per_class=80; done
done
```

## Timing note (⚠️ >1 h risk)

18 runs × ~2.5–3 min ≈ 45–55 min. If it drifts past 1 h, cut to seed `{0}` (9
runs, ~25 min) and treat λ curves as single-seed exploratory, then confirm the
winning λ with a 3-seed run. The embedding-cache optimization from EXP1 applies
here too and would roughly halve the time.

## Confounds controlled

- Same seed/data/steps across all λ for a given loss → only λ varies.
- κ learning-rate held fixed for vMF (else λ and κ-lr interact).
- The λ=0 point is the *actual* baseline notebook, not a re-implementation.

## OBSERVATION (fill after run)

- [ ] λ-vs-R@1 curve per loss (figure) with baseline band.
- [ ] Best λ* per loss and whether it clears the baseline band.
- [ ] Is vMF's drop a λ artifact (recovers at small λ) or intrinsic?
- [ ] Updated recommendation for the headline comparison config.
