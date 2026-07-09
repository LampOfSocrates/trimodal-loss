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

## OBSERVATION — run 2026-07-09 (16 classes, 1500 steps, seed 0 sweep + seeds 1–2 confirm)

Config: `num_classes=16, max_instances_per_class=80, sketches_per_photo=5,
steps=1500, batch_size=48`. Triangle **skipped** (EXP1+EXP4 agree it's inert).
The λ-vs-λ sweep below is **seed 0** (single-seed trend); the winner is then
re-run at seeds 1–2 (`.exp2b`) for a 3-seed headline.

### λ sweep (seed 0)

| loss | λ | R@1 inst | R@5 inst | R@1 cat | (s+t)→p R@1 |
|---|---|---|---|---|---|
| baseline | — | 0.205 | 0.560 | 0.965 | 0.302 |
| Fréchet | 0.1 | 0.215 | 0.560 | 0.965 | 0.307 |
| **Fréchet** | **0.3** | **0.237** | **0.610** | 0.975 | 0.312 |
| Fréchet | 1.0 | 0.205 | 0.598 | 0.968 | 0.282 |
| vMF | 0.002 | 0.170 | 0.490 | 0.973 | 0.287 |
| vMF | 0.01 | 0.165 | 0.492 | 0.978 | **0.315** |
| vMF | 0.05 | 0.115 | 0.393 | 0.907 | 0.233 |

### Findings

1. **Fréchet has a clean inverted-U with λ\* = 0.3** — the textbook signature of
   a well-behaved regularizer weight. At λ=0.3 it beats baseline on **R@1
   instance (+3.2 pts, 0.237 vs 0.205, +16% rel.)** and **R@5 instance (+5.0
   pts, 0.610 vs 0.560)**. Too small (0.1) is too weak; too large (1.0) collapses
   back to / below baseline. **This is the campaign's one clear positive result.**
2. **vMF's damage is NOT a pure λ artifact.** Shrinking λ from 0.05 → 0.002
   *reduces* the harm (R@1 0.115 → 0.170) but **never reaches baseline** (0.205)
   at any tested λ. So the vMF NLL term, as formulated (hybrid with detached μ +
   learned κ), is net-negative for sketch→photo instance retrieval across the
   whole λ range. It does *slightly* help the **composite text query** at λ=0.01
   (s+t R@1 0.315 > 0.302) — consistent with it doing something to the text
   geometry — but it loses the headline metric everywhere. Combined with EXP3
   (κ prediction unsupported), the vMF framing is **not working on this data**.
3. Category retrieval stays saturated (~0.97) except where vMF over-weights.

### Verdict & recommendation

- **Headline config: Fréchet-mean anchor at λ_frechet = 0.3.** It is the only
  spherical framing that beats the pairwise-InfoNCE baseline on the headline
  metric. Being confirmed at 3 seeds (see below).
- vMF: drop from the retrieval comparison as-is; revisit only with a different
  formulation (e.g. contrastive-normalized vMF mixture, or κ from real
  multi-artist sketches) — not a λ-tuning fix.
- Triangle: closed out (inert).
- **EXP5 gate is now MET** (a loss — Fréchet — clears the baseline band), so the
  FS-COCO real-triplet test is *justified*. It remains 🔴 **>1 h** (download +
  new loader) and is therefore left as the explicit next step, not auto-run.

### 3-seed confirmation of Fréchet λ=0.3 vs baseline — **tempers the headline**

Seeds {0,1,2}, same config:

| loss | R@1 inst (3 seeds) | raw R@1 per seed | R@5 inst (3 seeds) |
|---|---|---|---|
| baseline | 0.207 ± 0.002 | 0.205 / 0.210 / 0.205 | 0.574 ± 0.010 |
| Fréchet λ=0.3 | **0.220 ± 0.015** | **0.237** / 0.200 / 0.223 | **0.596 ± 0.014** |

**Honest reading:** the seed-0 R@1 of 0.237 that looked like a clean +3.2 pt win
was **an optimistic draw** — seeds 1,2 give 0.200 and 0.223. Over 3 seeds:

- **R@1 instance: +1.3 pts (0.220 vs 0.207), but within ~1 Fréchet-σ** (σ=0.015).
  Not a decisive win at 3 seeds.
- **R@5 instance: +2.2 pts (0.596 vs 0.574), ~1.5× pooled σ** — the more
  consistent signal, positive in the aggregate.

So the tempered conclusion: **Fréchet λ=0.3 is the best of the three framings and
shows a small, mostly-R@5 edge over baseline, but the R@1 margin is seed-sensitive
and not yet significant.** It is a *promising* signal worth the real-data test,
not a settled win. Firm it up with (a) more seeds, (b) a larger gallery (raise
`eval_max_queries`), (c) FS-COCO (EXP5). The single-seed λ curve (inverted-U with
peak at 0.3) is still the right λ; the *magnitude* of the gain was overstated by
one lucky seed.

### Caveats

- Sweep is single-seed; the confirmation run addresses only the winner. Other
  λ points are point estimates.
- Pseudo-text (A1); gallery 80; `steps=1500`. Absolute numbers are not
  publication-grade — the *relative* λ response is the result.
