# EXP4 · SHAPE — does spherical-triangle excess carry retrieval signal?

**Mnemonic:** SHAPE = the triangle term is a *shape* regularizer; check the
shape actually correlates with retrieval before trusting it.
**Status:** PLAN written 2026-07-09 · OBSERVATION pending.
**Est. runtime:** ✅ **~10–15 min** (well under 1 h) — one trained model + a
cheap per-triple correlation, no sweep.

## Motivation

The triangle loss minimizes the geodesic-triangle area E (Girard excess) of each
(s, v, t) triple, on the theory that a small E (near-collinear on a great circle)
is a "better" configuration. But E = 0 only means *coplanar-with-origin*, **not**
that the three points are close — the paper flags this. Before spending sweep
budget (EXP2) on the triangle loss, run the cheap decisive check: **does E
actually predict retrieval success per triple?** If triples the model retrieves
correctly have systematically smaller E than triples it misses, the term is
tracking something real. If E is uncorrelated with success, the regularizer is
pushing on a quantity that doesn't matter and we should say so and de-prioritize
it.

## Hypothesis

Among held-out triples, **E is negatively associated with retrieval success**:
correctly-retrieved (rank-0) triples have smaller mean E than missed triples,
and per-triple E correlates positively with retrieval rank (bigger triangle →
worse rank).

## Method

On a trained **baseline** model (so the diagnostic is not circular — we do *not*
use a triangle-trained model, which would trivially have small E):
1. Embed the test triples; for each compute E(s, v, t) and its sketch→photo
   rank of the correct photo.
2. Report: mean E for rank-0 (hits) vs rank>0 (misses); Spearman correlation
   between E and rank; and E for the 10 best vs 10 worst triads from the
   notebook's qualitative section.
3. Control: repeat with the pairwise angles (s·v, s·t, v·t) to check whether E
   adds anything **beyond** what the pairwise similarities already say (partial
   correlation of E with rank, controlling for s·v).

A tiny standalone script `docs/exp4_shape_probe.py` implements this using the
shared `lib` utilities (`spherical_triangle_excess`, `rank_test_queries`).

## Why this is decisive and cheap

No training sweep: one baseline model, one pass over the test set, a correlation.
It can *kill* the triangle framing early (if E ⟂ rank) or justify sweeping it in
EXP2 (if E predicts rank), for ~10 min of compute.

## OBSERVATION — run 2026-07-09 (baseline model, 1200 steps, 16 classes, 400 test queries)

Config: `steps=1200, num_classes=16, max_instances_per_class=80,
sketches_per_photo=5, batch_size=48`, HF Sketchy mirror. Baseline loss only
(non-circular). Raw numbers in `docs/exp4_shape_result.json`. **Train time
~2.8 min** (0.14 s/step at batch 48 — see the EXP1 timing implication below).

**Result:**

| quantity | value |
|---|---|
| top-1 hit rate | 0.228 |
| mean excess E, **hits** (rank 0) | **0.779** |
| mean excess E, **misses** (rank>0) | **0.838** |
| Spearman(E, rank), raw | **+0.555** (p ≈ 1e-34) |
| **partial** Spearman(E, rank) **given s·v** | **−0.22** |

**Interpretation — the headline (raw) correlation is a confound.** Marginally,
bigger triangles go with worse ranks (ρ=+0.55, hits have visibly smaller E than
misses), which at face value blesses the triangle term. **But** once we control
for the sketch–photo similarity s·v, the partial correlation **flips to −0.22**:
E is large precisely when s·v is small (a bad retrieval), and the pairwise term
*already* optimizes s·v. After removing that shared dependence, higher E is if
anything *mildly associated with better* rank. So E carries **almost no
independent signal** for retrieval beyond what the pairwise sketch–photo term
captures — and its residual sign is the opposite of what the regularizer assumes.

(The script's built-in verdict string — "E PREDICTS rank" — is triggered by the
crude `|partial_rho|>0.05` threshold and **mis-reads the sign**; the partial
correlation is negative. Corrected verdict below.)

**Verdict:** ⚠️ **De-prioritize the triangle term.** It is not free of value —
E does track configuration quality marginally — but it is largely redundant with
the pairwise sketch–photo similarity, and its *independent* contribution points
the wrong way. In EXP2, sweep it **last** and with low expectations; do not spend
multi-seed budget on it before Fréchet and vMF. This is exactly the early-kill
the cheap diagnostic was designed to deliver.

**Caveats:** pseudo-text captions (A1); single baseline seed; s·v is only one of
three pairwise angles — a fuller control would residualize on all of (s·v, s·t,
v·t). The partial-correlation sign is robust enough to act on, but the magnitude
should be reconfirmed at EXP1 scale with multiple seeds.
