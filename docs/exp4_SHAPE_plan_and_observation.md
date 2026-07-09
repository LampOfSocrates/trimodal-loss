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

## OBSERVATION (fill after run)

- [ ] Mean E: hits vs misses (with counts).
- [ ] Spearman(E, rank) and its p-value.
- [ ] Partial correlation of E with rank controlling for s·v — does E add
      signal beyond the pairwise sketch–photo similarity?
- [ ] Verdict: sweep the triangle loss in EXP2, or de-prioritize it?
