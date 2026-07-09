# Experiment campaign

One file per experiment: the **plan is written before** the run, the
**observation is filled in after**. Each is named
`expN_<MNEMONIC>_plan_and_observation.md`. Run **one training+test at a time**
(`python run.py ...`); every run writes its own `runs/<notebook>/<timestamp>/`
with `metrics.json`, so results are aggregated after the fact by
`docs/_aggregate.py`.

| # | Mnemonic | Question | Runtime | Gate |
|---|----------|----------|---------|------|
| 1 | **[SCALE](exp1_SCALE_plan_and_observation.md)** ✅ | Can the four losses even be separated at a real gallery size + seeds? | ~34 min | **Fréchet** ties R@1, +3–4pts R@5 & composite; triangle inert; vMF hurt by λ |
| 2 | **[WEIGH](exp2_WEIGH_plan_and_observation.md)** ✅ | Does each geometric term have a λ where it beats the baseline? | ~30 min | Fréchet **λ\*=0.3** (inverted-U); 3-seed edge small (R@1 +1.3 within σ, R@5 +2.2); vMF loses at every λ |
| 3 | **[KAPPA](exp3_KAPPA_plan_and_observation.md)** ✅ | Is κ_sketch < κ_photo once the training confound is removed? | ~7 min | **No** — not supported by any estimator; test blocked by the data |
| 4 | **[SHAPE](exp4_SHAPE_plan_and_observation.md)** ✅ | Does triangle excess E actually predict retrieval success? | ~3 min | Marginally, but redundant with s·v → de-prioritized |
| 5 | **[SCENE](exp5_SCENE_plan_and_observation.md)** ⏸️ | Does any framing beat the baseline on real human triplets (FS-COCO)? | 🔴 **> 1 h** | **gate now MET** (Fréchet won EXP2); still deferred — download + loader |

## Campaign result (so far)

**The Fréchet-mean consensus anchor (λ≈0.3) is the only spherical framing that is
consistently ≥ the pairwise-InfoNCE baseline** on Sketchy (mirror), with a clean
inverted-U λ response. The edge is **modest and seed-sensitive at 3 seeds**
(R@5 instance +2.2 pts fairly consistent; R@1 +1.3 pts within one σ) — a
*promising* signal, not a settled win. The **spherical-triangle** term is inert
(redundant with the pairwise angles; EXP1+EXP4), and the **vMF** framing neither
supports its κ prediction (EXP3) nor helps retrieval at any λ (EXP2) on this
data. Firming up the Fréchet edge needs more seeds + a larger gallery; the
real-text verdict awaits **EXP5 / FS-COCO** (gate met, >1 h, deferred).

**Runtime flags.** ✅ well under 1 h · ⚠️ borderline, watch it · 🔴 **over 1 h**
(EXP5 — dominated by FS-COCO download + a new loader, not GPU time). EXP5 is
gated: run it only after a Sketchy-scale loss has earned the real-data test.

**Suggested order:** SHAPE and KAPPA first (cheap, decisive diagnostics that can
kill or bless a framing), then SCALE, then WEIGH on whatever survives, then —
only if warranted — SCENE.
