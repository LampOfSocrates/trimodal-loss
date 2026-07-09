# Experiment campaign

One file per experiment: the **plan is written before** the run, the
**observation is filled in after**. Each is named
`expN_<MNEMONIC>_plan_and_observation.md`. Run **one training+test at a time**
(`python run.py ...`); every run writes its own `runs/<notebook>/<timestamp>/`
with `metrics.json`, so results are aggregated after the fact by
`docs/_aggregate.py`.

| # | Mnemonic | Question | Runtime | Gate |
|---|----------|----------|---------|------|
| 1 | **[SCALE](exp1_SCALE_plan_and_observation.md)** | Can the four losses even be separated at a real gallery size + seeds? | ⚠️ ~40–60 min | — |
| 2 | **[WEIGH](exp2_WEIGH_plan_and_observation.md)** | Does each geometric term have a λ where it beats the baseline? | ⚠️ ~50–60 min | after SCALE |
| 3 | **[KAPPA](exp3_KAPPA_plan_and_observation.md)** | Is κ_sketch < κ_photo once the training confound is removed? | ✅ ~20–30 min | — |
| 4 | **[SHAPE](exp4_SHAPE_plan_and_observation.md)** | Does triangle excess E actually predict retrieval success? | ✅ ~10–15 min | — |
| 5 | **[SCENE](exp5_SCENE_plan_and_observation.md)** | Does any framing beat the baseline on real human triplets (FS-COCO)? | 🔴 **> 1 h** | iff a loss wins EXP2 |

**Runtime flags.** ✅ well under 1 h · ⚠️ borderline, watch it · 🔴 **over 1 h**
(EXP5 — dominated by FS-COCO download + a new loader, not GPU time). EXP5 is
gated: run it only after a Sketchy-scale loss has earned the real-data test.

**Suggested order:** SHAPE and KAPPA first (cheap, decisive diagnostics that can
kill or bless a framing), then SCALE, then WEIGH on whatever survives, then —
only if warranted — SCENE.
