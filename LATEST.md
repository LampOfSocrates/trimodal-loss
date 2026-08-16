## What this is
Empirical research repo testing whether spherical-geometry loss framings (Fréchet-mean anchor, spherical-triangle excess, vMF with learnable per-modality κ) beat plain pairwise InfoNCE for sketch-image-text retrieval, on frozen CLIP + LoRA sketch path.

## Where it runs
Local dev (Windows, miniconda base env, cu130 torch). Notebooks/CLI (`run.py`) run locally against Sketchy data (HF mirror fallback) or mock data. Published site (rendered notebooks + experiment docs) via GitHub Pages / Quarto: https://lampofsocrates.github.io/trimodal-loss/

## Features
- `lib/`: sphere ops, data loader (official Sketchy / HF mirror / gdown / mock fallback), model, losses (Fréchet, triangle, vMF w/ learned or EMA κ), train/eval
- 4 notebooks (00 data+baseline, 01 Fréchet, 02 triangle, 03 vMF) sharing seed/backbone/protocol, each with best/worst triad diagnostics
- `run.py` CLI, smoke-test mode, per-run results folders under `runs/`
- Quarto site + GitHub Actions publish workflow
- Experiment campaign docs (`docs/exp1-5_*`), plan-then-observation format

## Recently tried
- 2026-07-09: Dropped broken docs/README nav link (Quarto excludes README.md)
- 2026-07-09: EXP2 WEIGH done + 3-seed confirm — Fréchet λ*=0.3 gives modest/seed-sensitive edge (R@5 +2.2, R@1 +1.3, within σ); vMF loses at all λ
- 2026-07-09: EXP1 SCALE done — Fréchet edges baseline on R@5/composite; triangle inert; vMF hurt by λ
- 2026-07-09: EXP3/EXP4 executed — added EMA-κ mode; SHAPE probe shows triangle excess redundant with pairwise s·v (de-prioritize); KAPPA test inconclusive (κ_sketch<κ_photo unsupported, blocked by single-photo/instance + degenerate captions)
- 2026-07-09: Initial commit — paper, lib, notebooks, experiment plans for all 5 experiments

## Next
- EXP5 SCENE (FS-COCO real human triplets) is gated as ready but deferred — >1h due to data download + new loader branch; this is the real trimodal/real-text test still outstanding
- Consider de-prioritizing spherical-triangle loss term per EXP4 finding (redundant signal)
- Campaign headline so far: only Fréchet-mean anchor (λ≈0.3) beats baseline, and only modestly — no strong winner yet
