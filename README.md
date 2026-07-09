# trimodal-loss

Empirical investigation: do spherical-geometry loss framings (Fréchet-mean
anchor, spherical-triangle regularizer, von Mises–Fisher with learnable
per-modality κ) beat a plain sum of pairwise InfoNCE terms for
sketch–image–text retrieval? See **`paper.md`** for the full design doc,
dataset analysis, and consolidated caveats.

📄 **Published site (rendered notebooks + experiment campaign):**
<https://lampofsocrates.github.io/trimodal-loss/>

## Experiment campaign (`docs/`)

One `docs/expN_<MNEMONIC>_plan_and_observation.md` per experiment — plan written
before the run, observation after. Status:

| # | Experiment | Runtime | Status | Headline |
|---|-----------|---------|--------|----------|
| 1 | SCALE — real gallery + seeds | ⚠️ ~34 min | ✅ done | Fréchet ties baseline R@1, **+3–4 pts R@5 & composite**; triangle inert; vMF hurt by λ=0.05 |
| 2 | WEIGH — λ sweep | ⚠️ ~30 min | ✅ done | Fréchet **λ\*=0.3** (inverted-U); 3-seed edge modest (R@5 +2.2, R@1 +1.3 within σ); vMF loses at every λ |
| 3 | KAPPA — honest κ test | ✅ ~7 min | ✅ done | κ_sketch<κ_photo **not** supported; blocked by single-photo/instance + degenerate captions |
| 4 | SHAPE — triangle-excess probe | ✅ ~3 min | ✅ done | E predicts rank (ρ=0.55) but redundant with s·v (partial ρ=−0.22) → **de-prioritize** |
| 5 | SCENE — FS-COCO real triplets | 🔴 **>1 h** | gate met, deferred | download + loader; the real-text test |

**Campaign headline:** the **Fréchet-mean anchor (λ≈0.3)** is the only spherical
framing consistently ≥ the pairwise-InfoNCE baseline on Sketchy — a modest,
seed-sensitive edge (clearest on R@5), not a settled win. Triangle is inert;
vMF's κ story is unsupported here. Real-text verdict awaits EXP5 (FS-COCO).

Reproduce a diagnostic: `python docs/exp4_shape_probe.py --set steps=1200 num_classes=16`.
Aggregate runs: `python docs/_aggregate.py`. Republish site: `bash _publish.sh`.

## Layout

```
paper.md                          design doc / working paper
lib/                              shared code (data, model, sphere ops, losses, training)
00_data_and_adaptation.ipynb      data + frozen CLIP + LoRA sketch path + InfoNCE baseline
01_frechet_mean_loss.ipynb        Fréchet/Karcher-mean anchor loss
02_spherical_triangle_loss.ipynb  spherical-excess (triangle area) regularizer
03_vmf_loss.ipynb                 vMF with learnable per-modality κ + cap diagnostics
run.py                            CLI runner — every run gets its own results folder
data/                             dataset cache (auto-downloaded or mock; gitignored)
runs/<notebook>/<timestamp>/      config.json, metrics.json, history.json, figures,
                                  executed_<notebook>.ipynb
```

## Quick start

```bash
pip install -r requirements.txt

# seconds-long end-to-end smoke test (toy backbone, mock data)
python run.py all --smoke

# real runs: frozen CLIP + LoRA; uses Sketchy if present in data/, else falls
# back to the mock dataset with printed manual-download instructions
python run.py 00
python run.py all --set steps=300 num_classes=16

# any lib/config.py key can be overridden
python run.py 03 --set steps=500 batch_size=48 lambda_vmf=0.1
```

Notebooks are also directly runnable in Jupyter (config cell at the top of
each). All four notebooks share the same data subset, seed, backbone, and
retrieval protocol, so their `metrics.json` files are directly comparable —
each notebook ends with a cross-notebook comparison table.

## Sketchy dataset

The loader tries in order (paper.md A2):

1. official Sketchy tree at `data/sketchy/**/photo/tx_*/...` (manual download
   of **rendered_256x256.7z** from <http://sketchy.eye.gatech.edu/>);
2. the Hugging Face mirror `justpers/sketchy` (~700 MB, auto-downloaded) —
   real human sketches, generated captions, instance groups recovered by
   photo-byte hashing;
3. a gdown attempt at the official Google Drive archive (usually blocked);
4. a clearly-flagged procedural mock dataset — fine for plumbing, meaningless
   for science.

## GPU

An RTX-class GPU is auto-used when a CUDA torch build is installed:
`pip install --force-reinstall --no-deps torch torchvision --index-url
https://download.pytorch.org/whl/cu130` (Blackwell cards need cu128+).
