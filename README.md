# trimodal-loss

Empirical investigation: do spherical-geometry loss framings (Fréchet-mean
anchor, spherical-triangle regularizer, von Mises–Fisher with learnable
per-modality κ) beat a plain sum of pairwise InfoNCE terms for
sketch–image–text retrieval? See **`paper.md`** for the full design doc,
dataset analysis, and consolidated caveats.

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
