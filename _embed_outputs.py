"""Copy the newest executed_<nb>.ipynb (from runs/) over each top-level
notebook, so the committed notebooks carry their outputs — the site is then
rendered WITHOUT re-execution (Quarto execute.enabled=false), exactly like the
xai-starter suite. Run after a full `python run.py all`.
"""
import glob
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBS = ["00_EDA", "00_data_and_adaptation", "01_frechet_mean_loss",
       "02_spherical_triangle_loss", "03_vmf_loss"]

for name in NBS:
    runs = sorted(glob.glob(str(HERE / "runs" / name / "*" / f"executed_{name}.ipynb")))
    if not runs:
        print(f"[skip] no executed copy for {name} — run `python run.py` first")
        continue
    shutil.copy(runs[-1], HERE / f"{name}.ipynb")
    print(f"[embed] {name}.ipynb  <-  {Path(runs[-1]).parent.name}")
