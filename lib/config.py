"""Central config with environment-variable overrides.

Every notebook has a config cell that calls get_config(); the CLI runner
(run.py) passes overrides via TRIMODAL_* env vars so notebooks stay runnable
both interactively and headless.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = dict(
    # --- data ---
    dataset="sketchy",            # "sketchy" (falls back to mock if absent) | "mock"
    data_root=str(PROJECT_ROOT / "data"),
    num_classes=8,                # subset of classes (Sketchy has 125)
    max_instances_per_class=12,   # photos per class in the subset
    sketches_per_photo=3,         # cap sketches used per photo (Sketchy has >=5)
    image_size=224,
    # --- model ---
    backbone="clip",              # "clip" | "toy" (tiny random nets, pure smoke test)
    clip_model="ViT-B-32",
    clip_pretrained="openai",
    lora_rank=8,
    lora_alpha=16.0,
    # --- training ---
    seed=42,
    device="auto",
    batch_size=24,
    steps=60,
    lr=1e-3,
    temperature=0.07,
    # pairwise InfoNCE weights (s-v, s-t, v-t) shared by every loss variant
    w_sv=1.0, w_st=0.5, w_vt=0.25,
    # baseline adaptive-margin triplet
    triplet_margin0=0.2, w_triplet=0.5,
    # loss-specific weights
    lambda_frechet=1.0, karcher_exact=True, karcher_iters=8,
    lambda_triangle=1.0,
    lambda_vmf=0.05, kappa_init=100.0,
    # kappa gets its own Adam param group: with a shared lr=1e-3 the raw-kappa
    # step is ~1e-3/step (Adam normalizes gradient scale), i.e. frozen in
    # practice. Its equilibrium solves A_d(kappa) = mean alignment — for CLIP
    # embeddings that's kappa in the hundreds, so init/lr must let it get there
    kappa_lr=1.0,
    # vMF grouped batching: instances per batch x triples per instance
    vmf_instances_per_batch=8, vmf_triples_per_instance=3,
    # --- eval ---
    recall_ks=(1, 5, 10),
    eval_max_queries=400,
)

_TYPES = {k: type(v) for k, v in DEFAULTS.items()}


def _coerce(key, raw):
    t = _TYPES.get(key, str)
    if t is bool:
        return raw.lower() in ("1", "true", "yes")
    if t is tuple:
        return tuple(int(x) for x in raw.split(","))
    return t(raw)


def get_config(**overrides) -> dict:
    """DEFAULTS < TRIMODAL_<KEY> env vars < explicit overrides."""
    cfg = dict(DEFAULTS)
    for key in cfg:
        raw = os.environ.get("TRIMODAL_" + key.upper())
        if raw is not None:
            cfg[key] = _coerce(key, raw)
    cfg.update(overrides)
    if os.environ.get("TRIMODAL_SMOKE") == "1":
        # fastest possible end-to-end path (CI-style); results meaningless
        cfg.update(dataset="mock", backbone="toy", steps=15, batch_size=16,
                   num_classes=6, max_instances_per_class=8, image_size=64)
    return cfg
