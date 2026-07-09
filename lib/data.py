"""Dataset handling: Sketchy download/loading + procedural mock fallback.

Every sample carries a `source` flag ("real" | "synthetic") end-to-end so
later notebooks can estimate kappa per source (paper.md §3.4 rules: kappa from
REAL sketches only; synthetic usable for augmentation only).

Sketchy layout expected under data/sketchy/ after extraction:
    256x256/photo/tx_000000000000/<class>/<name>.jpg
    256x256/sketch/tx_000000000000/<class>/<name>-<k>.png
"""

import hashlib
import json
import math
import random
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, Sampler

# Historic Google Drive id for rendered_256x256.7z from
# http://sketchy.eye.gatech.edu/ — old-format id, gdown frequently fails on it
# (Drive blocks large-file programmatic download). We try, then fall back.
SKETCHY_GDRIVE_IDS = ["0B7ISyeE8QtDdTjE1MG9Gcy1kSkE"]

MANUAL_INSTRUCTIONS = """
================== MANUAL DOWNLOAD REQUIRED: Sketchy ==================
Automatic download failed (Google Drive blocks large programmatic pulls).

1. Visit the Sketchy Database page:  http://sketchy.eye.gatech.edu/
   and download "Sketches and Photos" (rendered_256x256.7z, ~2 GB), or grab a
   mirror (search Kaggle for "sketchy database rendered_256x256").
2. Extract it so this path exists:
     {root}/sketchy/256x256/photo/tx_000000000000/<class>/*.jpg
     {root}/sketchy/256x256/sketch/tx_000000000000/<class>/*.png
   (py7zr is installed:  python -m py7zr x rendered_256x256.7z {root}/sketchy)
3. Re-run this notebook — it will pick the data up and skip the mock.

Falling back to the PROCEDURAL MOCK dataset so the pipeline stays runnable.
*** No scientific conclusion is valid on mock data (paper.md A2). ***
========================================================================
"""


@dataclass
class Triple:
    sketch_path: Path
    photo_path: Path
    caption: str
    class_id: int
    class_name: str
    instance_id: int   # unique per photo; multiple sketches share it
    source: str        # "real" | "synthetic"


# ----------------------------------------------------------- Sketchy -------

def _find_sketchy_root(data_root: Path) -> Path | None:
    base = data_root / "sketchy"
    hits = sorted(base.glob("**/photo/tx_*")) if base.exists() else []
    return hits[0].parent.parent if hits else None


def try_download_sketchy(data_root: Path) -> Path | None:
    """Attempt gdown pull + 7z extract. Returns extracted root or None."""
    existing = _find_sketchy_root(data_root)
    if existing:
        return existing
    dest = data_root / "sketchy"
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "rendered_256x256.7z"
    try:
        import gdown
        for gid in SKETCHY_GDRIVE_IDS:
            try:
                got = gdown.download(id=gid, output=str(archive), quiet=False)
                if got and archive.exists() and archive.stat().st_size > 1e8:
                    break
            except Exception as e:  # noqa: BLE001 — any Drive failure -> next id
                print(f"[data] gdown id {gid} failed: {e}")
        if archive.exists() and archive.stat().st_size > 1e8:
            import py7zr
            print("[data] extracting rendered_256x256.7z (one-time, slow)…")
            with py7zr.SevenZipFile(archive, mode="r") as z:
                z.extractall(path=dest)
            return _find_sketchy_root(data_root)
    except Exception as e:  # noqa: BLE001
        print(f"[data] sketchy auto-download failed: {e}")
    return None


def load_sketchy_triples(sketchy_root: Path, num_classes: int,
                         max_instances_per_class: int,
                         sketches_per_photo: int) -> list[Triple]:
    """Scan the extracted Sketchy tree into Triples. Text is a weak
    class-prompt (Sketchy has no captions — paper.md A1)."""
    photo_txs = sorted(sketchy_root.glob("photo/tx_*"))
    sketch_txs = sorted(sketchy_root.glob("sketch/tx_*"))
    photo_dir, sketch_dir = photo_txs[0], sketch_txs[0]
    classes = sorted(p.name for p in photo_dir.iterdir() if p.is_dir())
    # deterministic subset: spread across the alphabet rather than first-N
    step = max(1, len(classes) // num_classes)
    classes = classes[::step][:num_classes]

    triples, instance_id = [], 0
    for cid, cls in enumerate(classes):
        caption = f"a photo of a {cls.replace('_', ' ')}"
        photos = sorted((photo_dir / cls).glob("*.jpg"))[:max_instances_per_class]
        for photo in photos:
            sketches = sorted((sketch_dir / cls).glob(photo.stem + "-*.png"))
            for sk in sketches[:sketches_per_photo]:
                triples.append(Triple(sk, photo, caption, cid, cls,
                                      instance_id, source="real"))
            if sketches:
                instance_id += 1
    return triples


# ------------------------------------------------------ HF mirror ----------
# justpers/sketchy: 10,630 real human Sketchy sketch-photo pairs with
# GENERATED captions (pseudo-text, paper.md A1) — the photo is duplicated
# once per sketch, so instance identity is recovered by hashing photo bytes.
# This matches progression step 1 exactly: Sketchy + generated captions.

HF_MIRROR = "justpers/sketchy"


def try_download_sketchy_hf(data_root: Path) -> Path | None:
    root = data_root / "sketchy_hf"
    if (root / "train.jsonl").exists() and (root / "image").exists() \
            and (root / "sketch").exists():
        return root
    try:
        from huggingface_hub import hf_hub_download
        root.mkdir(parents=True, exist_ok=True)
        jl = hf_hub_download(HF_MIRROR, "train.jsonl", repo_type="dataset")
        shutil.copy(jl, root / "train.jsonl")
        for fname, sub in [("sketch.zip", "sketch"), ("image.zip", "image")]:
            if not (root / sub).exists():
                print(f"[data] downloading {HF_MIRROR}/{fname} …")
                zp = hf_hub_download(HF_MIRROR, fname, repo_type="dataset")
                with zipfile.ZipFile(zp) as z:
                    z.extractall(root / sub)
        return root
    except Exception as e:  # noqa: BLE001
        print(f"[data] HF mirror download failed: {e}")
        return None


def _resolve_hf(root: Path, sub: str, rel: str) -> Path:
    for cand in (root / sub / rel, root / sub / Path(rel).name):
        if cand.exists():
            return cand
    raise FileNotFoundError(root / sub / rel)


# Sketchy's 125 category names (multi-word first so they win over substrings).
# Used to recover a class label from the mirror's generated captions.
SKETCHY_CLASSES = [
    "alarm clock", "car (sedan)", "hermit crab", "hot-air balloon",
    "hot air balloon", "jack-o-lantern", "pickup truck", "sea turtle",
    "teddy bear", "wading bird", "wine bottle",
    "airplane", "ant", "ape", "apple", "armor", "axe", "banana", "bat",
    "bear", "bee", "beetle", "bell", "bench", "bicycle", "blimp", "bread",
    "butterfly", "cabin", "camel", "candle", "cannon", "castle", "cat",
    "chair", "chicken", "church", "couch", "cow", "crab", "crocodile",
    "crocodilian", "cup", "deer", "dog", "dolphin", "door", "duck",
    "elephant", "eyeglasses", "fan", "fish", "flower", "frog", "geyser",
    "giraffe", "guitar", "hamburger", "hammer", "harp", "hat", "hedgehog",
    "helicopter", "horse", "hotdog", "hourglass", "jellyfish", "kangaroo",
    "knife", "lion", "lizard", "lobster", "motorcycle", "mouse", "mushroom",
    "owl", "parrot", "pear", "penguin", "piano", "pig", "pineapple",
    "pistol", "pizza", "pretzel", "rabbit", "raccoon", "racket", "ray",
    "rhinoceros", "rifle", "rocket", "sailboat", "saw", "saxophone",
    "scissors", "scorpion", "seagull", "seal", "shark", "sheep", "shoe",
    "skyscraper", "snail", "snake", "songbird", "spider", "spoon",
    "squirrel", "starfish", "strawberry", "swan", "sword", "table", "tank",
    "teapot", "tiger", "tree", "trumpet", "turtle", "umbrella", "violin",
    "volcano", "wheelchair", "windmill", "window", "zebra",
]


def _class_from_prompt(prompt: str) -> str:
    """Heuristic (FLAGGED, paper.md A2): the mirror's captions are generated
    (BLIP-style, noisy) — recover the class as the earliest Sketchy category
    name mentioned in the caption; 'unknown' if none matches."""
    p = " " + prompt.lower() + " "
    best, pos = "unknown", len(p)
    for cls in SKETCHY_CLASSES:
        i = p.find(" " + cls)
        if 0 <= i < pos:
            best, pos = cls.replace(" ", "_"), i
    return best


def load_sketchy_hf_triples(root: Path, num_classes: int,
                            max_instances_per_class: int,
                            sketches_per_photo: int) -> list[Triple]:
    rows = [json.loads(l) for l in
            (root / "train.jsonl").read_text(encoding="utf-8").splitlines() if l]
    # recover instance identity: identical photo bytes => same instance
    cache = root / "instance_map.json"
    if cache.exists():
        hashes = json.loads(cache.read_text())
    else:
        print("[data] hashing photos to recover instance groups (one-time)…")
        hashes = {r["target"]: hashlib.md5(
            _resolve_hf(root, "image", r["target"]).read_bytes()).hexdigest()
            for r in rows}
        cache.write_text(json.dumps(hashes))

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(hashes[r["target"]], []).append(r)

    by_class: dict[str, list[list[dict]]] = {}
    for members in groups.values():
        by_class.setdefault(_class_from_prompt(members[0]["prompt"]),
                            []).append(members)
    # deterministic subset: most-populated classes first; drop caption-parse
    # failures ('unknown') — class labels there are unusable for per-category
    # metrics, and enough cleanly-labeled instances remain
    by_class.pop("unknown", None)
    classes = sorted(by_class, key=lambda c: (-len(by_class[c]), c))[:num_classes]

    triples, instance_id = [], 0
    for cid, cls in enumerate(sorted(classes)):
        for members in by_class[cls][:max_instances_per_class]:
            photo = _resolve_hf(root, "image", members[0]["target"])
            for r in members[:sketches_per_photo]:
                triples.append(Triple(
                    _resolve_hf(root, "sketch", r["source"]), photo,
                    r["prompt"], cid, cls, instance_id, source="real"))
            instance_id += 1
    return triples


# ------------------------------------------------------- mock fallback -----

MOCK_CLASSES = ["circle", "square", "triangle", "star", "cross",
                "hexagon", "arrow", "ring"]


def _shape_points(cls: str, cx, cy, r, rot):
    def poly(n, phase=0.0, radii=None):
        pts = []
        for i in range(n):
            a = rot + phase + 2 * math.pi * i / n
            rr = r if radii is None else radii[i % len(radii)]
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        return pts
    if cls == "square":
        return poly(4, math.pi / 4)
    if cls == "triangle":
        return poly(3, -math.pi / 2)
    if cls == "hexagon":
        return poly(6)
    if cls == "star":
        return poly(10, -math.pi / 2, radii=[r, 0.45 * r])
    if cls == "cross":
        w = 0.35 * r
        raw = [(-w, -r), (w, -r), (w, -w), (r, -w), (r, w), (w, w), (w, r),
               (-w, r), (-w, w), (-r, w), (-r, -w), (-w, -w)]
        cs, sn = math.cos(rot), math.sin(rot)
        return [(cx + x * cs - y * sn, cy + x * sn + y * cs) for x, y in raw]
    if cls == "arrow":
        raw = [(-r, -0.3 * r), (0.2 * r, -0.3 * r), (0.2 * r, -0.7 * r),
               (r, 0), (0.2 * r, 0.7 * r), (0.2 * r, 0.3 * r), (-r, 0.3 * r)]
        cs, sn = math.cos(rot), math.sin(rot)
        return [(cx + x * cs - y * sn, cy + x * sn + y * cs) for x, y in raw]
    return None  # circle / ring drawn as ellipses


def _draw_mock_photo(cls, rng, size=256):
    bg = tuple(rng.randint(140, 230) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    dr = ImageDraw.Draw(img)
    for _ in range(60):  # light texture so photos aren't flat color
        x, y = rng.randint(0, size), rng.randint(0, size)
        dr.point((x, y), fill=tuple(min(255, c + rng.randint(-25, 25)) for c in bg))
    cx, cy = rng.randint(90, size - 90), rng.randint(90, size - 90)
    r = rng.randint(45, 80)
    rot = rng.uniform(0, math.pi)
    color = tuple(rng.randint(0, 130) for _ in range(3))
    pts = _shape_points(cls, cx, cy, r, rot)
    if pts:
        dr.polygon(pts, fill=color)
    elif cls == "ring":
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=int(r * 0.3))
    else:
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return img, (cx, cy, r, rot)


def _draw_mock_sketch(cls, geom, rng, jitter, size=256):
    """jitter>0 -> 'human-like' wobbly strokes (source=real proxy);
    jitter=0 -> clean perfect outline (source=synthetic proxy). The clean
    style is deliberately a tighter cap — it lets 03's synthetic-vs-real
    diagnostic demonstrate the kappa-inflation effect on mock data."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    cx, cy, r, rot = geom
    # humans re-center and re-scale; synthetic keeps exact geometry
    if jitter:
        cx += rng.uniform(-15, 15); cy += rng.uniform(-15, 15)
        r *= rng.uniform(0.8, 1.25); rot += rng.uniform(-0.25, 0.25)
    pts = _shape_points(cls, cx, cy, r, rot)
    width = rng.randint(2, 4) if jitter else 3

    def wobble(p):
        return (p[0] + rng.uniform(-jitter, jitter),
                p[1] + rng.uniform(-jitter, jitter))
    if pts:
        pts = [wobble(p) for p in pts] + [wobble(pts[0])]
        dr.line(pts, fill=(0, 0, 0), width=width, joint="curve")
    else:
        n = 24
        ring = [wobble((cx + r * math.cos(2 * math.pi * i / n),
                        cy + r * math.sin(2 * math.pi * i / n)))
                for i in range(n + 1)]
        dr.line(ring, fill=(0, 0, 0), width=width, joint="curve")
        if cls == "ring":
            r2 = r * 0.55
            ring2 = [wobble((cx + r2 * math.cos(2 * math.pi * i / n),
                             cy + r2 * math.sin(2 * math.pi * i / n)))
                     for i in range(n + 1)]
            dr.line(ring2, fill=(0, 0, 0), width=width, joint="curve")
    return img


def build_mock_dataset(data_root: Path, num_classes: int,
                       max_instances_per_class: int,
                       sketches_per_photo: int, seed: int) -> list[Triple]:
    """Generate (once, deterministic) a mock Sketchy-like tree on disk.
    Each photo gets `sketches_per_photo` jittered 'real' sketches plus one
    clean 'synthetic' sketch, so per-source diagnostics have both sources."""
    rng = random.Random(seed)
    root = data_root / "mock_sketchy"
    marker = root / f"v1_{num_classes}c_{max_instances_per_class}i_{sketches_per_photo}s_{seed}.done"
    classes = MOCK_CLASSES[:num_classes]
    triples, instance_id = [], 0
    regenerate = not marker.exists()
    for cid, cls in enumerate(classes):
        (root / "photo" / cls).mkdir(parents=True, exist_ok=True)
        (root / "sketch" / cls).mkdir(parents=True, exist_ok=True)
        caption = f"a photo of a {cls} shape"
        for i in range(max_instances_per_class):
            ppath = root / "photo" / cls / f"{cls}_{i}.png"
            photo, geom = _draw_mock_photo(cls, rng)  # keep rng in sync always
            if regenerate:
                photo.save(ppath)
            sk_specs = [(k, 6.0, "real") for k in range(sketches_per_photo)]
            sk_specs.append((sketches_per_photo, 0.0, "synthetic"))
            for k, jitter, source in sk_specs:
                spath = root / "sketch" / cls / f"{cls}_{i}-{k}.png"
                sk = _draw_mock_sketch(cls, geom, rng, jitter)
                if regenerate:
                    sk.save(spath)
                triples.append(Triple(spath, ppath, caption, cid, cls,
                                      instance_id, source))
            instance_id += 1
    if regenerate:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    return triples


# ----------------------------------------------------------- dataset -------

def prepare_triples(cfg) -> tuple[list[Triple], str]:
    """Returns (triples, dataset_name_actually_used)."""
    data_root = Path(cfg["data_root"])
    if cfg["dataset"] == "sketchy":
        root = _find_sketchy_root(data_root)
        if root:
            print(f"[data] using official Sketchy tree at {root}")
            return load_sketchy_triples(root, cfg["num_classes"],
                                        cfg["max_instances_per_class"],
                                        cfg["sketches_per_photo"]), "sketchy"
        hf = try_download_sketchy_hf(data_root)
        if hf:
            print(f"[data] using Sketchy HF mirror ({HF_MIRROR}) at {hf} — "
                  "real human sketches, GENERATED captions (pseudo-text)")
            return load_sketchy_hf_triples(hf, cfg["num_classes"],
                                           cfg["max_instances_per_class"],
                                           cfg["sketches_per_photo"]), "sketchy-hf"
        root = try_download_sketchy(data_root)
        if root:
            return load_sketchy_triples(root, cfg["num_classes"],
                                        cfg["max_instances_per_class"],
                                        cfg["sketches_per_photo"]), "sketchy"
        print(MANUAL_INSTRUCTIONS.format(root=data_root))
    triples = build_mock_dataset(data_root, cfg["num_classes"],
                                 cfg["max_instances_per_class"],
                                 cfg["sketches_per_photo"], cfg["seed"])
    return triples, "mock"


def split_triples(triples: list[Triple], test_frac=0.25):
    """Deterministic instance-level split (a photo and ALL its sketches land
    on the same side — no leakage)."""
    def is_test(t):
        h = hashlib.md5(f"{t.class_name}/{t.photo_path.stem}".encode()).digest()
        return h[0] / 255.0 < test_frac
    train = [t for t in triples if not is_test(t)]
    test = [t for t in triples if is_test(t)]
    return train, test


def summarize_triples(triples: list[Triple]) -> dict:
    """Counts for EDA: totals, per-class, per-source, sketches-per-instance."""
    from collections import Counter
    per_class = Counter(t.class_name for t in triples)
    per_source = Counter(t.source for t in triples)
    sk_per_inst = Counter()
    inst_class = {}
    for t in triples:
        sk_per_inst[t.instance_id] += 1
        inst_class[t.instance_id] = t.class_name
    hist = Counter(sk_per_inst.values())
    return dict(
        n_triples=len(triples),
        n_instances=len(sk_per_inst),
        n_classes=len(per_class),
        per_class=dict(per_class.most_common()),
        per_source=dict(per_source),
        sketches_per_instance_hist=dict(sorted(hist.items())),
        instances_per_class=dict(Counter(inst_class.values()).most_common()),
    )


class TriDataset(Dataset):
    def __init__(self, triples: list[Triple], transform):
        self.triples = triples
        self.transform = transform

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, i):
        t = self.triples[i]
        return dict(
            sketch=self.transform(Image.open(t.sketch_path).convert("RGB")),
            photo=self.transform(Image.open(t.photo_path).convert("RGB")),
            caption=t.caption,
            class_id=t.class_id,
            instance_id=t.instance_id,
            source=t.source,
        )


def collate(batch):
    return dict(
        sketch=torch.stack([b["sketch"] for b in batch]),
        photo=torch.stack([b["photo"] for b in batch]),
        caption=[b["caption"] for b in batch],
        class_id=torch.tensor([b["class_id"] for b in batch]),
        instance_id=torch.tensor([b["instance_id"] for b in batch]),
        source=[b["source"] for b in batch],
    )


class GroupedInstanceSampler(Sampler):
    """Batches = (instances_per_batch groups) x (triples_per_instance members
    of the same instance). Gives vMF per-instance mu/kappa something to
    estimate within a batch — this is exactly the structure Sketchy's >=5
    sketches/photo provides (paper.md §3.3)."""

    def __init__(self, dataset: TriDataset, instances_per_batch: int,
                 triples_per_instance: int, seed: int = 0):
        self.groups = {}
        for i, t in enumerate(dataset.triples):
            self.groups.setdefault(t.instance_id, []).append(i)
        self.ipb, self.tpi = instances_per_batch, triples_per_instance
        self.rng = random.Random(seed)

    def __iter__(self):
        ids = list(self.groups)
        self.rng.shuffle(ids)
        for j in range(0, len(ids) - self.ipb + 1, self.ipb):
            batch = []
            for gid in ids[j:j + self.ipb]:
                members = self.groups[gid]
                take = min(self.tpi, len(members))
                batch.extend(self.rng.sample(members, take))
            yield batch

    def __len__(self):
        return max(1, len(self.groups) // self.ipb)
