# Trimodal (Sketch–Image–Text) Contrastive Loss via Spherical Geometry

**Status:** working paper / design doc — v0.1 (empirical investigation, results pending)

## 1. Problem statement

Sketch-based image retrieval with a text bridge trains three encoders —
sketch `s`, photo `v`, text `t` — into a shared embedding space. The standard
recipe is a **weighted sum of three symmetric pairwise InfoNCE terms**
(s↔v, s↔t, v↔t), optionally plus a sketch–photo triplet term, on a frozen
CLIP backbone with LoRA/adapters on the sketch path.

Normalized CLIP-style embeddings live on the unit hypersphere S^{d-1}, so
cosine-similarity InfoNCE is *already* a spherical objective. The question this
project asks is narrower and sharper:

> Does making the spherical geometry **explicit at the triple level** — treating
> each concept's (s, v, t) as three points on S^{d-1} with a joint geometric
> objective — beat the plain sum of pairwise terms, which optimizes the three
> angles independently?

We deliberately use **directional statistics** (Fréchet means, geodesic
distances, spherical excess, von Mises–Fisher densities), **not** literal
spherical coordinates φ₁…φ_{d-1}. Hyperspherical coordinates are
singular at the poles and ill-conditioned in high dimensions (the Jacobian
involves products of sines that vanish exponentially in d); every quantity we
need is available coordinate-free from dot products.

## 2. The three loss framings

Throughout, embeddings are L2-normalized: s, v, t ∈ S^{d-1}. Geodesic distance
is the angle: d_g(x, y) = arccos(x·y).

### 2.1 Fréchet-mean anchor (`01_frechet_mean_loss.ipynb`)

For each concept's triple, compute the Fréchet/Karcher mean
m = argmin_μ Σ d_g(μ, x_i)², and pull all three modalities toward m:

    L_frechet = (1/3) Σ_{x ∈ {s,v,t}} d_g(x, m)²

The anchor is the group's own consensus, so **no modality is privileged** —
unlike triplet losses anchored on sketch or text. For a tight cluster the
normalized arithmetic mean approximates the Karcher mean to second order; we
implement both the fast approximation and the true iterative version
(Riemannian gradient descent with exp/log maps) and expose a switch.

Design choices (flagged):
- The anchor **m is detached** from the graph (treated as a target). Letting
  gradients flow through m is valid but makes the term a pure variance
  minimizer that can be satisfied by collapse; the contrastive terms prevent
  collapse either way, but a detached anchor makes the term's role ("pull
  toward current consensus") cleaner and cheaper.
- Used as a **regularizer**: L = L_InfoNCE-sum + λ_f · L_frechet. Alone it has
  no between-concept repulsion.

### 2.2 Spherical-triangle area regularizer (`02_spherical_triangle_loss.ipynb`)

s, v, t are vertices of a geodesic triangle on S^{d-1} (any three points span
a ≤3-dim subspace, so the 2-sphere machinery applies exactly). By Girard's
theorem its area equals the spherical excess E = A + B + C − π. E → 0 when the
three points become collinear on a great circle (or coincide). Minimizing

    L_tri = E(s, v, t)

**couples the three angles** that a pairwise-InfoNCE sum treats independently:
it penalizes "wide" configurations where each pairwise angle is moderate but
the triple is spread in two directions. We compute E with **L'Huilier's
theorem** (numerically stable, works entirely from the three side lengths):

    tan(E/4) = √[ tan(σ/2)·tan((σ−a)/2)·tan((σ−b)/2)·tan((σ−c)/2) ],
    σ = (a+b+c)/2,  a,b,c = geodesic side lengths.

Caveats (flagged):
- E = 0 does **not** imply the three points are close — only coplanar-with-
  origin (near a common great circle). So L_tri is strictly a *shape*
  regularizer and must accompany attraction terms. This is also its interest:
  it adds information pairwise terms don't carry.
- Degenerate triangles (two vertices nearly identical) need clamping;
  implemented with epsilon-clamps inside the tangents and arccos.

### 2.3 von Mises–Fisher with per-modality concentration (`03_vmf_loss.ipynb`) — primary framing

Model each concept c as a vMF distribution on S^{d-1} with shared mean
direction μ_c and **per-modality concentration** κ_m, m ∈ {sketch, photo, text}:

    p(x | μ_c, κ_m) = C_d(κ_m) · exp(κ_m · μ_cᵀ x)

The loss is the negative log-likelihood of each embedding under its concept's
vMF with its modality's κ:

    L_vMF = Σ_m [ −κ_m · μ_cᵀ x_m − log C_d(κ_m) ]

with κ_m **learnable** (softplus-parameterized scalars). This is the framing
we care most about because:

- **Sketch abstraction variance falls out as low κ_sketch.** Five people draw
  the same photo five ways; the sketch cap on the sphere is wider than the
  photo cap. κ_sketch < κ_photo is a *prediction* this framing makes and we
  can test it.
- κ acts as a **learned per-modality uncertainty weight**: the gradient on
  x_m is scaled by κ_m, so noisy modalities are automatically down-weighted —
  replacing hand-tuned loss weights λ_sv, λ_st, λ_vt. (This is exactly the
  spherical analogue of heteroscedastic uncertainty weighting à la
  Kendall & Gal, with −log C_d(κ) as the regularizer that stops κ → 0.)

**Early empirical observation (smoke-scale, Sketchy-HF mirror):** with a
frozen photo/text tower and a trainable sketch path, learned κ came out
*highest* for sketch (κ_s 607 > κ_t 550 > κ_v 534 after 600 steps) — the
opposite of the naive prediction. Mechanism: sketches are the only modality
the optimizer can move, so the vMF term trains them toward the consensus μ
and κ_sketch measures **post-adaptation concentration on training data**, not
intrinsic modality noise. The κ_sketch < κ_photo prediction should therefore
be tested (a) with encoders at rest, or (b) via held-out per-instance κ̂
(Banerjee) on the test split — which the diagnostic section of
`03_vmf_loss.ipynb` provides. Needs confirmation at real scale.

Numerical notes (flagged, important):
- log C_d(κ) involves log I_{d/2−1}(κ), the log modified Bessel function with
  order ν = d/2 − 1 (ν = 255 for d = 512). We implement it with the **uniform
  asymptotic expansion** (Abramowitz & Stegun 9.7.7), which is accurate for
  large ν and differentiable in κ; we validate it against `scipy.special.ive`
  at fp64 in the notebook.
- μ_c per concept: for Sketchy, ≥5 sketches + 1 photo + text per instance let
  us estimate μ_c (and even per-instance κ̂_sketch via Banerjee et al.'s
  approximation κ̂ ≈ R̄(d − R̄²)/(1 − R̄²)) **within a batch**. For
  single-triple datasets, μ_c is just the (Karcher or normalized) mean of the
  triple — degenerate but still usable, with κ estimated at class/global level.
- Like the other two, used alongside InfoNCE: vMF-NLL handles within-concept
  geometry and uncertainty; InfoNCE handles between-concept repulsion. A
  fully generative alternative (vMF mixture with contrastive normalization
  over concepts in the batch) is noted as future work.

### 2.4 Baseline to beat

Weighted sum of three symmetric InfoNCE terms (s↔v, s↔t, v↔t) + an
**adaptive-margin sketch–photo triplet**: margin_ij = m₀ · (1 − cos(t_i, t_j)),
i.e. semantically close concepts (per the text bridge) get a smaller required
margin. (Flagged: this specific adaptive-margin form is our simple
instantiation of the idea; the literature has several variants.)

Backbone: frozen CLIP (open_clip ViT-B/32 by default); **LoRA on the sketch
path only** — same visual tower, low-rank adapters that are toggled on for
sketch forward passes and off for photo forward passes, so photo/text
embeddings stay exactly CLIP's and text remains the zero-shot semantic bridge.

## 3. Dataset selection

### 3.1 The core constraint

**True sketch+photo+text triplets barely exist.** Photo–text pairs exist at
web scale (that's CLIP), and sketch–photo pairs exist at dataset scale, but
datasets where a *human* drew a sketch of a *specific* photo that also has a
*human* caption are nearly a null set. Every design decision below is downstream
of this fact.

### 3.2 Candidate datasets

| Dataset | Size | Sketch type | Text | Triple? | Role here |
|---|---|---|---|---|---|
| **Sketchy** | ~12,500 photos, 125 classes, ≥5 sketches/photo | human, object-level | ✗ (class labels only) | pseudo (generated text) | **prototyping + vMF κ estimation** |
| **FS-COCO** | ~10,000 scene sketches (7k/3k split) | human freehand, scene-level, stroke timing | ✓ human captions | **✓ real** | **primary benchmark** |
| **SketchyCOCO** | ~14k triples | **synthetic** (composed from Sketchy-like parts) | MS-COCO captions | pseudo (synthetic sketch) | secondary, with caveat |
| **CSTBIR** | ~2M composite sketch+text queries, 108k scenes | mixed | ✓ (query text) | query-style | optional scale-up |

**Sketchy** (Sangkloy et al. 2016): ~12,500 photos across 125 object classes
with at least 5 human sketches drawn per photo (~75k sketches). Object-level,
genuinely human-drawn. **No native captions** — text must be weak class-level
prompts ("a photo of a {class}") or generated captions (BLIP-style), both
flagged as pseudo-text. Its killer feature for us: **≥5 sketches per photo
means per-instance κ_sketch is estimable** — Sketchy is the *best* fit for the
vMF framing, which is exactly why we prototype there despite the weak text.

**FS-COCO** (Chowdhury et al. 2022): ~10,000 human freehand *scene* sketches,
each paired with the photo it depicts **and a human caption**, 7k/3k
train/test split, with stroke timing available (unused here, but interesting
for future curriculum/partial-sketch work). This is, to our knowledge, the one
real human trimodal dataset at usable scale → **primary benchmark**; any
headline claim about trimodal losses must hold here. Limitation: **one sketch
per scene**, so κ_sketch is only estimable at class/global level, not
per-instance.

**SketchyCOCO** (Gao et al. 2020): ~14k sketch–photo–caption triples with
MS-COCO captions, but the sketches are **synthetic** — composed by pasting
class-conditional sketch parts into scene layouts — and most scenes contain
fewer than one salient foreground instance's worth of genuine sketch
variation. Use as a **secondary robustness benchmark**, never as the source of
a headline claim, and never for κ_sketch estimation (see §3.4).

**CSTBIR** (composite sketch+text retrieval): ~2M composite sketch+text
queries over ~108k scenes. Query-style rather than triple-style, mixed sketch
provenance. **Optional scale-up** once the losses are validated — useful for
testing whether κ-weighting survives at scale, not for validating it.

### 3.3 The structural tension

The vMF framing's per-instance κ wants Sketchy's multiple-sketches-per-photo
structure; the Fréchet-mean and triangle framings work on any single-triple
dataset. No single dataset serves both goals with real text:

- Sketchy: right *sketch multiplicity*, wrong (absent) *text*.
- FS-COCO: right *text*, sketch multiplicity of 1.

So the design is: **estimate what each dataset can support, claim only what
FS-COCO confirms.** Per-instance κ results come from Sketchy with pseudo-text
(clearly labeled); the trimodal-loss comparison that matters scientifically is
run on FS-COCO with class/global κ.

### 3.4 Synthetic sketches: augmentation only, with a geometric warning

Synthetic sketches (SketchyCOCO's compositions, edge-map conversions,
model-generated sketches) are usable for **pretraining/augmentation only**,
and everything must be **validated and evaluated on real sketches**. The
reason is geometric, and it directly attacks our losses:

> Synthetic sketches form a **tighter, differently-centered spherical cap**
> than real sketches. A generator has one "style"; humans have many. Mixing
> sources therefore (a) **inflates κ_sketch** — the model learns sketches are
> more concentrated than human sketches actually are, defeating the entire
> point of the vMF uncertainty weighting — and (b) **offsets the Fréchet-mean
> / triangle anchor** toward the synthetic cap center, biasing the consensus
> point all three modalities are pulled toward.

Rules encoded in the codebase:
1. **κ_sketch is estimated from REAL sketches only.**
2. **Every sample carries a `source` flag** (`real` / `synthetic`) end-to-end,
   from dataset loader to metrics.
3. **FS-COCO's fully-human triples remain the headline benchmark**, whatever
   synthetic data is used upstream.

### 3.5 Recommended progression

1. **Prototype** all losses on **Sketchy + generated/class captions** (fast,
   object-level, per-instance κ available). ← *this repo's notebooks*
2. **Port to FS-COCO** for the real trimodal claim (human triples, scene-level).
3. **SketchyCOCO** for robustness under domain shift and the synthetic-cap
   diagnostic at scale.

### 3.6 Evaluation

- **Recall@K / Acc.@K** (K ∈ {1, 5, 10}) for sketch→photo retrieval, and
  (sketch+text)→photo for the composite query setting.
- Reported **per-instance** (correct = the exact paired photo) and
  **per-category** (correct = any photo of the same class) — sketch retrieval
  results change character dramatically between these two and both are needed.
- **Synthetic-vs-real cap diagnostic**: per source, estimate the sketch
  embedding mean direction and κ̂ (Banerjee approximation); report (i) the
  **angle between the synthetic and real cap means** and (ii) **κ̂ per
  source**. If synthetic κ̂ ≫ real κ̂ or the cap angle is large, synthetic
  data is geometrically off-manifold and augmentation weights should drop.
  Implemented in `03_vmf_loss.ipynb`.

## 4. Implementation plan (this repo)

```
trimodal-loss/
  paper.md                      ← this document
  lib/                          ← shared module (data, model, sphere ops, losses, training)
  00_data_and_adaptation.ipynb  ← data + frozen CLIP + LoRA sketch path + InfoNCE baseline + retrieval
  01_frechet_mean_loss.ipynb    ← Fréchet/Karcher anchor loss
  02_spherical_triangle_loss.ipynb ← spherical-excess regularizer
  03_vmf_loss.ipynb             ← vMF with learnable per-modality κ + cap diagnostic
  data/                         ← datasets cache (gitignore)
  runs/<notebook>/<timestamp>/  ← every CLI run writes config.json, metrics.json, plots, executed notebook
```

All four notebooks train on the same subset with the same seed, backbone, and
retrieval protocol, so numbers are directly comparable; each writes
`metrics.json` to its own run folder.

## 5. Assumptions & caveats (consolidated)

- **A1 — Pseudo-text on Sketchy.** Class-prompt text is weak; conclusions
  about the text modality from Sketchy are provisional until FS-COCO.
- **A2 — Dataset provenance ladder.** The loader tries, in order: (1) the
  official Sketchy tree if present in `data/`; (2) the Hugging Face mirror
  `justpers/sketchy` — 10,630 real human Sketchy sketch–photo pairs with
  *generated* captions, instance identity recovered by hashing the duplicated
  photo bytes, class labels parsed from captions by a flagged heuristic;
  (3) a gdown attempt at the official Google Drive archive (usually blocked);
  (4) a clearly-labeled procedurally-generated mock dataset (shape "photos" +
  jittered outline "sketches") so the pipeline is smoke-testable end-to-end.
  **No scientific conclusion is valid on mock data**, and mirror-based class
  labels/captions are pseudo-annotations (A1).
- **A3 — Detached Fréchet anchor** (§2.1), **shape-only triangle term**
  (§2.2), **hybrid vMF+InfoNCE** (§2.3) are all deliberate simplifications,
  each flagged where implemented.
- **A4 — Bessel approximation.** log I_ν via uniform asymptotic expansion;
  validated against scipy at fp64, but κ values near 0 with small d would
  need the series expansion instead (not our regime).
- **A5 — Small-subset smoke runs.** Default configs are sized to run on CPU in
  minutes. Real comparisons need the full subset + GPU
  (`pip install torch --index-url https://download.pytorch.org/whl/cu130`
  for the RTX 5070 Ti) and multiple seeds.
- **A6 — Adaptive margin form** (§2.4) is one simple instantiation.
- **A7 — LoRA placement**: adapters on the ViT block MLP linears (c_fc,
  c_proj), sketch-forward only. Attention projections are skipped because
  open_clip's nn.MultiheadAttention reads its projection *parameters*
  functionally — wrapping those modules would silently no-op. qkv-LoRA would
  need a different attention implementation; not swept here.
