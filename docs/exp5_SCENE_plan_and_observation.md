# EXP5 · SCENE — port to FS-COCO for the real trimodal claim

**Mnemonic:** SCENE = FS-COCO's human *scene* sketches with real captions — the
only genuine sketch+photo+text triplet benchmark.
**Status:** PLAN written 2026-07-09 · OBSERVATION **deferred** (see runtime).
**Est. runtime:** 🔴 **> 1 HOUR — flagged.** Dominated by data acquisition, not
training. Do NOT start unattended without confirming the download budget.

## Why this is the experiment that matters

Everything on Sketchy uses **pseudo-text** (generated or class-prompt captions,
paper.md A1). No claim about the *text* modality — and therefore about the
trimodal losses as trimodal — is real until it holds on human triplets. FS-COCO
is that dataset: ~10,000 human freehand scene sketches, each with the photo it
depicts **and a human caption**, 7k/3k split. It is the headline benchmark in
the design doc's progression (step 2).

## Why it is flagged > 1 hour

1. **Download & storage.** FS-COCO ships the sketches (+ vector/raster + stroke
   timing) and references MS-COCO photos. Pulling the sketch archive and the
   corresponding COCO images is multi-GB and network-bound — realistically
   **30–90 min** on a home connection before a single training step.
2. **New loader.** `lib/data.py` needs an FS-COCO branch: match each sketch to
   its COCO `image_id`, load the human caption, emit `Triple(..., source=
   "real")` at **scene** granularity. One sketch per scene ⇒ per-instance κ is
   **not** estimable here (κ only at class/global level — paper.md §3.3); the
   vMF notebook must fall back to global κ on this dataset.
3. **Preprocessing.** Scene sketches are large line drawings; resizing/caching
   to 224² for CLIP adds a one-time pass over ~10k images.

Training itself, once data is local, is comparable to EXP1 (~40 min for a
multi-seed comparison) — so the >1 h flag is **acquisition + wiring**, not GPU.

## Method (when run)

1. Add `dataset="fscoco"` to `lib/data.py`: download via the FS-COCO release
   (GitHub `pinakinathc/fscoco` / project page), map sketches → COCO photos →
   captions, honor the official 7k/3k split, tag `source="real"`.
2. Re-run all four notebooks with `dataset=fscoco`, scene-level gallery.
3. Report the same metrics as EXP1 **plus** the first *real-text* trimodal
   numbers: composite (sketch+text)→photo vs sketch-only, to see whether the
   text bridge helps more with human captions than with pseudo-text.
4. vMF: global/class-level κ only; the per-instance κ̂ diagnostic is disabled
   with an explicit note (single sketch per scene).

## Decision before running

Because of the >1 h acquisition cost, EXP5 should run **only after** EXP1–EXP4
have shown at least one geometric loss is worth the real-data test — otherwise we
spend the download budget confirming a null result we already have on Sketchy.
Gate: *run EXP5 iff some loss clears the baseline band in EXP2.*

## OBSERVATION (fill after run — currently DEFERRED)

- [ ] Data acquisition time and final on-disk size (record actuals vs the
      estimate above).
- [ ] Real-text trimodal metrics table (all four losses).
- [ ] Does the text bridge help more with human captions than pseudo-text?
- [ ] Global/class κ on FS-COCO (per-instance not available) — sign check only.
- [ ] Final verdict: does any spherical framing beat the baseline on the one
      real human triplet benchmark?
