# Formation Recognition — Training Plan

## Context

The Hudl AI desktop app already has an offensive-formation classifier
(`models/offense_positions/formModel.pt`, an MLP trained on 11 normalized
(x, y) snap-frame positions → 24 classes). Two problems block it from being
useful:

1. **Data is the bottleneck.** Only ~89 labeled clips spread across 24 classes
   means most classes have 1–4 examples. The model is severely under-trained,
   and `models/offense_positions/play_predictions.csv` confirms heavy skew
   (DETROIT / TRIPS OPEN dominate, many predictions <0.3 confidence).
2. **The architecture and inputs are weak.** A flat MLP on raw (x, y) is
   sensitive to player ordering, ignores pre-snap motion entirely, and there
   is no defensive-formation model at all.

Goal: end up with **(a) a labeling workflow to grow the dataset from ~89 to
500–1000+ clips, (b) a permutation-invariant offense model with hierarchical
labels, (c) a pre-snap motion encoder, and (d) a parallel defensive
formation model**, all wired into `scripts/staticProcess.py` and the CSV
output the app already consumes.

User-confirmed scope (all of):
- Fix the existing offense model
- Add defensive formation model
- Use pre-snap motion / temporal info
- Redesign architecture (set-based / GNN)

User-confirmed main blocker: **not enough labeled data**, with no
additional labels available beyond the current ~89 clips.

---

## Phase 1 — Grow the labeled dataset (the unblock)

Without more data, no architectural change matters. This phase is the
biggest investment.

### 1.1 Build a labeling tool
- New file: `modelTraining/labeling_tool.py` (PySide6, reusing widgets from `app/`).
- Inputs per clip, already produced by the existing pipeline:
  - snap-frame detections from `cache/<video>/snap_detections.json`
  - field-projected points from `scripts/perFrameHomographyTransform.py`
- UI: draw the 11 offense + 11 defense points on a normalized field
  (53.33 × 100 yd). Two label fields per clip:
  - **Offense:** base formation (DALLAS / TREY / EMPTY / SLOT / TRIPS /
    DENVER / DETROIT …) **plus** variation tags (Y OFF, U OFF, TITE,
    WING, OPEN — multi-select).
  - **Defense:** front (3-4 / 4-3 / Bear / 46 / Even / Odd / Nickel /
    Dime) **plus** coverage shell (C0 / C1 / C2 / C3 / C4 / C2-Man /
    C-Match).
- Output: appends rows to
  `cache/<video>/offense_positions.csv` and a new
  `cache/<video>/defense_positions.csv`.

### 1.2 Active-learning ordering
- Run the current offense model over all unlabeled snaps; sort by
  ascending top-1 confidence and present **low-confidence clips first**.
  This concentrates human effort on the examples the model is most
  confused by.
- Also surface clips whose top-2 classes are within 0.05 (boundary
  cases — the most useful supervision).

### 1.3 Pseudo-label / cluster-assist
- KMeans (k≈15) on the existing 29-D geometric feature vector → present
  one cluster at a time so the user labels visually-similar plays in
  batches (5–10× speedup vs. clip-by-clip).
- Reuse the feature code in
  `modelTraining/train_offense_positions.py` (centroid-centered features,
  pairwise distances, PCA eigenvalues) — do **not** re-implement.

### 1.4 Augmentation (free data multiplier)
- Mirror plays left↔right (flip x around field centerline). Doubles the
  dataset and matches real symmetry of football formations.
- Yard-line translation jitter (±10 yd along x); small Gaussian jitter
  per player (~0.3 yd) to simulate detection noise.
- These belong in the dataset builder, not the model — see Phase 3.

**Target by end of Phase 1:** 500+ offense-labeled clips, 500+
defense-labeled clips.

---

## Phase 2 — Hierarchical taxonomy

The current 24 flat labels mix base formation and variation tags, so the
model can't share signal between, e.g., "DALLAS WG" and "DALLAS Y OFF".

- New: `formations/taxonomy.json`
  - `offense_base`: ~8–10 canonical base formations
  - `offense_tags`: ~6 variation tags (multi-label)
  - `defense_front`: ~8 fronts
  - `defense_coverage`: ~6 coverages
- New: `formations/taxonomy.py` — load and validate. Provides a
  mapping from the **legacy 24-class labels** in
  `models/offense_positions/metadata.json` to (base, tags) so existing
  labels are not lost.
- Pipeline output (Phase 7) emits: `OFF FORM` (base), `FORM VARIATION`
  (comma-joined tags), `DEF FRONT`, `COVERAGE`.

---

## Phase 3 — Permutation-invariant offense model

Flat MLP on `[nx1, ny1, …, nx11, ny11]` is order-sensitive: the same
formation with players in a different list order looks different.

### 3.1 Architecture: DeepSets (recommended first)
Per-player encoder `φ(xᵢ, yᵢ, posᵢ)` → sum/mean pool → classifier head
`ρ`. Naturally permutation-invariant.
- Per-player input: `(x, y, position_class_onehot)` where
  `position_class` comes from `yolo_models/positionDetection.pt`
  (QB, RB, WR, TE, OL — already in the pipeline).
- Two heads on the pooled embedding:
  - Base-formation head: softmax over `offense_base` (cross-entropy).
  - Tags head: sigmoid per tag (binary cross-entropy, multi-label).

### 3.2 Why not the flat MLP
- Current model: ~3K params, no symmetry priors → memorizes the 89-clip
  train set.
- DeepSets adds the right inductive bias (set, not sequence) with
  similar parameter count.

### 3.3 Stretch: small Transformer
Once DeepSets baseline is in place, swap `φ + pool` for a 2-layer
self-attention block (11 tokens, dim 32). Same I/O. Useful if base-form
accuracy plateaus.

### 3.4 Training
- New: `modelTraining/train_offense_setmodel.py` — *replaces* the flat
  MLP path, but loads the same CSV produced by
  `build_offense_positions_dataset.py`.
- Stratified 5-fold CV (the dataset is too small for a single
  train/val split — current code uses 80/20 which gives 17 val samples
  across 24 classes).
- Class-weighted loss + the mirror/jitter augmentation from §1.4.
- Save artifacts to `models/offense_formations_v2/` (do not overwrite
  the legacy model until Phase 7 swap-in).

---

## Phase 4 — Pre-snap motion encoder

A static snap frame misses motion, shifts, and unstable alignments. Use
the ~1.5 s pre-snap window the pipeline already tracks.

- Reuse `scripts/perFrameHomographyTransform.py` output: per-frame
  field-coordinate tracks for each player.
- Per-player track encoder: small GRU over the last ~30 frames of
  `(Δx, Δy)` displacements → 16-D motion embedding.
- Concat the motion embedding to the per-player input of the DeepSets
  encoder from Phase 3. No new pooling needed.
- New: `modelTraining/build_offense_motion_dataset.py` — extracts per
  clip an `(11, T, 2)` tensor of pre-snap tracks aligned to the snap
  frame. Reuses snap-detection logic already in
  `scripts/staticProcess.py` (lines ~160–370).
- Train as in Phase 3. Motion features are most useful for distinguishing
  formations that look similar at snap (e.g., TRIPS vs. TRIPS OPEN after
  late motion).

---

## Phase 5 — Defensive formation model

Mirror the offense pipeline; defense was never modeled.

- New: `modelTraining/build_defense_positions_dataset.py` — same shape
  as the offense builder, but selects the **defending 11** using the
  Defense/Ref class from `yolo_models/positionDetection.pt`.
- New: `modelTraining/train_defense_positions.py` — same DeepSets
  architecture as Phase 3, but two heads:
  - `defense_front` (softmax)
  - `defense_coverage` (softmax)
- New: `models/defense_formations/{model.pt, metadata.json}`.
- Defense benefits most from **box-count features** (number of defenders
  in the tackle box, depth of safeties) — compute these in the per-player
  encoder so they survive permutation invariance.

---

## Phase 6 — Robustness & validation

- Stratified k-fold CV on every model (k=5).
- Confusion matrix per fold; top-3 accuracy alongside top-1.
- **Invariance tests** (new
  `modelTraining/tests/test_invariance.py`):
  - Mirror x → predicted base formation must be stable.
  - Translate ±5 yd along y → must be stable.
  - Add ±0.3 yd Gaussian jitter → top-1 must not flip more than ~5%
    of the time.
- Temperature scaling on validation logits to calibrate confidence
  (current confidences of 0.15–0.30 are meaningless).

---

## Phase 7 — Integrate into the app

- Edit `scripts/staticProcess.py` (lines ~160–370): replace the legacy
  flat-MLP call with the v2 set-based predictor; produce both
  offense and defense outputs per snap.
- Edit `app/fileAccess.py` (line 314 has the CSV column dict): add
  `'DEF FRONT': ""`, `'COVERAGE': ""`. `OFF FORM` and `FORM VARIATION`
  already exist.
- Edit `app/processingDialog.py` (line 177 — the static-process step)
  to display new fields.
- Keep the legacy `formModel.pt` loadable via a config flag for a
  rollback path until the v2 model is validated on >100 unseen clips.

---

## Critical files

**Read / reuse (do not rewrite):**
- `modelTraining/build_offense_positions_dataset.py` — feature
  extraction logic (centroid centering, pairwise distances, PCA). The
  set-based model reuses the per-player `(x, y)` inputs but the
  geometric features remain useful as auxiliary inputs.
- `modelTraining/train_offense_positions.py` — class-weighting,
  augmentation noise, normalization constants (x/100, y/53.33). Keep
  these constants identical so feature spaces stay comparable across
  models.
- `scripts/staticProcess.py` (lines 160–370) — snap extraction +
  side-normalization logic.
- `scripts/perFrameHomographyTransform.py` — per-frame field
  projection; the motion encoder consumes its output unchanged.
- `yolo_models/positionDetection.pt` — already separates
  offense/defense; both new datasets depend on it.
- `models/offense_positions/metadata.json` — legacy 24-class label
  list, mapped into the new taxonomy in Phase 2.

**New files:**
- `modelTraining/labeling_tool.py`
- `modelTraining/build_offense_motion_dataset.py`
- `modelTraining/build_defense_positions_dataset.py`
- `modelTraining/train_offense_setmodel.py`
- `modelTraining/train_defense_positions.py`
- `modelTraining/tests/test_invariance.py`
- `formations/taxonomy.json`
- `formations/taxonomy.py`
- `models/offense_formations_v2/` (artifacts)
- `models/defense_formations/` (artifacts)

**Edited:**
- `scripts/staticProcess.py`
- `app/fileAccess.py`
- `app/processingDialog.py`

---

## Verification

End-to-end checks, in order:

1. **Labeling tool sanity.** Run `python modelTraining/labeling_tool.py
   --cache cache/TestingFootage/` — confirm a snap is rendered, a label
   can be saved, and a row is appended to
   `cache/TestingFootage/offense_positions.csv`.
2. **Model trains.** Run
   `python modelTraining/train_offense_setmodel.py --csv-path
   cache/TestingFootage/offense_positions.csv` and confirm 5-fold CV
   metrics print. Top-1 base-formation accuracy on the existing 89
   clips should already beat the legacy model (it will, since base
   formation has only ~8 classes vs. 24).
3. **Invariance tests pass.** `pytest
   modelTraining/tests/test_invariance.py`.
4. **App integration.** Open the app, run the pipeline on
   `temptestingdata/` clips, and confirm the tracks CSV is populated
   with `OFF FORM`, `FORM VARIATION`, `DEF FRONT`, `COVERAGE`.
5. **Calibration.** Inspect a fresh
   `models/offense_formations_v2/play_predictions.csv` — confidences
   should span 0.5–0.95 (vs. the current 0.15–0.30) on the same clips.

---

## Suggested execution order

The phases are written in dependency order, but in practice:

1. Phases 1.1–1.3 (labeling tool) **first** — without it, nothing else
   moves.
2. Phase 2 (taxonomy) in parallel — small, decoupled.
3. Phase 3 (set-based offense) once 200+ labels exist.
4. Phase 5 (defense) once Phase 3 is working; the architectures are
   identical, so it's mostly data work.
5. Phase 4 (motion) **last** — biggest engineering effort, smallest
   per-clip win, only meaningful once base accuracy is healthy.
6. Phases 6–7 throughout.
