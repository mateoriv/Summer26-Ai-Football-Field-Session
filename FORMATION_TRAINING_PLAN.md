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

## Guiding principle — do no harm, and don't out-run the data

With ~89 clips across 24 classes, **no architecture rescues us** — a bigger
or fancier model trained on the same tiny data overfits *harder*, not less.
So the work is split into three buckets, and the dangerous ones are gated
behind data and validation:

1. **Free wins (no labels, no training risk).** Collapsing the label space
   (taxonomy) and computing deterministic geometric attributes (shotgun /
   alignment, box count). These help *with the data we already have* and
   cannot make the model worse — they reduce what the learned model must
   carry. **Do these first.**
2. **Data-efficient modeling (gated on validation).** A permutation-invariant
   DeepSets offense model. Worth building — it has a better inductive bias
   than the flat MLP and is more data-*efficient* — but it only **ships** if
   it beats the legacy model on held-out cross-validation folds.
3. **Data-gated work (blocked until the dataset grows).** The defensive model
   (zero labels exist today) and the pre-snap motion encoder (most parameters,
   least data justification). These do **not** start until the labeled set
   reaches the thresholds below.

**Hard gates:**
- The legacy `formModel.pt` stays the **default** predictor. A new model is
  swapped in *only after* it beats the legacy model on stratified 5-fold CV
  **and** on >100 unseen clips (Phase 5).
- **Defense model:** blocked until **300+** defense-labeled clips exist.
- **Motion encoder:** deferred until **500+** offense-labeled clips exist.

---

## Phase 0 — Geometric attributes (no ML, no labels)

Some formation attributes are **measurements, not learned patterns**, and
should never be pushed through a data-starved classifier. Computing them
deterministically is reliable on day one and removes load from the model.

### 0.1 QB alignment / shotgun
- Read the raw per-frame detections in `cache/<video>/positions/*.json`
  (the `positionDetection.pt` model emits a distinct `qb` class, separate
  from `oline`).
- Measure QB depth = separation between the QB and the offensive-line
  centroid along the line-of-scrimmage normal, in field yards (use the
  homography-projected coordinates so depth is in yards, not pixels).
- Classify `under_center` / `pistol` / `shotgun` by depth thresholds
  (≈ <1.5 / 1.5–4 / ≥4 yd; tune on real clips).
- **Robustness (the QB label is not 100% reliable on 89 training clips):**
  1. take the **highest-confidence** `qb` detection (handles duplicates);
  2. if **no** `qb`, fall back to the **deepest offensive player behind the
     `oline`** — the QB by geometry regardless of label;
  3. flag low-confidence cases for review rather than guessing.

### 0.2 Box count / safety depth (defense, no labels)
- Count defenders inside the tackle box and measure deepest-safety depth
  directly from projected coordinates. Useful as auxiliary inputs to the
  defense model later, and meaningful on their own now.

Output: emit these as columns alongside the formation prediction so the app
shows them immediately, independent of any model retrain.

---

## Phase 1 — Hierarchical taxonomy (the free win, do first)

The current 24 flat labels mix base formation and variation tags, so the
model can't share signal between, e.g., "DALLAS WG" and "DALLAS Y OFF".
Collapsing them is the **single change that helps with the data we already
have**: ~24 classes at ~3 examples each becomes ~8 base classes at ~10 each.

- New: `formations/taxonomy.json`
  - `offense_base`: ~8–10 canonical base formations
  - `offense_tags`: ~6 variation tags (multi-label)
  - `defense_front`: ~8 fronts
  - `defense_coverage`: ~6 coverages
- New: `formations/taxonomy.py` — load and validate. Provides a
  mapping from the **legacy 24-class labels** in
  `models/offense_positions/metadata.json` to (base, tags) so existing
  labels are not lost.
- Pipeline output (Phase 5) emits: `OFF FORM` (base), `FORM VARIATION`
  (comma-joined tags), `DEF FRONT`, `COVERAGE`, plus the Phase 0 geometric
  columns (`QB ALIGN`, etc.).

This phase is small, decoupled, and risk-free — it only relabels existing
data into a coarser space.

---

## Phase 2 — Grow the labeled dataset (the unblock)

Without more data, no architectural change matters. This phase is the
biggest investment and the long pole for everything in bucket 3.

### 2.1 Build a labeling tool
- New file: `modelTraining/labeling_tool.py` (PySide6, reusing widgets from `app/`).
- Uses the **taxonomy from Phase 1** for its label fields (build the taxonomy
  first so the tool writes canonical labels from day one).
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

### 2.2 Active-learning ordering
- Run the current offense model over all unlabeled snaps; sort by
  ascending top-1 confidence and present **low-confidence clips first**.
  This concentrates human effort on the examples the model is most
  confused by.
- Also surface clips whose top-2 classes are within 0.05 (boundary
  cases — the most useful supervision).

### 2.3 Pseudo-label / cluster-assist
- KMeans (k≈15) on the existing 29-D geometric feature vector → present
  one cluster at a time so the user labels visually-similar plays in
  batches (5–10× speedup vs. clip-by-clip).
- Reuse the feature code in
  `modelTraining/train_offense_positions.py` (centroid-centered features,
  pairwise distances, PCA eigenvalues) — do **not** re-implement.

### 2.4 Augmentation (free data multiplier)
- Mirror plays left↔right (flip x around field centerline). Doubles the
  dataset and matches real symmetry of football formations.
- Yard-line translation jitter (±10 yd along x); small Gaussian jitter
  per player (~0.3 yd) to simulate detection noise.
- These belong in the dataset builder, not the model — see Phase 3.

**Targets:** 300+ defense-labeled clips unblocks Phase 6; 500+
offense-labeled clips unblocks Phase 7. Reach 200+ offense labels before
starting Phase 3.

---

## Phase 3 — Permutation-invariant offense model (gated on validation)

Flat MLP on `[nx1, ny1, …, nx11, ny11]` is order-sensitive: the same
formation with players in a different list order looks different. DeepSets
is more data-*efficient* (it bakes in permutation invariance instead of
spending scarce examples learning it), but it still **ships only if it wins
on cross-validation** — see the hard gate above. Start once 200+ labels exist.

### 3.1 Architecture: DeepSets (recommended first)
Per-player encoder `φ(xᵢ, yᵢ, posᵢ)` → sum/mean pool → classifier head
`ρ`. Naturally permutation-invariant.
- Per-player input: `(x, y, position_class_onehot)` where
  `position_class` comes from `yolo_models/positionDetection.pt`
  (qb, running_back, wide_receiver, tight_end, oline — already in the
  pipeline).
- Two heads on the pooled embedding:
  - Base-formation head: softmax over `offense_base` (cross-entropy).
  - Tags head: sigmoid per tag (binary cross-entropy, multi-label).
- Geometric attributes from Phase 0 (e.g. QB alignment) are fed as
  deterministic inputs/columns — not learned by this model.

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
- Class-weighted loss + the mirror/jitter augmentation from §2.4.
- Save artifacts to `models/offense_formations_v2/` (do not overwrite
  the legacy model — the swap-in is gated in Phase 5).

---

## Phase 4 — Robustness & validation (the gate mechanism)

This phase is *how* we decide whether Phase 3 is allowed to ship. Run it
before any swap-in.

- Stratified k-fold CV on every model (k=5).
- Confusion matrix per fold; top-3 accuracy alongside top-1.
- **Decision rule:** the v2 model swaps in only if it beats the legacy
  model on base-formation top-1 (and does not regress top-3) across folds.
- **Invariance tests** (new
  `modelTraining/tests/test_invariance.py`):
  - Mirror x → predicted base formation must be stable.
  - Translate ±5 yd along y → must be stable.
  - Add ±0.3 yd Gaussian jitter → top-1 must not flip more than ~5%
    of the time.
- Temperature scaling on validation logits to calibrate confidence
  (current confidences of 0.15–0.30 are meaningless).

---

## Phase 5 — Integrate into the app (do no harm)

- Edit `scripts/staticProcess.py` (lines ~160–370): add the v2 set-based
  predictor and the Phase 0 geometric attributes; produce offense (and,
  once available, defense) outputs per snap.
- **The legacy `formModel.pt` remains the default**, behind a config flag.
  The v2 model becomes default only after passing the Phase 4 gate **and**
  beating the legacy model on >100 unseen clips. Rollback path stays until
  then.
- Edit `app/fileAccess.py` (line 314 has the CSV column dict): add
  `'DEF FRONT': ""`, `'COVERAGE': ""`, `'QB ALIGN': ""`. `OFF FORM` and
  `FORM VARIATION` already exist.
- Edit `app/processingDialog.py` (line 177 — the static-process step)
  to display new fields.

---

## Phase 6 — [DATA-GATED: 300+ defense labels] Defensive formation model

Mirror the offense pipeline; defense was never modeled. **Hard-blocked
until 300+ defense-labeled clips exist** — this is purely a data task today,
not a modeling task.

- New: `modelTraining/build_defense_positions_dataset.py` — same shape
  as the offense builder, but selects the **defending 11** using the
  `defense` class from `yolo_models/positionDetection.pt`.
- New: `modelTraining/train_defense_positions.py` — same DeepSets
  architecture as Phase 3, but two heads:
  - `defense_front` (softmax)
  - `defense_coverage` (softmax)
- New: `models/defense_formations/{model.pt, metadata.json}`.
- Defense benefits most from **box-count features** (number of defenders
  in the tackle box, depth of safeties) — these come from Phase 0.2 and
  are fed as deterministic per-player inputs so they survive permutation
  invariance.

---

## Phase 7 — [DATA-GATED: 500+ offense labels, deferred] Pre-snap motion encoder

A static snap frame misses motion, shifts, and unstable alignments. But a
temporal model has the **most parameters and the least data justification**,
so it is the textbook way to make things worse on a small set. **Deferred
until 500+ offense-labeled clips exist**, and only if base accuracy is
already healthy.

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
  projection; the motion encoder and Phase 0 depth measurement consume
  its output unchanged.
- `scripts/positionDetection.py` / `yolo_models/positionDetection.pt` —
  classes: `defense, oline, qb, ref, running_back, tight_end,
  wide_receiver`. Phase 0 and both new datasets depend on these.
- `models/offense_positions/metadata.json` — legacy 24-class label
  list, mapped into the new taxonomy in Phase 1.

**New files:**
- `formations/taxonomy.json`
- `formations/taxonomy.py`
- `modelTraining/geometric_attributes.py` (Phase 0: QB alignment, box count)
- `modelTraining/labeling_tool.py`
- `modelTraining/train_offense_setmodel.py`
- `modelTraining/tests/test_invariance.py`
- `models/offense_formations_v2/` (artifacts)
- `modelTraining/build_defense_positions_dataset.py` (Phase 6, gated)
- `modelTraining/train_defense_positions.py` (Phase 6, gated)
- `models/defense_formations/` (Phase 6, gated)
- `modelTraining/build_offense_motion_dataset.py` (Phase 7, deferred)

**Edited:**
- `scripts/staticProcess.py`
- `app/fileAccess.py`
- `app/processingDialog.py`

---

## Verification

End-to-end checks, in order:

1. **Geometric attributes (no training needed).** Run the Phase 0 module on
   a processed `cache/<video>/` and confirm QB alignment
   (under_center/pistol/shotgun) is emitted per clip, with the
   deepest-back fallback firing when no `qb` is detected.
2. **Taxonomy maps cleanly.** Confirm every legacy 24-class label maps to a
   (base, tags) pair via `formations/taxonomy.py` with no label loss.
3. **Labeling tool sanity.** Run `python modelTraining/labeling_tool.py
   --cache cache/TestingFootage/` — confirm a snap is rendered, a label
   can be saved, and a row is appended to
   `cache/TestingFootage/offense_positions.csv`.
4. **Model trains and clears the gate.** Run
   `python modelTraining/train_offense_setmodel.py --csv-path
   cache/TestingFootage/offense_positions.csv` and confirm 5-fold CV
   metrics print. Base-formation top-1 must **beat the legacy model** on
   the same clips before any swap-in.
5. **Invariance tests pass.** `pytest
   modelTraining/tests/test_invariance.py`.
6. **App integration, legacy still default.** Open the app, run the
   pipeline on `temptestingdata/` clips, confirm the tracks CSV is
   populated with `OFF FORM`, `FORM VARIATION`, `QB ALIGN` (and
   `DEF FRONT`, `COVERAGE` once Phase 6 ships), while the legacy model
   remains the default predictor until the gate is cleared.
7. **Calibration.** Inspect a fresh
   `models/offense_formations_v2/play_predictions.csv` — confidences
   should span 0.5–0.95 (vs. the current 0.15–0.30) on the same clips.

---

## Suggested execution order

Re-sequenced so nothing data-hungry ships before the data exists:

1. **Phase 1 (taxonomy)** and **Phase 0 (geometric attributes)** first —
   both are free wins that help with the current 89 clips and carry no
   overfitting risk.
2. **Phase 2 (labeling tool + active learning + clustering)** — the unblock;
   uses the Phase 1 taxonomy. Everything in bucket 3 waits on this.
3. **Phase 3 (set-based offense)** once 200+ labels exist, **gated** by
   **Phase 4 (validation)** — it ships only if it beats the legacy model.
4. **Phase 5 (app integration)** with the legacy model still default until
   the gate is cleared on >100 unseen clips.
5. **Phase 6 (defense)** only after 300+ defense labels exist.
6. **Phase 7 (motion)** **last and deferred** — only after 500+ offense
   labels and a healthy base accuracy; biggest engineering effort, smallest
   per-clip win, highest overfitting risk on small data.
