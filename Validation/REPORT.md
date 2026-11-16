# Validating SHINE denoising without ground truth — Tier 2 and Tier 3 report

Datasets: `710_80A_Insitu(8)` and `710_80A_Insitu(9)` HAADF (Shoaib-MLC, 2026-09-19)
Status: Tier 2 (denoised result) and Tier 3 (temporal behaviour) complete for both datasets;
Tier 3 retraining control, single-frame (`--frame_num=1`) control and neighbour-corruption
augmentation (`--temporal_aug_p=0.25`) on Insitu(8).

## 1. Purpose

A self-supervised denoiser has no clean reference to compare with, so a referee can always ask
whether a structure in the denoised image was in the sample or was made by the network. The
synthetic ground-truth tests (Tier 1) were done in the previous publication. This report covers
**Tier 2**: tests on the real data that need no ground truth. They answer three questions:

1. Did the denoiser remove only noise? (residual test)
2. What does the network make out of pure noise? (null test)
3. Do two models that never saw the same noise produce the same image, and how close is that image to
   a frame neither of them saw? (even/odd test, held-out error, spectral error)

Units used throughout: **σ** is the noise standard deviation of one raw pixel, measured from
consecutive frames; errors in squared units are given in **σ²**. Particle contrast is the
particle-minus-background level of the time-averaged raw frame (~2000 for Insitu(8), ~3000 for
Insitu(9), raw units).

## 2. Data and production models

| | Insitu(8) | Insitu(9) |
|---|---|---|
| Raw frames | `HAADF-tif/`, 191 × 1024², float32 | `HAADF-tif/`, 77 × 1024², float32 |
| Denoised (published) | `HAADF_denoised/experiment20260920/denoised/` | same layout |
| Checkpoint that made them | `epoch=39.ckpt` | `epoch=42.ckpt` |
| Noise σ per raw pixel | 1563 | 1516–1521 |
| Frame-to-frame drift (median) | 0.11 px | 0.07 px |

Production settings (both): `3x3_blind`, 5 frames, Adam lr 1e-3, batch 8, crop 256, 100 epochs,
fp16, L2 loss. `main.py --test=1` predicts with the **best-validation** checkpoint, not the newest
file in `model/`; `find_checkpoint.py` identifies it by re-denoising frames (exact match: median
difference 0).

Detector facts that shape the methods:
- HAADF signal is not electron counts, so a frame cannot be split into two independent halves by
  binomial thinning. Independent noise realisations come from different frames instead.
- Noise is correlated along the fast-scan direction (correlation 0.14 at Δx = 1 px, ~0 along y).
- The raw power spectra contain sharp, frame-wide peaks: periodic scan interference.
- Production output is min/max-rescaled to each raw frame, so its intensities are not in raw
  units. Every test first fits `raw ≈ a·denoised + b` (2 parameters per frame); a global linear map
  cannot create or remove structure.

## 3. Methods

### 3.1 Residual test (`residual_test.py`)
The residual `r_t = raw_t − (a·den_t + b)` is compared with an independent noise estimate from
consecutive raw frames, `n_t = (raw_t − shift(raw_{t+1})) / √2`. The shift comes from phase
correlation of the denoised frames and is applied in Fourier space, which leaves the noise power
spectrum unchanged. If only noise was removed, `r_t` and `n_t` agree in:
- variance at each signal level (ratio > 1: signal removed; < 1: noise kept),
- autocorrelation (incl. the fast-scan correlation),
- radial and 2D power spectra (spectral ratio 1 at all frequencies),
- zero time-mean everywhere (16 px block z-scores against the noise reference),
- zero correlation with the denoised image (whole frame and 32 px blocks).

It also reports the intensity bias against signal level, the low-pass difference between the
time-mean raw and time-mean denoised images (for the static parts of the scene, the time-mean raw
frame is nearly noise-free), and how much power the denoised output keeps at each sharp
scan-interference peak.

### 3.2 Null test (`null_test.py`)
A 256 px patch is replaced in **all five input frames** by a flat level plus noise. The rest of the
frame stays real, so the network sees production normalisation and context. Two kinds of noise:
- synthetic: Gaussian, shaped to the measured 2D noise power spectrum and variance;
- real: frame-difference noise cut from the same place.

There are 6 particle-free locations × 24 frames × 2 noise types = 288 null patches. A stress case
feeds whole windows of pure noise. The same multi-scale blob detector (scale-normalised Laplacian
of Gaussian, σ = 1.5–12 px; score ≈ blob amplitude / 2 in raw units) runs on the null outputs and on
every production frame. **The largest null score is the false-detection floor**: a real detection
below it cannot be told apart from a feature the network made out of noise.

### 3.3 Even/odd test (`train_evenodd.sh`, `evenodd_test.py`)
The stack is split into interleaved halves. Model E is trained on even frames only and model O on
odd frames only, with production settings (`recursive_factor=2.2` gives each the same number of
steps). Each model denoises its own half, so the two outputs share no noise, in training or at
inference. Even frame t (model E) is compared with odd frame t+1 (model O) after registration.
Interleaving keeps the compared frames one frame apart; splitting into first and second halves
would compare frames about 95 frames apart. The first and last two pairs are dropped (see 5.4).

Splitting each frame by electron events (the dose-fraction estimator) is the ideal version for
counted data. The frame split is its equivalent for HAADF, with one difference: A and B are
different time points, so **real change between t and t+1 counts as error**. All accuracy numbers
below are therefore upper bounds.

Measurements:
1. **Held-out error** — the dose-split estimator applied to frames:
   `E‖f(A) − B‖² − σ²_B = ‖f(A) − x‖² (+ ½‖x_t − x_{t+1}‖²)`, with `A` = raw even frame,
   `B` = raw odd frame t+1, and `σ²_B` estimated from the frame pair itself (this includes readout,
   scan and non-Poisson noise). The same estimate is made for simple estimators that use the even
   half only (Gaussian blur 1–3 px, 5-frame mean) and for the raw frame (= 1 σ² by construction).
2. **Spectral error** — the same subtraction ring by ring:
   `E(k) = ⟨|F[f(A)] − F[B]|²⟩_ring − N(k)`, with `N(k)` from frame differences (same Hann
   window), against the signal power `S(k) = ⟨|F[B]|²⟩_ring − N(k)`. `E(k)/S(k) < 1` means the
   frequency is reproduced better than leaving it out. The relative error is shown only where a
   single frame carries measurable signal (`S > 0.02·N`). Sharp interference peaks are masked out
   of the rings.
3. **FRC(f(A), raw B)** (accuracy; capped by the noise in one raw frame) beside FRC(E, O)
   (reproducibility only — it cannot detect errors both models share) and FRC(raw, raw).
4. **E − O maps** in units of σ, plus the time-mean |E − O|.
5. **Blob reproducibility**: detections above the null floor in E, matched in O within
   max(2 px, blob σ). The chance level comes from matching against O shifted to an unrelated position.

## 4. Protocol

Environment: `~/anaconda3/envs/TEM_denoiser/bin/python` (`PY` below). Run from `Validation/`.
Results used here are in `Experiment/validation_710_Insitu{8,9}_HAADF/`.

```bash
PY=~/anaconda3/envs/TEM_denoiser/bin/python
D="/mnt/h/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(8)"
RAW="$D/HAADF-tif"; DEN="$D/HAADF_denoised/experiment20260920/denoised"
OUT=../Experiment/validation_710_Insitu8_HAADF

# 0. which checkpoint made the published frames
$PY find_checkpoint.py "$RAW" "$DEN" "$D/HAADF_denoised/experiment20260920/model"

# 1. residual test (CPU, ~3 min for 191 frames)
$PY residual_test.py "$RAW" "$DEN" $OUT/residual

# 2. null test (GPU, ~10 min); CKPT = output of step 0
$PY null_test.py "$RAW" "$DEN" "$D/HAADF_denoised/experiment20260920/model/epoch=39.ckpt" $OUT/null

# 3a. even/odd models (GPU, ~16 min per half; symlinks the halves, trains, predicts)
SHINE_PYTHON=$PY ./train_evenodd.sh "$RAW" "$PWD/$OUT/evenodd"

# 3b. even/odd analysis (CPU, ~5 min); uses the null floor from step 2
$PY evenodd_test.py "$RAW" $OUT/evenodd/even/experimentrun/denoised \
    $OUT/evenodd/odd/experimentrun/denoised $OUT/evenodd/analysis \
    --null_summary $OUT/null/null_summary.json [--roi y0,y1,x0,x1]
```

Notes:
- `train_evenodd.sh` has the production hyper-parameters built in; extra `main.py` arguments
  after the two paths override them. If a dataset was denoised with other settings, pass the same
  settings here.
- Each script writes a `*_summary.json` (all numbers below), a `.npz` with curves and maps, and
  PNG figures.
- `null_test.py` uses `--model`, `--patch 256`, `--n_patches 6`, `--n_frames 24`; `evenodd_test.py`
  drops `--skip_edges 2` pairs at each end.

## 5. Results

Figures: `figures/insitu{8,9}_{residual_test, residual_diagnostics, residual_autocorr, null_test,
evenodd_test, temporal_test}.png`, `figures/temporal_summary.png`, `figures/insitu8_temporal_control.png`,
`figures/insitu9_particle_lowerleft.png`, `figures/insitu{8,9}_overview.png`.

### 5.1 Residual test — raw minus denoised is noise

| | Insitu(8) | Insitu(9) |
|---|---|---|
| Residual / noise variance, median over signal levels (range) | 1.01 (0.92–1.22) | 1.00 (0.96–1.16) |
| Spectral ratio < 0.05 / 0.05–0.25 / ≥ 0.25 cycles/px | 1.00 / 1.03 / 1.00 | 0.97 / 1.04 / 1.01 |
| Fast-scan autocorrelation, residual vs noise | 0.142 vs 0.138 | 0.143 vs 0.140 |
| corr(residual, denoised), largest 32 px block | 0.025 | 0.048 |
| Interference power kept in denoised output | < 1% | ≤ 3% |
| Peak low-frequency bias (time-mean raw − denoised, 4 px blur) | 101 (~5% of contrast) | 178 (~6%) |

- **No structure added.** Residual and noise agree in level, spectrum and correlation. No
  periodicity appears in the output that is not in the raw data. Scan interference is removed.
- **Smooth intensity bias.** Averaged over time, the denoised background in the top-left of
  Insitu(8) is about 100 units brighter than raw, and particle cores are slightly over-bright.
  It is a smooth distortion, not structure, but it matters for quantitative HAADF intensities.
- **Real lattice attenuated (Insitu(9), lower-left particle).** The 13.6 px fringe at
  (fy, fx) = (0.039, 0.062) is in the raw particle crop at 53× the median power (1.3–1.5× in
  background crops). It is not at any scan-interference frequency, so it is real lattice. On the
  particle, the residual carries 10–14% more power than noise below 0.1 cycles/px, and the
  time-mean particle is about 150 units dimmer than raw. The lattice changes between frames and is
  partly averaged out: real detail is lost, none is invented.

### 5.2 Null test — flat noise in, flat image out

| | Insitu(8) | Insitu(9) |
|---|---|---|
| Output std / input noise std in null patches (median) | 0.024 | 0.029 |
| Same, whole-window pure-noise stress case | 0.034 | 0.042 |
| False-detection floor (largest null blob score), synthetic / real noise | 177 / 113 | 159 / 137 |
| Floor as blob amplitude | ~350 units (~18% of contrast) | ~320 units (~10%) |
| Real detections above the floor (median per frame) | 81 | 264 |

### 5.3 Even/odd test — independent models agree; accuracy without ground truth

Agreement between the two models:

| | Insitu(8) | Insitu(9) |
|---|---|---|
| Pairs used | 91 | 34 |
| std(E − O) | 0.050 σ | 0.063 σ |
| Time-mean \|E − O\|, background / particles | ~0.03–0.05 σ / ≤ 0.15 σ | ~0.03–0.05 σ / ≤ 0.15 σ |
| FRC(E, O), 0.05–0.1 cycles/px (raw pair: 0.10–0.11) | 0.29 | 0.28 |

Accuracy against the held-out frame (upper bounds):

| Held-out error (σ²) | Insitu(8) | Insitu(9) |
|---|---|---|
| **SHINE** | **0.0081** | **0.0093** |
| Gaussian blur 3 px / 2 px / 1 px | 0.017 / 0.030 / 0.101 | 0.017 / 0.031 / 0.104 |
| 5-frame mean (even half) | 0.20 | 0.20 |
| Raw frame | 1.00 | 1.00 |
| SHINE RMS error | ≤ 0.090 σ = 141 units (≤ 7.1% of contrast) | ≤ 0.097 σ = 147 units (≤ 4.8%) |

Spectral error:

| | Insitu(8) | Insitu(9) |
|---|---|---|
| Single frame has measurable signal below | 0.113 cycles/px (8.8 px) | 0.074 cycles/px (13.5 px) |
| SHINE E/S, 0.01–0.05 / 0.05–0.1 cycles/px | 0.39 / 0.93 | 0.53 / 0.83 |
| SHINE E/S reaches 1 | never, within the signal band | never, within the signal band |
| Gaussian blur 3 px / 5-frame mean reach 1 at | 0.027 / 0.035 | 0.020 / 0.027 |

Blob reproducibility (fraction found in O, true pair / chance):

| Blob score band | Insitu(8) | Insitu(9) |
|---|---|---|
| Lowest quarter above floor | 0.29 / 0.02 | 0.31 / 0.14 |
| Middle half | 0.37–0.54 / 0.02–0.04 | 0.48–0.58 / 0.16–0.24 |
| Top quarter | 0.90–0.94 / 0.08–0.12 | 0.78–0.82 / 0.40–0.60 |
| Small blobs (σ ≤ 2.5 px) | 0.20 / 0.01 | 0.12 / 0.06 |

Reading:
- **No invention from noise.** Structure made from noise would differ between E and O; the
  outputs differ by about 0.05 σ.
- **Accurate where the data allow.** SHINE's error against an unseen frame is about half that of
  the best simple filter. Every frequency with measurable signal is reproduced better than leaving it
  out, where the simple filters fail 3–4× lower in frequency. Near the signal limit the margin is
  thin (E/S ≈ 0.8–0.9): fine detail there is attenuated, as in 5.1.
- **Nothing added where there is no signal.** Beyond the signal limit, SHINE's error curve lies
  on the signal curve: it misses the (tiny) signal and adds no power of its own. Invented fine
  structure would push the error above the signal.
- **Shared errors are invisible to E vs O.** If E and O had independent errors, their spread would
  imply an error of only 0.0013–0.002 σ²; the held-out estimate is 4–6× larger. The difference is
  error both models share (smoothing, lattice attenuation, intensity bias) plus real change. FRC(E, O)
  is therefore reported as reproducibility, never as resolution.
- **Faint, small blobs are not individually reliable.** All score bands reproduce above chance,
  but blobs just above the null floor do so only 20–50% of the time (part of the shortfall is real
  change between frames). Argue from strong blobs (score ≳ 550 for Insitu(8), ~90% reproduced) or
  from statistics over many faint ones, not from single faint dots.

### 5.4 Artefact found: mirrored windows at the stack ends
`Utils.idxreturn` mirrors the 5-frame window at the ends (frame 0 is denoised from [2,1,0,1,2]).
The duplicated frames' noise looks like a confirmed signal to the network:
- noise carried over from neighbouring frames is about twice as high: corr(den_0, raw_1 − raw_6) =
  0.055, vs 0.025 mid-stack and 0.027 for frame 0 with five distinct frames ([4,3,0,1,2]);
- the first and last two frames show 4–5× more blob detections (350–430 vs ~80 per frame in Insitu(8)).

The middle of the stack is unaffected. Fix: build edge windows from distinct frames. That changes
production behaviour and is not applied; until then, exclude or flag frames 0, 1, T−2, T−1.

## 6. Limitations

- Accuracy numbers include real change between consecutive frames, so they are upper bounds;
  disagreement concentrates at particle edges, where the sample really changes.
- The null test bounds false positives in flat background only. Structured backgrounds are covered
  by the even/odd test.
- The blob detector and matching radius are one reasonable choice; the conclusions are the trends
  against score and chance, not the exact fractions.
- Each half-model has a window spanning twice the time of the production model, so it is somewhat
  weaker. This makes the even/odd test conservative.
- Two HAADF datasets from one session. Other detectors (indirect, LAADF) need their own noise
  checks (spectrum, correlation) before reuse.

## 7. Tier 3 — temporal behaviour of the 5-frame input

### 7.1 Why editing the input measures temporal leakage
SHINE is trained on 5-frame inputs, but the loss is computed only against the **centre** frame,
and the blind spot is spatial and applies only to the centre frame. The four neighbouring frames are fully
visible, including the pixel being predicted, so training teaches the network to lean on them
wherever the scene is static. Temporal leakage is that learned reliance; it is fixed in the
trained weights. Tier 3 probes it by editing the **raw** input of one frame t0 (real noise and
context stay) and measuring the change in the outputs of t0−2 … t0+2. The edit enters as the
centre frame when t0 is denoised, and as a neighbour when t0±1 and t0±2 are denoised. Responses are
differences between edited and unedited outputs of the same model, so the noise cancels. Amplitudes are
fitted by least squares with the known profile, plus a constant background where needed.

### 7.2 Tests (`temporal_test.py`)
- **Insert:** a Gaussian blob (σ = 2 or 5 px) added to frame t0 only, at 8 particle-free sites ×
  10 frames. Amplitude A = (0.5, 1, 2, 4) × 2 × null floor, i.e. about 0.5–4× the false-detection
  floor, 0.1–0.9 σ. Ideal response: 1 at t0, 0 elsewhere.
- **Remove:** a real isolated particle subtracted from frame t0 only. Particles are found at each t0
  by segmenting that frame's production output, because the particles in Insitu(8) move a lot over
  the stack. The profile is the production output of t0 on raw units, minus the local background,
  tapered. `raw_left` reports how much of the particle remains in the edited raw frame.
  Ideal: 0 at t0, 1 elsewhere.
- **Jump:** the same particle moved by d = 2, 5, 10 or 20 px (along x and along y) in frame t0 only. A
  joint fit gives the amplitude at the old and at the new position. Ideal at t0: new 1, old 0.
- **Neighbours:** the four neighbours replaced by far frames, by noise-only frames, or by copies of
  the centre frame. Measured: the change in output, and the held-out error against raw t+3
  (outside every window). Also measured: the **noise transfer kernel**, adding 0.5 σ realistic
  noise to one input frame t+k and measuring how much passes into output t (same-pixel
  regression coefficient, and power ratio).
- **Retraining control** (`temporal_control.sh`, Insitu(8)): in production a real event is part
  of the training data, but an inference-only edit is not. An edited copy of the stack (64 blob
  insertions at 4× floor and 27 particle removals, in 15 frames) was trained with production
  settings. Validation loss is flat after about 2 epochs (0.9883–0.9888, in production too), so
  the "best" epoch is close to random; **all saved checkpoints** were compared: production epochs
  32/39/48 against retrained 3/30/39.
  The two models were then applied to **new** removal and jump events at frames at least 6 frames
  from any edit (frames 24, 49, …, 176). This separates "learned the specific events" from
  "changed its general behaviour".

### 7.3 Protocol
```bash
# inference sweep (GPU, ~10-15 min per dataset)
$PY temporal_test.py sweep "$RAW" "$DEN" "$CKPT" $OUT/temporal --null_summary $OUT/null/null_summary.json
#   options: --tests insert,remove,jump,neighbours  --n_t0 10  --t0s <frames>

# retraining control (GPU, ~20 min): edited stack -> train -> compare all checkpoints
SHINE_PYTHON=$PY ./temporal_control.sh "$RAW" "$DEN" "$M/epoch=32.ckpt $M/epoch=39.ckpt $M/epoch=48.ckpt" \
    $OUT/null/null_summary.json $PWD/$OUT/temporal_control

# new events (not edited in the control stack), each model
$PY temporal_test.py sweep "$RAW" "$DEN" <ckpt> $OUT/temporal_control/new_events_<label> \
    --null_summary $OUT/null/null_summary.json --tests remove,jump --t0s 24 49 74 100 125 151 176
```
`M` is the production `model/` directory. Outputs: `temporal_summary.json`, `temporal_rows.json`,
`temporal_test.png`; the control writes `control_summary.json`, `control_rows_*.json` and
`control_comparison.png`.

One-page summary figure (CPU only; reads the summary JSONs above):
```bash
E=../Experiment
$PY temporal_summary_figure.py figures/temporal_summary.png \
    --sweep "Insitu(8)=$E/validation_710_Insitu8_HAADF/temporal" "Insitu(9)=$E/validation_710_Insitu9_HAADF/temporal" \
    --control $E/validation_710_Insitu8_HAADF/temporal_control/new_events_prod \
              $E/validation_710_Insitu8_HAADF/temporal_control/new_events_retrained --control_label "Insitu(8)"
```

### 7.4 Results
**Summary figure: `figures/temporal_summary.png`**. It has one panel per finding:
(1) no ghost features: a blob inserted at 4x the detection threshold in one raw frame stays below
the threshold in every output frame, with the largest-blob, strongest-insertion means shown;
(2) the noise kernel; (3) the response to one-frame events in the edited frame, mean over jump
distances; (4) the production model vs the retrained model on new events, with 10 px jumps
averaged over x and y.
Detailed figures: `figures/insitu{8,9}_temporal_test.png`, `figures/insitu8_temporal_control.png`.

**One-frame feature (insert), fraction of the inserted amplitude recovered (mean over sites, frames and the four amplitudes):**

| Output frame − t0 | −2 | −1 | **0** | +1 | +2 |
|---|---|---|---|---|---|
| Insitu(8), σ 2 px | 0.04 | 0.05 | **0.03** | 0.05 | 0.04 |
| Insitu(8), σ 5 px | 0.09 | 0.10 | **0.08–0.11** | 0.10 | 0.09 |
| Insitu(9), σ 2 px | 0.04 | 0.04 | **0.02** | 0.04 | 0.04 |
| Insitu(9), σ 5 px | 0.09 | 0.10 | **0.06–0.09** | 0.10 | 0.09 |

The largest leaked amplitude in any neighbouring frame is 0.73× (Insitu(8)) and 0.66× (Insitu(9))
the false-detection floor amplitude, for blobs inserted at 4× the floor.

**Removed particle, amplitude in the output relative to unedited:**

| Output frame − t0 | −2 | −1 | **0** | +1 | +2 | particle left in raw t0 |
|---|---|---|---|---|---|---|
| Insitu(8) (37 events) | 0.80 | 0.78 | **0.77** | 0.78 | 0.79 | −0.02 |
| Insitu(9) (34 events) | 0.81 | 0.80 | **0.83** | 0.80 | 0.80 | 0.24 |

For Insitu(9) the subtraction left 24% of the particle in the raw frame, because the production
output under-estimates particles there (5.1). The genuine fill-in is therefore about 0.6–0.8.

**Particle moved in frame t0 only, amplitude in output t0 at the old / new position:**

| d (px) | 2 | 5 | 10 | 20 |
|---|---|---|---|---|
| Insitu(8) | 0.81 / 0.19 | 0.77 / 0.24 | 0.76 / 0.26 | 0.77 / 0.24 |
| Insitu(9) | 0.84 / 0.17 | 0.82 / 0.21 | 0.84 / 0.25 | 0.84 / 0.16 |

(x and y jumps agree within 0.03 except Insitu(9) d = 20 px: x 0.16, y 0.24.)
In the neighbouring outputs the moved particle appears at 0.23–0.35 of its amplitude, at its new position.

**Neighbours and noise transfer:**

| | Insitu(8) | Insitu(9) |
|---|---|---|
| Held-out error, real neighbours | 0.008 σ² | 0.010 σ² |
| … centre frame copied as neighbours | 0.019 | 0.019 |
| … far frames as neighbours | 0.056 | 0.024 |
| … noise-only neighbours | 0.052 | 0.079 |
| Same-pixel noise pass-through, own frame / each neighbour | 0.0005 / 0.006–0.007 | 0.0008 / 0.005–0.006 |
| Noise power transfer per input frame | 0.0003–0.0004 | 0.0004–0.0005 |

**Retraining control (Insitu(8)), response in output t0:**

| | Production 32 / 39 / 48 | Retrained 3 / 30 / 39 |
|---|---|---|
| Insert σ 2 px (control events) | 0.030 / 0.032 / 0.029 | 0.047 / 0.034 / 0.031 |
| Insert σ 5 px (control events) | 0.098 / 0.095 / 0.090 | 0.146 / 0.105 / 0.102 |
| Remove (control events, in training) | 0.78 / 0.78 / 0.79 | 0.74 / **0.23 / 0.15** |
| Remove (new events, not in training) | 0.78 (epoch 39) | **0.21** (epoch 39) |
| Jump 10 px x, new events, old / new | 0.77 / 0.26 (epoch 39) | 0.48 / 0.46 (epoch 39) |

### 7.5 Reading
- **No ghost features.** A feature present in one frame spreads into the neighbouring outputs at
  only 4–10% of its amplitude, always below the false-detection floor (≤ 0.73× even at 4× the
  floor). SHINE does not create a detectable feature in a frame where the raw data have none.
- **The blind spot works in time too.** A frame's own noise passes into its output at 0.05–0.08%
  (same pixel). Each neighbour's noise passes at about 0.6%. The total noise power carried through is
  about 0.2% of σ², consistent with the Tier 2 residual and even/odd results.
- **Strong temporal smoothing of abrupt events.** With the production models:
  - a faint feature that lasts one frame keeps only 2–11% of its amplitude;
  - a particle missing from one frame is still shown there at about 78% (Insitu(8));
  - a particle that jumps for one frame is shown mostly (77–84%) at its old position.

  SHINE does not invent structure, but it **carries real structure across neighbouring frames**. Its
  time resolution for discrete events is several frames, not one.
- **The neighbours carry most of the information.** Replacing them with noise or far frames
  raises the held-out error 3–8×. Copying the centre frame into them doubles it: duplicated
  noise looks like confirmed signal, the same mechanism as the stack-end artefact (5.4).
- **The temporal prior is learned from the data's dynamics.** After training on a stack that
  contains particle disappearances, the model stops filling them in, **including for new events it
  never saw** (0.21 vs 0.78), and places jumped particles about halfway between the old and new
  positions (0.48 / 0.46 vs 0.77 / 0.26). It changed its general behaviour; it did not memorise
  the edits. Faint insertions are too weak to change it. So how much SHINE trusts neighbouring frames reflects how
  persistent the features in its training stack are. Abrupt events that are rare in a dataset will
  be smoothed; a dataset full of abrupt events trains a model that follows them more closely. A model
  pre-trained on one dataset and applied to another inherits the first dataset's prior.

### 7.6 Consequences for claims about dynamics
- Claims about **what is present** (particles, clusters, shapes) are supported by Tiers 2 and 3:
  nothing appears that is not in the data, and nothing leaks across frames above detection.
- Claims about **when** something happens at single-frame precision (appearance, disappearance,
  jumps, one-frame intermediates) should not rest on the 5-frame output alone. Check them against
  the raw frames, or against a single-frame (`frame_num=1`) model, which has no temporal prior.
- The stack-end mirroring (5.4) should be fixed or the end frames excluded, because duplicated
  neighbours are the strongest temporal effect measured here.

## 8. Remaining limitations of Tier 3
- The removal profile comes from the production output itself, so particles that SHINE
  under-estimates are only partly removed (Insitu(9): 24% left). Fill-in values are upper bounds
  on genuine fill-in.
- The retraining control used many edits (27 particle disappearances in 191 frames), far more than
  real data would usually contain. It shows the direction and size of the learned prior, not its
  value for any particular real dataset.
- Apart from the single-frame control (§9), only the 5-frame `3x3_blind` configuration was tested.

## 9. Single-frame control (Insitu(8))

### 9.1 Why
Tier 3 shows that the 5-frame model spreads a one-frame event over about five output frames.
A model trained with `--frame_num=1` sees only the frame it denoises, so it has no temporal prior.
It answers two questions. How much accuracy does the temporal input buy? And does a claimed
single-frame event survive without that input? Network, blind spot (3x3) and training settings
are the same as production.

### 9.2 Protocol
```bash
# trains OUT/full (whole stack, denoised with the best-val checkpoint) and OUT/evenodd/{even,odd}
SHINE_PYTHON=$PY ./train_single_frame.sh "$RAW" $PWD/$OUT/single_frame          # ~40 min GPU
# checkpoint check, residual, null, even/odd, temporal sweep (same events and amplitudes as 5-frame)
SHINE_PYTHON=$PY ./single_frame_tests.sh "$RAW" "$DEN" $OUT/null/null_summary.json $PWD/$OUT/single_frame
# figures
$PY single_frame_compare.py figures/insitu8_single_frame_compare.png --raw "$RAW" --prod_den "$DEN" \
    --prod $OUT --sf_den $OUT/single_frame/full/experimentrun/denoised --sf $OUT/single_frame \
    --frame 104 --crop 560,750,340,530
$PY temporal_examples.py "$RAW" "$DEN" <ckpt> figures/insitu8_temporal_examples[_single_frame].png \
    --null_summary $OUT/null/null_summary.json [--frame_num 1 --label "Insitu(8), 1-frame model"]
```
`null_test.py`, `temporal_test.py sweep` and `find_checkpoint.py` take `--frame_num` (default 5).
The single-frame outputs came from epoch=80.

### 9.3 Results
Figures: `figures/insitu8_single_frame_compare.png`, `figures/insitu8_temporal_examples.png`
(5-frame), `figures/insitu8_temporal_examples_single_frame.png` (1-frame).

| | 5-frame (production) | 1-frame |
|---|---|---|
| held-out error / noise variance (even/odd) | 0.0081 | 0.0106 (best Gaussian blur 0.0170) |
| E/S reaches 1 at | stays < 1 up to 0.113 cycles/px | 0.074 cycles/px |
| E/S, 0.01–0.05 / 0.05–0.1 cycles/px | 0.39 / 0.93 | 0.72 / 1.01 |
| std(E−O) / σ | 0.050 | 0.055 |
| null floor (blob score) | 177 | 329 |
| noise left on null patches | 2.4% | 2.2% |
| blobs above own floor; reproduced | 4716; 52% | 1577; 67% |
| residual/noise variance ratio | 1.01 | 1.01 |
| residual PSD ratio, f < 0.05 | 1.00 | 1.62 |
| largest den/raw power ratio at an interference peak | 1.19 | 1.78 (at ≥120 px periods) |
| intensity bias vs level (min, max) | −48, +36 | −112, +106 |
| removal: shown in output t0 | 0.77 | 0.05 |
| jump (2–20 px): old / new position at t0 | 0.78 / 0.23 | 0.09 / 0.96 |
| jump 10 px: old / new | 0.76 / 0.27 | 0.03 / 1.05 |
| blob σ 5 px kept at t0, A = 0.5 / 1 / 2 / 4 × threshold | 0.08 / 0.09 / 0.09 / 0.11 | 0.15 / 0.19 / 0.31 / 0.71 |
| blob σ 2 px kept at t0, same amplitudes | 0.03 at all four | 0.04 / 0.04 / 0.05 / 0.07 |
| blob leak into outputs t0±1, t0±2 | 0.09–0.11 | 0 (by construction) |
| own-frame noise pass-through | 0.0005 | 0.0007 |

### 9.4 Reading
- **Timing:** the 1-frame model follows one-frame events almost exactly. A removed particle
  disappears (0.05), and a particle moved ≥5 px is shown at its new position (0.9–1.1). The
  5-frame model's fill-in in Tier 3 therefore comes from its temporal input, not from the blind
  spot or the network's spatial prior.
- **Accuracy:** the temporal input is worth about 25% lower held-out error (0.0081 vs 0.0106).
  It also moves the length scale where error equals signal from about 13 px to finer than 9 px.
  Both models beat the best Gaussian blur.
- **Faint features:** without temporal input the model's response depends on amplitude. Strong
  blobs are kept (0.71 at 4× threshold) and faint ones are mostly removed (0.15 at 0.5×). Small
  (σ 2 px) blobs are suppressed by both models. Its null floor is about twice as high (329 vs
  177): what survives on flat noise is sparser but higher-amplitude. Above its own floor fewer
  blobs are found, and more of them are reproduced (67% vs 52%).
- **Smooth bias:** the 1-frame output has about twice the low-frequency intensity bias. This
  shows as smooth ≥120 px variations, not as compact structure.

### 9.5 Consequences
- To show that an event happened in one frame, show it in the 1-frame output (or raw) next to the
  5-frame output. The 1-frame model has the same network and training, and Tier 3 shows it
  follows one-frame edits.
- For morphology and quantitative intensity, the 5-frame output is more accurate and has less
  smooth bias. For timing, the 1-frame output is the faithful one. The two agree on what is
  present (both pass the residual, null and even/odd tests).

## 10. Reducing temporal leakage: neighbour-corruption augmentation (Insitu(8))

### 10.1 Idea
The 5-frame model leans on its neighbours because, in the training data, neighbours almost always
agree with the centre frame. Geometric augmentation (resize, crop, flips in `TrainLoader_plain`)
is applied identically to all five frames, so it cannot change this. The new option
`--temporal_aug_p` (`main.py`, `Utils/Dataloader_plain.py: corrupt_neighbours`; default 0 =
production behaviour) makes a fraction p of training samples have neighbours that disagree:
1–4 neighbours are replaced by a frame ≥10 frames away or by themselves shifted 5–20 px, over the
whole frame or a feathered random box (128–512 px). Replacements are real frames, so the noise
statistics are unchanged. The centre frame (the target) is never changed, so the self-supervised
loss stays unbiased. Validation samples are not corrupted.

### 10.2 Protocol
```bash
SHINE_PYTHON=$PY ./train_variant.sh "$RAW" $PWD/$OUT/corrupt_p025 --temporal_aug_p=0.25   # ~45 min GPU
SHINE_PYTHON=$PY ./variant_tests.sh "$RAW" "$DEN" $OUT/null/null_summary.json $PWD/$OUT/corrupt_p025
$PY variant_compare.py figures/insitu8_variant_compare.png --variant "5-frame, production=$OUT" \
    --variant "5-frame, corruption p=0.25=$OUT/corrupt_p025" --variant "1-frame=$OUT/single_frame"
$PY temporal_examples.py "$RAW" "$DEN" $OUT/corrupt_p025/full/experimentrun/model/epoch=49.ckpt \
    figures/insitu8_temporal_examples_corrupt_p025.png --null_summary $OUT/null/null_summary.json \
    --label "Insitu(8), 5-frame model with neighbour corruption p=0.25"
```
`train_variant.sh` / `variant_tests.sh` take any extra `main.py` arguments (`FRAME_NUM` env for
the tests). The edit events are the same as in §7–9 and are not in the training data. The
p=0.25 outputs came from epoch=49.

### 10.3 Results
Figures: `figures/insitu8_variant_compare.png`, `figures/insitu8_temporal_examples_corrupt_p025.png`.

| | 5-frame production | 5-frame, p = 0.25 | 1-frame |
|---|---|---|---|
| held-out error / noise variance | 0.0081 | **0.0079** | 0.0106 |
| E/S reaches 1 at (cycles/px) | > 0.113 | 0.105 | 0.074 |
| held-out error with far-away frames as neighbours | 0.056 | 0.022 | – |
| particle removed: shown at t0 (ideal 0) | 0.77 | **0.32** | 0.05 |
| removal seen in outputs t0±1, ±2 (ideal 1) | 0.79 | 0.86 | 1.00 |
| jump 2–20 px: old / new at t0 | 0.78 / 0.23 | 0.56 / 0.45 | 0.09 / 0.96 |
| jump 20 px: old / new | 0.77 / 0.23 | 0.46 / 0.52 | 0.02 / 1.08 |
| blob σ 5 px at 4× threshold, kept at t0 | 0.11 | 0.16 | 0.71 |
| blob leak into t0±1, ±2 | 0.09–0.11 | 0.08–0.09 | 0 |
| null floor / noise left on null patches | 177 / 2.4% | 174 / 2.3% | 329 / 2.2% |
| blobs above floor; reproduced | 4716; 52% | 4890; 50% | 1577; 67% |
| residual PSD ratio f < 0.05; bias vs level | 1.00; −48/+36 | 0.96; −52/+29 | 1.62; −112/+106 |
| own-frame / per-neighbour noise pass-through | 0.0005 / 0.006 | 0.0005 / 0.006 | 0.0007 / – |

### 10.4 Reading
- **No accuracy cost.** Held-out error, E(k)/S(k), the null test and the residual test match production.
- **Leakage is reduced, not removed.** Removal fill-in drops from 0.77 to 0.32, and a jumped
  particle is now split about evenly between its old and new positions instead of 80/20. The
  removal also leaks less into the neighbouring outputs (0.86 vs 0.79 of the particle kept there).
- **Faint one-frame features are unchanged** (0.16 vs 0.11). A feature below the noise in a single
  frame cannot be told apart from noise by its centre-frame context, so the network still follows
  the neighbours. Only the 1-frame model keeps them, and only when they are strong (§9).
- **Robust to wrong neighbours:** with unrelated frames as neighbours the error is 2.5× lower
  than production's.
- Only Insitu(8) has been tested.

### 10.5 p = 0.5
Same protocol (`train_variant.sh ... --temporal_aug_p=0.5`, output dir `corrupt_p05`; the
outputs came from epoch=49). Figures: `figures/insitu8_variant_compare.png` (all four models),
`figures/insitu8_temporal_examples_corrupt_p05.png`.

| | production | p = 0.25 | p = 0.5 | 1-frame |
|---|---|---|---|---|
| held-out error / noise variance | 0.0081 | 0.0079 | 0.0080 | 0.0106 |
| E/S reaches 1 at (cycles/px) | > 0.113 | 0.105 | > 0.113 | 0.074 |
| particle removed: shown at t0 | 0.77 | 0.32 | 0.27 | 0.05 |
| jump 2–20 px: old / new | 0.78 / 0.23 | 0.56 / 0.45 | 0.51 / 0.50 | 0.09 / 0.96 |
| jump 20 px: old / new | 0.77 / 0.23 | 0.46 / 0.52 | 0.37 / 0.60 | 0.02 / 1.08 |
| blob σ 5 px at 4× threshold, kept at t0 | 0.11 | 0.16 | 0.16 | 0.71 |
| held-out error, far-away frames as neighbours | 0.056 | 0.022 | 0.019 | – |
| null floor / noise left | 177 / 2.4% | 174 / 2.3% | 174 / 2.0% | 329 / 2.2% |
| blobs above floor; reproduced | 4716; 52% | 4890; 50% | 4813; 47% | 1577; 67% |

Accuracy is still unchanged at p = 0.5, and leakage drops a little further. Most of the gain
comes from the first 0.25. The faint-blob response does not change with p, as expected (§10.4).

### 10.6 p = 0.75 and p = 0.9
Same protocol (output dirs `corrupt_p075`, `corrupt_p09`; outputs came from epoch=49 and
epoch=61). Figures: `figures/insitu8_p_sweep.png` (trend vs p), `figures/insitu8_variant_compare.png`
(all six models), `figures/insitu8_temporal_examples_corrupt_p075.png`, `..._p09.png`.

| | prod (p=0) | 0.25 | 0.5 | 0.75 | 0.9 | 1-frame |
|---|---|---|---|---|---|---|
| held-out error / noise variance | 0.0081 | 0.0079 | 0.0080 | 0.0083 | 0.0085 | 0.0106 |
| E/S reaches 1 at (cycles/px) | > 0.113 | 0.105 | > 0.113 | > 0.113 | > 0.113 | 0.074 |
| particle removed: shown at t0 | 0.77 | 0.32 | 0.27 | 0.16 | 0.11 | 0.05 |
| jump 2–20 px: old / new | 0.78 / 0.23 | 0.56 / 0.45 | 0.51 / 0.50 | 0.42 / 0.58 | 0.33 / 0.65 | 0.09 / 0.96 |
| jump 20 px: old / new | 0.77 / 0.23 | 0.46 / 0.52 | 0.37 / 0.60 | 0.25 / 0.71 | 0.16 / 0.76 | 0.02 / 1.08 |
| blob σ 5 px at 4× threshold, kept at t0 | 0.11 | 0.16 | 0.16 | 0.24 | 0.27 | 0.71 |
| held-out error, far-away frames as neighbours | 0.056 | 0.022 | 0.019 | 0.016 | 0.014 | – |
| null floor / noise left | 177 / 2.4% | 174 / 2.3% | 174 / 2.0% | 182 / 2.2% | 200 / 1.7% | 329 / 2.2% |
| blobs above floor; reproduced | 4716; 52% | 4890; 50% | 4813; 47% | 4218; 50% | 3504; 53% | 1577; 67% |

Leakage keeps falling steadily with p; there is no plateau. The cost starts after p = 0.5:
held-out error rises 0.0080 → 0.0083 → 0.0085 (+5% at p = 0.9 vs production, still far below
the 1-frame 0.0106), and the null floor rises to 182 and 200 (finest detectable blob ~13%
stronger at p = 0.9). Fewer blobs pass the higher floor (3504) but their reproducibility is
unchanged. Faint one-frame blobs start to come through (0.24, 0.27), the first p at which they do.
Trade-off: p = 0.5 is the largest p with no measurable accuracy cost; p = 0.75 removes about
half of the remaining leakage for a 3–5% accuracy cost.

## 11. Realistic motion: a faint atom present in every frame (Insitu(8))

### 11.1 Why
The one-frame edits of §7–10 (appear/disappear/jump in a single frame) are a worst case; real
atoms and particles persist. Here the atom is in every frame and moves.

### 11.2 Protocol
`moving_atom.py`: the panel-C blob (Gaussian σ 5 px, amplitude 4× the production detection
threshold) is added to raw frames t0−5 … t0+5 at the 8 flat-background sites, 10 t0 values
(real noise kept). Outputs t0−3 … t0+3 are measured, so each measured output's whole 5-frame input
lies inside the motion. Conditions: stationary (baseline); drift 8 px/frame along x and along y;
hop 8 px or 20 px at t0, staying at the new site. Measured on the output change (with − without
atom): for drift, a least-squares split onto blobs at the atom's true positions in frames t−2 … t+2
(ideal: all at "now"), the centroid error along the motion, and length/width; for a hop, a split
onto the old and new sites. Models: production, p = 0.5, p = 0.75, 1-frame (inference only).
Figures: `figures/insitu8_moving_atom.png`, `figures/insitu8_moving_atom_examples.png` (drift),
`figures/insitu8_moving_atom_examples_hop20.png`.
Image strips (raw input, clean reference = mean of all 191 raw frames + atom, Gaussian blur 3 px,
and each SHINE model; one t0, one site; `moving_atom_images.py --cond drift_x|hop_20`):
`figures/insitu8_moving_atom_images_drift_x.png`, `figures/insitu8_moving_atom_images_hop_20.png`.

```bash
$PY moving_atom.py "$RAW" "$DEN" $OUT/moving_atom --null_summary $OUT/null/null_summary.json \
    --model_spec "production=<ckpt>" --model_spec "p 0.5=<ckpt>" --model_spec "p 0.75=<ckpt>" \
    --model_spec "1-frame=<ckpt>:1"
$PY moving_atom_figure.py $OUT/moving_atom figures/insitu8_moving_atom.png figures/insitu8_moving_atom_examples.png
```

### 11.3 Results

| | production | p = 0.5 | p = 0.75 | 1-frame |
|---|---|---|---|---|
| stationary atom: amount kept (ideal 1) | 0.97 | 0.97 | 1.03 | 0.72 |
| drift 8 px/frame: shown at true position (ideal 1) | 0.14 | 0.25 | 0.36 | 0.69 |
| drift: shown at positions of t±1 / t±2 (ideal 0) | 0.17 / 0.12 | 0.14 / 0.09 | 0.14 / 0.08 | 0.02 / 0.00 |
| drift: total shown | 0.71 | 0.70 | 0.80 | 0.72 |
| drift: mean centroid error along motion (px) | −0.06 | 0.00 | 0.05 | 0.02 |
| drift: length / width (ideal 1) | 2.63 | 2.20 | 1.99 | 1.01 |
| hop 8 px, new site at t0−1 / t0 / t0+1 (ideal 0 / 1 / 1) | 0.36 / 0.57 / 0.78 | 0.31 / 0.64 / 0.81 | 0.30 / 0.72 / 0.88 | 0.02 / 0.69 / 0.71 |
| hop 20 px, new site at t0−1 / t0 / t0+1 | 0.28 / 0.49 / 0.74 | 0.19 / 0.57 / 0.78 | 0.17 / 0.67 / 0.86 | 0.00 / 0.69 / 0.71 |

(For the 1-frame model the hop values are relative to its own 0.72 stationary level.)

### 11.4 Reading
- No atom is created or misplaced. For an atom that persists, every model puts it on its true
  path; the drift centroid error averages under 0.1 px.
- The 5-frame models behave like a camera with a 5-frame exposure. A fast-moving atom
  (8 px/frame, 1.6σ per frame) becomes a streak along its own path (length/width 2.6 for
  production), and a hop is spread over about 4 frames centred on t0 (the
  old and new sites are each partly shown from t0−2 to t0+1). Augmentation shortens both:
  length/width 2.0 and the new site at t0 0.72 for p = 0.75.
- A stationary faint atom is kept in full by every 5-frame model (0.97–1.03) but only 0.72 by the
  1-frame model. This is the reason for using 5 frames: it keeps faint persistent features at
  full strength, at the cost of temporal resolution.
- For referees: the output timing is accurate to about ±2 frames, and features moving
  more than ~σ per frame are motion-blurred along their path, not displaced or duplicated.
  Convert with pixel size and frame time to give a physical speed limit.
