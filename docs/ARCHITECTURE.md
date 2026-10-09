# ATK Diffusion Toolkit — architecture

How `ATK_DIFFUSION_PLAN.md` (the plan of record) and `DETECTION_DESIGN.md`
(the detector) map onto the code. Read those two first; this file is the
contract between the parts of the package, so they can be built and changed
independently without drifting apart.

## 1. Two environments, one package

| Environment | Where | Has | Used for |
|---|---|---|---|
| **core** | ATK's `envs\atk_core` (Python 3.11) | numpy, scipy, onnxruntime (+ PySide6, but the toolkit never imports Qt) | everything classical, the signal cut, CPU inference of trained models, products, maps, the cabled-loop arithmetic |
| **train** | `envs\atk_diffusion` (Python 3.11, built by ATK's `get_diffusion.bat`) or the toolkit's own `.venv` | torch, torchvision, torchaudio, torchsig 2.2.0, onnx, lightning | dataset generation with TorchSig, training, ONNX export, the experiments |

Python **3.11** is the target (ATK ships 3.11 in `.python\`). TorchSig 2.2.0
declares Python ≥ 3.10 and is pure Python; its compiled dependencies (numba,
opencv) have 3.11 wheels on Windows. Do not use 3.12+ syntax.

**Importing `atk_diffusion` imports nothing heavy.** `torch` is imported only
inside `atk_diffusion.learn.*` and `atk_diffusion.experiments.*` modules, and
those packages' `__init__.py` files import nothing. `torchsig` is imported
only inside `atk_diffusion.synth.torchsig_backend`. `onnxruntime` only inside
`atk_diffusion.detect.onnx_models` and functions that need it.

## 2. Rules every module follows

1. Header line: `# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.`
   (exception: `text/loop.py` is MIT — see LICENSE §3). Then
   `from __future__ import annotations`.
2. A module docstring that says what it is, WHY it is built this way, and
   which section of the plan it implements. Bill's own words, where they
   shaped the code, are quoted. Limits are stated, not hidden.
3. **No Qt, no network at run time, no writes outside the paths it is
   given.** Data goes under an `RfData` root (`atk_diffusion.paths`) or an
   explicit output folder; never the user profile, never AppData.
4. **Plain words.** Functions that can fail in a way an operator must
   understand return or raise a sentence (`why`), not a code. Long jobs take
   `progress: Callable[[str], None] | None`.
5. **Tiers** (`atk_diffusion.provenance`): every output that is not the
   record carries `atk:tier`, set with `provenance.tier_for(method)`. A new
   method must be added to `METHOD_TIERS` before it may write output.
6. **Cards** (`atk_diffusion.cards`): every trained model is saved with
   `cards.save(dir, card, weights_file)` and loaded with
   `cards.load(dir, expect_kind=…, for_profile=…)`. A model without a card
   does not load.
7. **Profiles** (`atk_diffusion.profiles`): RF data, datasets and models are
   per receiver profile; the profile's sample rate is the only rate a model
   sees. A mismatch raises `ProfileMismatch` with the plan's sentence.
8. **Classical baselines first** (plan §7). Every learned tool has its
   classical comparator beside it, and an experiment that scores both.
   Every reconstruction tool reports a **hallucination rate**: how often it
   produces a signal, pulse, word or character where the truth has none.
9. **Tests**: pytest, `tests/test_<area>_<topic>.py`, fast (a few seconds a
   file), CPU only, seeded, no network. A test needing PyTorch calls
   `pytest.importorskip("torch")` INSIDE the test or fixture, never at module
   scope (ATK's lesson: a module-scope skip silently skips the pure tests
   beside it). Torch tests call `torch.set_num_threads(1)` and use tiny
   models. Tests write only under `tmp_path` (the `rf` fixture is an RfData
   in a temporary folder; `conftest.py` points `ATK_RF_DATA` at a temporary
   folder for the whole session).
10. **Verify against the real package.** Code written against TorchSig,
    torchvision or onnxruntime has a test section that runs against the real
    package when it is importable, and is skipped with a printed reason when
    it is not — never a stub written from the docs.

## 3. The core (built first, stable)

| Module | What |
|---|---|
| `paths` | `RfData(root)`: the rf_data layout (`captures`, `cabled`, `synthetic`, `datasets`, `models`, `runs`, `cuts`, `products`, `labeled`, `profiles`, `environments`, `shared`), root rules (AppData and sync folders refused), `WriteLog` (`record`, `verify`) |
| `profiles` | profile ids, `FAMILIES`, `canonical_rates`, `bandwidth_class`, `canonical_for`, `StftGeometry`, `FamGeometry`, `ReceiverProfile` (+ `SafeInput`), `load_profile`/`save_profile`, `profile_from_meta`, `check_match` → `ProfileMismatch`, `describe` |
| `sigmf` | `write_pair`, `load`, `read_meta`, `write_meta`, `Annotation`, `annotations`, `add_annotations`, `validate`, `ATK_KEYS` |
| `provenance` | `TIERS`, `TIER_WORDS`, `METHOD_TIERS`, `tier_for`, `decoded_from_note`, `stamp`, hashes |
| `cards` | `ModelCard`, `KINDS`, `new_card`, `save`, `load`, `find`, `summary`, `CardRefusal` |
| `capabilities` | what is installed, in words |
| `dsp.iq` | bytes ↔ complex for cu8 / ci8 / ci16 / ci16q11 / cf32; `power_dbfs`, `clipped_fraction`, `deinterleave` |
| `dsp.resample` | `shift`, `decimate`, `cut_to_canonical` (integer decimation to the class's canonical rate), `resample` and `resample_capture` (the one logged way to another profile's rate) |
| `detect.classes` | the class table v1: `CLASSES`, `FAMILIES`, `DECODERS`, `cycle_frequencies`, `cp_lags`, `tools_for`, `technology`, `UNKNOWN` |
| `detect.boxes` | `Detection` (the detector's one currency), `merge`, `overlap_tf`, `SOURCE_STYLES`, `box_style`, `legend`, colour science |

## 4. The areas and their contracts

### 4.1 Front end, CFAR, tracker, confirmer, prototypes, pipeline (`dsp`, `detect`)

- `dsp.stft`: `spectrogram(x, geom) -> (S_db, frame_times)`;
  `tiles(x, fs, geom, floor) -> iterator of Tile(spec_db_above_floor
  [rows, bins] float32, t0, t1, f0, f1, row_period, bin_hz)`; time is
  max-pooled to `geom.tile_rows`; tiles overlap by `geom.tile_overlap`.
- `dsp.floor`: `NoiseFloor` — measured per bin from a terminated capture or
  estimated robustly (a low percentile over time), tracked slowly
  (`update`), `above(spec_db) -> dB above floor`. Absolute level is kept
  beside the tile for measurement, never fed to a detector (§2).
- `dsp.cfar`: `ca_cfar(psd_db_above_floor, pfa, guard, train) -> mask`;
  `energy_proposer(tile, pfa) -> list[Detection]` (contiguous runs → boxes;
  `sources=("energy",)`, `snr_db`).
- `detect.tracker`: `Track` and `Tracker.update(dets) -> tracks` (Hungarian
  assignment on time-frequency overlap and family; `scipy.optimize.
  linear_sum_assignment`); first/last seen, duty, PRI for bursts, drift, hop.
- `detect.confirm`: `Confirmer(registry)` — `registry[decoder_key] =
  callable(detection, iq, fs) -> ConfirmResult | None`; the host (ATK)
  supplies the callables; the toolkit owns the rules: only a decoder
  confirms; the decoder wins the label; disagreement is kept.
- `detect.prototypes`: `PrototypeBank` per (profile, canonical class):
  `add(cls, embeddings, source)`, `classify(embedding) -> (cls | UNKNOWN,
  distance, threshold)`, `calibrate_thresholds(held_out)`, `teach(cls,
  embeddings)` (seconds, no GPU; below the example floor the threshold is
  wider and the bank says so), `save/load`.
- `detect.onnx_models`: `Proposer2D` and `Classifier1D` wrappers over
  onnxruntime (CPU) — load through `cards.load`, refuse profile mismatch,
  measure latency. I/O names are fixed (§5).
- `detect.pipeline`: `DetectorPipeline(profile, …).feed(iq, center_hz,
  t_start, epoch=None) -> list[Detection]`; `run_on_capture(path, …)`;
  `status()` (latency, disagreements, the refusal line). Energy always on;
  cyclic and learned each switchable; low-SNR denoised path optional and
  flagged.

### 4.2 Cyclostationary processing, filters, escalation, the cut (`cyclo`, `dsp.measure`, `cut`)

- `cyclo.scf`: FAM and SSCA (ported from ATK's own `atk/core/siga/csp`,
  which is Bill's code and tested), `cyclic_profile` (max over f of |S|),
  conjugate and non-conjugate; `scf_image(x, fs, geom) -> (img, f_axis,
  alpha_axis)` for the classifier's second input.
- `cyclo.probes`: targeted, cheap: `symbol_rate_line(x, fs, rates)` (power
  or delay-multiply + Goertzel), `cp_probe(x, fs, lag_s)` (the OFDM cyclic
  prefix — the cell-tower probe), `carrier_conj(x, fs)`; each returns a
  statistic with the threshold DERIVED from the noise-only distribution at
  that integration (never tuned), and the integration time.
- `cyclo.proposer`: `cyclic_proposer(x, fs, center_hz, profile, classes)
  -> list[Detection]` scanning `classes.cycle_frequencies()` and
  `classes.cp_lags()` per CFAR region or across the span; `sources=
  ("cyclic",)`, `alpha_hz`, `integration_s`.
- `cyclo.escalate`: `IqBuffer` (the last N seconds at the profile's rate,
  2–10 s), `EscalationPolicy` (triggers: ambiguous confidence, SNR below
  the profile's threshold, the hunter dwelling, the analyst asking), and
  `escalate(buffer, region) -> list[Detection]` flagged `escalated`.
- `cyclo.filters`: `matched_parameters(cut) -> {symbol_rate, carrier_offset,
  timing}` for the demodulator; `fresh_clean(x, fs, alphas)` and
  `fresh_separate(x, fs, alpha_sets)` (blind adaptive FRESH / cyclic Wiener);
  `score(x_multi, fs, alpha)` (SCORE beamforming on the Kraken's five
  channels). Each returns the output and the MEASURED before/after SNR.
- `dsp.measure`: occupied bandwidth, SNR above floor, symbol rate and
  carrier offset from the cyclic profile, burst length, PRI, duty —
  classical, checkable.
- `cut`: the signal cut (DETECTION_DESIGN §4.2), see §6 for the folder.

### 4.3 Synthetic data and datasets (`synth`, `dsp.impair`)

- `synth.native`: numpy generators for every class with `native=…` in the
  class table, at any sample rate, with exact labels (symbol rate, carrier
  offset, SNR, bandwidth). Works without TorchSig.
- `synth.torchsig_backend`: TorchSig 2.2.0 at the profile's exact rate (the
  rate is set from the profile, never typed); narrowband (one signal) and
  wideband (many) generation; TorchSig metadata → SigMF annotations.
- `dsp.impair`: measure a receiver's impairments from a terminated capture
  (floor shape, DC spike, IQ imbalance, ENOB, spurs) → `profiles\<p>.json`
  `impairments`; apply them to synthetic data.
- `synth.environments` + `synth.scene`: environment profiles (plan §3.6) and
  the wideband scene composer.
- `synth.datasets`: the dataset builder (§5) and the cabled-set ingester.

### 4.4 Training (`learn`) and evaluation (`experiments`)

- `learn.proposer2d` (FCOS, torchvision, ResNet-18 class), `learn.
  classifier1d` (IQ branch + SCF branch, embedding, cycle regression),
  `learn.ssl`, `learn.calibrate` (temperature scaling, open-set thresholds),
  `learn.export` (ONNX) — all per profile; cards written by `cards.save`.
- `learn.diffusion` (DDPM/DDIM core, SNR ↔ timestep), `learn.unet`,
  `learn.denoiser` (B3), `learn.inpaint` (D1), `learn.translator` (A6),
  `learn.augment` (B4), `learn.fingerprint` (C1/C2), `learn.radiomap`
  (E2/E5 residual), `learn.position` (E3), `learn.vitals` (I1),
  `learn.genclass`, `learn.beacon`, `learn.anomaly` (R).
- `experiments.*`: each track's first experiment as a function that returns
  a result dict and writes a report (markdown + JSON) under the profile's
  `runs\`. Synthetic stand-ins make every experiment runnable here; the
  real run on Bill's captures is the same function pointed at his data.

### 4.5 Repair, maps, fingerprints, text, the field tools

- `repair`: `iq_dropout` (D1 classical), `speech` (D2), `pulses` (D3),
  `tracks` (D4), `document` and `audio_inpaint` (D5).
- `geo`: `products` (GeoTIFF/GeoJSON), `dted`, `terrain`, `propagation`,
  `reach` (E5), `posterior` (E1), `radiomap` (E2), `whereami` (E3),
  `aperture` (E4), `contours`.
- `fingerprint`: `features` (C1 classical), `library`, `social` (C3).
- `text`: `loop` (MIT; the host-side diffusion loop shared with
  Palimpsest §8.6), `backends`, `novelty` (F1), `revise` (F2), `extract`
  (F3), `style` (F4).
- `ask` (B5 point-and-ask, teach), `hunt` (B6), `cabled` (A6 transmit
  safety, the loop), `sensing` (I1/I2 ESP32 CSI), `hf` (J).

## 5. Shared formats

### Built datasets — `<rf_data>\<profile>\datasets\<name>\`

```
manifest.json    {name, profile, sample_rate, kind: narrowband|wideband,
                  generator: "torchsig 2.2.0"|"native"|"cabled"|…,
                  canonical: {class, rate, decimation}   (narrowband)
                  stft: {...}, fam: {...},              (geometry used)
                  classes: [names], families: [names], splits: {train, val, test},
                  label_sources: [...], environment: "", resampled: false,
                  created, params, files: {relpath: sha256}}
train\ val\ test\
  narrowband: shard_NNN.npz   iq (N, L) complex64 at the canonical rate,
                              label (N,) int32, family (N,) int32,
                              snr_db, symbol_rate_hz, carrier_offset_hz,
                              bandwidth_hz (N,) float32, [scf (N, H, W) float16]
  wideband:   scene_NNN.sigmf-data/-meta (the record, labels as SigMF
              annotations) + tiles_NNN.npz  spec (N, rows, bins) float16
              dB above floor, boxes (M, 6) float32 [tile, row0, bin0, row1,
              bin1, family], box_class (M,) int32
```

### ONNX model I/O (training ↔ inference)

- **proposer2d** — in `tile` float32 [1, 1, rows, bins] (dB above floor,
  normalised as `card.input["normalize"]` says); out `boxes` float32 [K, 4]
  (row0, bin0, row1, bin1 in tile pixels), `scores` float32 [K], `labels`
  int64 [K] (family index into `card.input["families"]`).
- **classifier1d** — in `iq` float32 [B, 2, L], `scf` float32 [B, 1, H, W];
  out `logits` [B, C], `embedding` [B, D] (L2-normalised), `cycle` [B, 2]
  (symbol rate and carrier offset, normalised by `card.input["cycle_scale"]`).
- **denoiser** — in `x` float32 [B, C, …], `t` int64 [B]; out `eps`; the
  noise schedule is in `card.input["schedule"]`, so a single-step denoise at
  the SNR-matched timestep runs in numpy + onnxruntime.

### The signal cut — `<rf_data>\<profile>\cuts\<UTC stamp>_<centre Hz>\`

```
original.sigmf-data/-meta   the box: shifted, low-passed, integer-decimated to
                            the canonical rate (cf32). atk:tier = record,
                            atk:source_capture, atk:source_sample_start/count,
                            atk:decimation, atk:canonical_class, atk:cut_by
cleaned.sigmf-data/-meta    (optional) atk:tier from the method, atk:method,
                            atk:method_params, atk:snr_before_db/after_db,
                            atk:model_sha256 for learned methods
separated_N.sigmf-*         (optional) FRESH separate outputs
analysis.json               measurements, cyclic peaks, class or UNKNOWN,
                            fingerprint, cleans[], routes[]
scf.npy / scf.png           the spectral correlation surface
cyclic_profile.npy / .png   the cyclic domain profile with its peaks
report.md                   written from the facts in analysis.json
```

A cut folder is self-contained: readable without ATK, importable into a new
install, and the unit the RF social graph and the watchlist point at.

## 6. What ATK owns (FUTURE_PLANS.md, 2026-10-08)

The toolkit has no UI. ATK owns the sub-tabs (AI Detect, Cuts, Hunt), the
waterfall's boxes and its right-click, the airlock and `get_diffusion.bat`,
`rf_data` in the recorder, the cabled-loop page, map layers from products,
point-and-ask's chat turn, the ESP32 sensor port, the hour log, and the RF
social graph in Network Link. ATK's adapter is `atk/core/diffusion_host.py`.
