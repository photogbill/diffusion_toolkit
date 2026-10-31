# ATK Diffusion Toolkit

Learned-RF and diffusion-model tools for ATK (the Analyst Toolkit), in their
own engine repository with an ATK adapter. Package `atk_diffusion`,
Python 3.11; no Qt and no network at run time. **All rights reserved**
(see `LICENSE`; `text/loop.py` alone is MIT, shared with Palimpsest).

Two environments, one rule: the **classical half** (the front end, the
detectors, the cuts, cyclostationary processing, the ONNX runtime, the
library) runs in ATK's core environment on the CPU; the **training half**
(PyTorch with CUDA, TorchSig 2.2) lives in its own environment and produces
the models the first half runs. Every model carries a card, and a model
without a card does not load.

## Getting it running

- **Beside ATK (the normal case):** in the ATK folder run
  `get_diffusion.bat`. It puts the classical half into ATK's core
  environment and builds the training environment (`envs\atk_diffusion`)
  from ATK's own Python. `/core` installs only the ATK half.
- **On its own:** `install.bat` here builds `.venv` (`/cpu` without an
  NVIDIA GPU).
- Then `atkdiff.bat status`, and `atkdiff.bat --help` for every command.

Data never lives in this folder: everything goes to `rf_data\` beside the
code (one folder per receiver profile), with a write log of what produced
each file.

## What it provides

| Area | What |
|---|---|
| **The RF data foundation** | receiver profiles and canonical sample rates (the sample-rate law and its refusal), `rf_data\` with its write log, SigMF labels and tiers, provenance, model cards, the logged resampler, impairment measurement and application, TorchSig 2.2 and native generators, environment profiles and the scene composer, the dataset builder |
| **The cabled loop** | transmit-safety arithmetic and refusals, transmit files, plan / measure / align, the cabled-set ingester, and the translator between receivers — for recordings made over a cable, never over the air |
| **The AI signal detector** | a front end in dB above the measured floor; three proposers (CFAR energy, cyclic, a 2D FCOS); an IQ + SCF 1D classifier with embeddings and cycle regression; prototypes, open-set handling and teaching; tracker, confirmer and calibration; ONNX export and CPU inference. On pure noise: zero false alarms in 20 s. |
| **Cyclostationary processing and the cut** | FAM / SSCA, the cyclic profile, the LTE/NR cyclic-prefix probe (timing, offset, cell count), symbol-rate lines, the escalation buffer and its three classical filters (matched, FRESH, SCORE), the cut folder, route and report — ATK's right-click *Cut signal…* |
| **Sifting — the deep scan** | the operator's list, a plan in words, a blind segmented search at several widths with a derived Gamma(K) null, a ledger, record-first dwells with confirmation, a schedule and a run log; a BPSK carrier found blind 10 dB under the floor at 250 kS/s |
| **The hunter** | a goal in words, a rule policy, a bounded receive-only action set, a log and a simulator; in a simulated band it found 25 % of bursts against 4 % for a fixed scan |
| **Denoisers and repair** | classical denoisers (Wiener, wavelet, median) with measured before/after SNR; the DDPM core, U-Nets and the SNR↔timestep map behind the learned weak-burst denoiser; IQ dropout detection with interpolation, AR and Janssen fills; speech enhancement at arm's length with a hallucination check; pulse deinterleaving and PRI completion with inferred pulses flagged; track, document and audio gap fills |
| **Fingerprints and the social graph** | classical features, a library with UNKNOWN, a channel-resilient CNN, denoise-then-fingerprint; co-occurrence, schedule, co-movement and reply graphs with i2 and GeoJSON export |
| **Geolocation** | a posterior cloud (MCMC / importance sampling, HPD contours — 90 % regions held the truth 92 % of 300 times), where-am-I from a map, a self-calibrated synthetic aperture, and **predicted reach**: a DTED reader (levels 0–2), free space / two-ray / Bullington / Deygout / ITM (Longley-Rice), antenna patterns and reach maps |
| **The training library** | one training folder per receiver and one per signal type; real examples cut from recordings; dirty, fresh and AI-generated copies with a quality gate and an *invented* tier; the dataset builders (validation and test are real only) and the trainers with plain-words verdicts; retire and bring back; the report — ATK's Setup → RFMLS Training |
| **Text, CSI, HF** | the host-side diffusion loop for text (novelty, revise, extract, style); ESP32 CSI parsing with presence and vitals; WSPR spots, Maidenhead and the HF openness matrix |
| **Experiments** | each area's first experiment as a command, with a report, so a number on real hardware is one command away |

The detectors are set by a false-alarm probability rather than a tuned
threshold, every learned output carries its tier, and nothing in the
classical half needs the GPU.

## Documents

| File | What it is |
|---|---|
| `docs/ARCHITECTURE.md` | How the code is organised: the two environments, the rules every module follows, the shared formats (datasets, ONNX I/O, the cut folder). |
| `docs/FIRST_EXPERIMENTS.md` | The runbook: each area's first experiment as commands to type. |

## Tests

```
python -m pytest tests
```

About 1,200 tests, all without a GPU or a radio, in a few minutes.
