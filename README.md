# ATK Diffusion Toolkit

Diffusion-model and learned-RF tools for ATK, in their own engine repo with
an ATK adapter. **All rights reserved** (see `LICENSE`; `text/loop.py` alone
is MIT, shared with Palimpsest). Package `atk_diffusion`, Python 3.11.

**Status, 2026-10-09: built.** Every track of the plan has its code and its
first experiment, proved on synthetic stand-ins (1,195 tests). The real
numbers need your receivers, captures and GPU — `docs/FIRST_EXPERIMENTS.md`
is the runbook, `docs/BUILD_STATUS.md` says what is built and measured.

## Getting it running

- **Beside ATK (the normal case):** in `D:\Analyst_Toolkit\ATK` run
  `get_diffusion.bat`. It puts the classical half into ATK's core
  environment and builds the training environment (`envs\atk_diffusion`:
  PyTorch with CUDA, TorchSig 2.2.0) from ATK's own Python. `/core` does
  only the ATK half.
- **On its own:** `install.bat` here builds `.venv` (`/cpu` without an
  NVIDIA GPU).
- Then `atkdiff.bat status`, and `atkdiff.bat --help` for every command.

Data never lives in this folder: everything goes to
`D:\Analyst_Toolkit\rf_data\`, one folder per receiver profile (plan §3).

## The documents

| File | What it is |
|---|---|
| `ATK_DIFFUSION_PLAN.md` | The plan of record: principles, the RF data foundation (receiver profiles, the sample-rate law, `rf_data\`, the cabled loop, environment profiles, products), every track with its first experiment, the build order, evaluation, licenses, the papers mapped, decisions, change log. |
| `DETECTION_DESIGN.md` | The AI signal detector in detail: three proposers, the cut and classifier, cyclostationary processing, the right-click signal cut, low-SNR escalation and the three classical filters, training, CPU deployment, licenses, classes, labels, evaluation, build steps, where it sits on screen, decisions. |
| `docs/BUILD_STATUS.md` | What is built, what was measured here, what needs your hardware; what was found and fixed while checking; deviations from the plans and why. |
| `docs/FIRST_EXPERIMENTS.md` | The runbook: each track's first experiment as commands to type, in the plan's build order. |
| `docs/ARCHITECTURE.md` | How the plans map onto the code: the two environments, the rules every module follows, the shared formats (datasets, ONNX I/O, the cut folder). |

## The code

| Package | What |
|---|---|
| `paths`, `profiles`, `sigmf`, `provenance`, `cards`, `capabilities` | the foundation: rf_data and its write log, receiver profiles and canonical rates, SigMF labels, tiers, model cards (a model without a card does not load) |
| `dsp` | IQ datatypes, resampling, the front end, the noise floor, CFAR, measurement, impairments, classical denoisers |
| `cyclo` | FAM/SSCA, cyclic probes (incl. the LTE/NR cyclic-prefix probe), the cyclic proposer, escalation, matched / FRESH / SCORE |
| `detect` | the class table, detections and box styling, tracker, confirmer, prototypes, ONNX inference, the pipeline |
| `cut` | the signal cut: cut, analyze, clean, route, save |
| `synth` | TorchSig 2.2 and native generators, labels, environment profiles, scenes, datasets |
| `learn` | training (PyTorch, training environment only): proposer, classifier, SSL, calibration, export, the diffusion models |
| `experiments` | each track's first experiment, with a report |
| `repair`, `fingerprint`, `geo`, `text`, `ask`, `hunt`, `cabled`, `sensing`, `hf` | the tracks D, C, E, F, B5, B6, the cabled loop, I and J |

## Where the rest lives

- **ATK's side** — sub-tabs, the right-click, the subsystem airlock,
  `rf_data` in the recorder, the hour log, map layers from products — is in
  ATK itself; what it owes is listed in `D:\Analyst_Toolkit\ATK\FUTURE_PLANS.md`,
  section dated 2026-10-08.
- **Palimpsest** (the host-side diffusion loop shared with this repo is its
  plan §8.6): `D:\Analyst_Toolkit\Palimpsest\PALIMPSEST_PLAN.md`.
- **Athanor** (spikes S7 and S8 for Palimpsest):
  `D:\Analyst_Toolkit\Athanor\ATHANOR_PLAN.md`.
- **Passive radar** inputs are noted in the plan (§4.G) and belong in `atkpr`.
- **Papers**: `D:\Analyst_Toolkit\new papers\` (+ `additional\`,
  `additional\more\`), mapped in the plan §9.
