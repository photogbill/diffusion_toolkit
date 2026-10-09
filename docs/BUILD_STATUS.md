# Build status — 2026-10-09

What is built, how it was checked, and what still needs your hardware, your
captures or your GPU. Plain rule used throughout: **built** means the code
path exists end to end and has tests; **measured here** means a number came
out of it on synthetic stand-ins in the build container (Linux, 2 CPUs, no
GPU, no radios); **real** numbers only come from your machine
(`docs/FIRST_EXPERIMENTS.md` says how to get each one).

## In numbers

| | |
|---|---|
| Toolkit engine (`atk_diffusion`) | 133 modules, about 62,500 lines; no Qt, no network at run time |
| Toolkit tests | 1,195, all passing (`python -m pytest tests`) in about 4 minutes on 2 CPUs |
| ATK side | 17 new modules and pages, 11 changed files, about 23,000 new lines with tests; 16 new test files |
| Python | 3.11 (ATK's own); TorchSig 2.2.0 installed and generating data for real at 2.4 MS/s |

## Track by track

| Track | Built | Measured here (synthetic) | Needs from you |
|---|---|---|---|
| **A** foundation | profiles, rf_data + write log, SigMF labels, the refusal, the logged resampler, impairment measure/apply, TorchSig 2.2 + native generators, dataset builder, environment profiles and the scene composer (`us-va-nokesville`), domain gap and minutes-to-acceptable harnesses | datasets for several profiles; refusals; TorchSig and native both labelled in SigMF | a terminated capture per receiver; data-sheet safe inputs; the cabled loop; yard minutes |
| **A6** translator, **cabled loop** | transmit-safety arithmetic and refusals (D10), transmit files, plan/measure/align, the cabled-set ingester, the paired translator | the safety refusals; stand-in receivers for the translator | bladeRF or HackRF, cable, DC block, attenuators — `cabled check` first |
| **B1/B2** detector | front end (dB above floor), CFAR energy proposer, cyclic proposer, FCOS 2D proposer, IQ+SCF 1D classifier with embeddings and cycle regression, prototypes/open set/teach, tracker, confirmer, calibration, ONNX export and CPU inference, model cards | pure noise: **0 false alarms in 20 s**; about 64 ms per tile on this 2-CPU box (budget 200 ms) | training on the GPU on your datasets; `experiment detector` |
| **B3** weak-burst denoiser | DDPM core, U-Nets, SNR↔timestep, spectrogram/IQ denoiser, core-env runtime (numpy + ONNX), the experiment | classical baseline on synthetic noise: Pd 0.9 at about −3.0 dB (Wiener), −1.5 (wavelet), −0.9 (median), −2.0 (none); matched filter (knows the waveform) below −15 dB; 0 hallucinations in 100 noise tiles | a trained denoiser (GPU) and your terminated captures for the learned row |
| **B4** augmentation | TFD-lite time-frequency diffusion, judged by the gap | stand-ins only | cabled recordings |
| **B5** point-and-ask, teach | payload (crop + measurements + prompt), teach as a prototype | — | the Cuts tab in ATK |
| **B6** hunter | goal in words, rule policy, bounded receive-only action set, log, simulator, ATK live loop with dry run | simulated band: **found 25 % of bursts vs 4 % for a fixed scan**, median 0.9 s to find | AI detection on with a live radio |
| **B7** cyclostationary + the cut | FAM/SSCA, cyclic profile, CP probe (timing, offset, cell count), symbol-rate lines, escalation buffer, matched/FRESH/FRESH-separate/SCORE, the cut folder, route, report | 9 CP-cell tests, filters with measured before/after SNR | right-click → Cut signal… in ATK |
| **C1/C2** fingerprints | classical features, library with UNKNOWN, channel-resilient CNN, denoise-then-fingerprint | told two simulated same-model radios apart 100 % at 5–30 dB — **but rejected a stranger 0 % of the time** (see below) | two of your handhelds, cabled |
| **C3** social graph | co-occurrence, schedule, co-movement, reply; i2/GeoJSON export; ATK import | — | after C1 |
| **D1** IQ dropout | detection, interpolation/AR/Janssen fills, learned inpainter, the DSD-sync experiment | 2 ms DMR gap: sync after the gap 57 % (linear) vs 15 % (zeros); 0 hallucinated bursts | a trained inpainter |
| **D2** speech | spectral subtraction, MMSE-LSA, learned enhancer at arm's length, WER with a hallucination check | — | your clips and transcripts, Whisper command |
| **D3** pulses | deinterleaving, PRI completion, inferred pulses flagged | 99.7 % of pulses to the right emitter; **0 of 1,013 inferred pulses hallucinated**; 79 % fill recall | radar recordings |
| **D4/D5** tracks, documents, audio | Kalman/RTS and great-circle fills, deskew/denoise/deblur, audio gap fill | library tests | ATK hooks in OCR / air picture come later |
| **E1** posterior cloud | MCMC/importance posterior, HPD contours | 90 % regions held the truth 92 % of 300 times (honest); fail when bias is unmodelled, as they should | — |
| **E3** where-am-I | kernel, map inversion, MDN | synthetic drive world: median error about 1.2–1.4 km | drive CSVs with GPS truth |
| **E4** synthetic aperture | self-calibrated moving aperture, direct position determination | 509 m from the tower over a simulated loop | Kraken snapshots on a driven loop |
| **E5** predicted reach | DTED reader (levels 0–2), terrain, free space / two-ray / Bullington / Deygout / **ITM (Longley-Rice)**, antenna patterns, reach map, measured and learned layers; **`products reach` on the command line** | physics 5.4 dB RMS at held-out drive points → 2.6 dB after kriging; the learned correction made it **worse** (17.9 dB) at its default 500 steps — not shipped by plan §7 | DTED1 tiles, a radio's parameters; a drive |
| **F** text | host-side diffusion loop (MIT, Palimpsest §8.6), backends (autoregressive via your GGUFs, Transformers), novelty, revise, extract, style | planted corpus: TF-IDF baseline agreed best (Spearman 0.89, AUC 1.00); model scores 0.69–0.89 | your documents; DiffusionGemma waits on llama.cpp PR #24427 |
| **H** hour log | ATK: hourly entries from recorded events only, annotations, CSV/DOCX/PDF | — | it runs by itself once opened |
| **I1/I2** CSI | ESP32 CSI parser/serial, PulseFi pipeline, LSTM, presence | breathing error 0.2/min, heart 1.0/min on synthetic CSI; **0 %** of empty-room windows reported a breathing rate | the ESP32 pair |
| **J** HF now | WSPR spots, Maidenhead, openness matrix, map arcs, voacapl at arm's length | code path only (synthetic spots) | spot files from your Kiwis |
| **R** research | beacon co-design, generative classification, CyberWolf anomaly (context, never suppression) | small runs | — |

## ATK's side (`FUTURE_PLANS.md` 2026-10-08, items 1–10)

1. **rf_data in the recorder** — ● Record I/Q files every capture under
   `rf_data\<profile>\captures\` with the profile in the file, hashed into
   the write log; falls back to `data\iq` with the reason when it cannot.
2. **Right-click cut** — the waterfall box menu has *Cut signal…*; it opens
   on RF → Cuts.
3. **Sub-tabs** AI Detect, Cuts, Hunt (and CSI Sensing), each with its box,
   OFF by default; the waterfall gains only detection boxes, a legend chip
   row and the menu item. Box styling per DETECTION_DESIGN §12.2 — with one
   colour moved (below).
4. **The airlock** — `get_diffusion.bat`, the toolkit as an optional
   subsystem (missing / adapter / startup), install.bat's summary line.
5. **The cabled-loop tool** — Setup & Config → Diffusion Toolkit, with the
   transmit-safety arithmetic shown as you type.
6. **Map layers** — the Geospatial map's Layers tab lists rf_data products,
   coloured by how they were made.
7. **Point-and-ask** — *Ask* on the Cuts tab.
8. **ESP32 CSI sensor** — RF → CSI Sensing (a measurement, never a
   diagnosis); replay and a practice log until the pair arrives.
9. **The hour log** — Analysis Suite → Hour Log.
10. **RF social graph** — Network Link → *Import RF social graph…*.

All ATK tests for these pass with real Qt (offscreen). The rest of ATK's
suite was run file by file before and after the change; see the delivery
note for the comparison.

## Things found while checking, and what was done

- **PySide6 6.12.0 (released 2026-10-08) crashes ATK on Python 3.11.** It
  loses a reference to `True` on every signal and to `None` on many Qt
  calls; after about a thousand signals the process aborts with
  `bool_dealloc`. A live radio emits that many in seconds. Your current
  install predates it; **the next `install.bat` would have pulled it.**
  install.bat now pins `PySide6<6.12` and self-tests the pair it installed
  (6.11.2 passes: 5,000 signals, no change).
- **Every background job in ATK leaked** (`workers.submit` kept each Worker
  alive for ever). One per click was invisible; the CSI page's estimate
  every 5 seconds made it grow. Fixed; workers are freed and still counted.
- **The weak-burst experiment's first version was wrong**: it fed the
  detector's integrated layer max-pooled data as if it were mean-pooled, so
  the no-denoiser detector fired on every noise tile and its column looked
  best. Fixed and pinned by a test; the numbers above are after the fix.
- Smaller: the Cuts viewer could act on a different cut than the one shown
  after a quick second click; the detector thread could be destroyed while
  still running at exit; one menu per right-click was never freed; the
  hunt log could fail to close with a full queue; the CSI port could stay
  open if the box was unticked while it was opening. All fixed, with tests.

## Deviations from the plans, and why

- **Profile datatypes are the ones on disk**: `rtlsdr_2400000_cu8` (not
  `ci8` — RTL samples are unsigned), `bladerf1_…_ci16`.
- **Canonical rates for the RTL at 2.4 MS/s are 48 k / 480 k / 2.4 M**, not
  the sketched 48 k / 240 k / 1.2 M: 240 kHz cannot hold a 250 kHz LoRa
  chirp with a guard, 1.2 MHz cannot hold a 2 MHz ADS-B signal. bladeRF and
  HackRF rates are exactly the plan's.
- **Cyclostationary boxes are vermilion `#ff5400`, not cyan** — by
  DETECTION_DESIGN §12.2's own rule: ATK already draws LTE cells in cyan
  (`#3fd0e0`), so a cyan dashed box would read as an LTE tower. AI stays
  magenta.
- **Tiers use ATK's existing vocabulary**: a Wiener/matched/FRESH/SCORE
  output is CLEANED, an interpolated fill INFERRED, a diffusion output
  INVENTED (the plan called every cleaned file Invented). The rule that the
  original is the record and nothing else is, is unchanged.
- **`text/loop.py` is MIT** (it is Palimpsest's loop, shared); LICENSE §3
  says so. Everything else is all rights reserved.
- **LTE's cyclic feature** is the CP at lag 1/15 kHz with cycle frequencies
  at the 14 kHz symbol rate — corrected in DETECTION_DESIGN §3.

## Honest limits

- No model here has been trained at full size: the GPU is yours. Every
  trainer runs end to end on tiny stand-ins and writes a card.
- Fingerprinting's open-set rejection is the weak point on simulated radios
  (strangers matched as a known radio). Treat a fingerprint match as a
  proposal until the real two-radio run measures it.
- The learned E5 correction is not yet better than kriging; the physics and
  measured layers are what the map should show.
- TorchSig is MIT and runs at the profile's exact rate; its wideband scenes
  were exercised, but your RF environment is only a prior until scored.
- DiffusionGemma itself was not available here; the loop was proved on a
  toy masked model and against the autoregressive backend. Since the
  afternoon of 2026-10-09 ATK runs DiffusionGemma itself
  (`get_diffusion_gemma.bat`: PR #24427's own server, with ATK's patch, as
  a Cognitive Core). That server generates whole replies and does not
  expose per-step logits, so this repo's host-side loop (`text.loop`) still
  needs Transformers or a merged llama.cpp to drive the model step by step.
