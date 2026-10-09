# ATK Diffusion Toolkit — plan of record

> *"I feel like we should incorporate diffusion somewhere into ATK, but I'm
> not sure where or why. Honestly, it's just a strange gut instinct."*
> — Bill, 2026-10-08. The instinct was right: a diffusion model is a
> denoiser, and ATK is a denoising tool with a chat window on the side.

| | |
|---|---|
| **Status** | **Built 2026-10-09** — the engine (`atk_diffusion`, every track in §4) and ATK's side (`FUTURE_PLANS.md` 2026-10-08 items 1–10). Every first experiment runs here on synthetic stand-ins; the real numbers wait on Bill's receivers, captures and GPU — `docs/BUILD_STATUS.md` (what is built and measured) and `docs/FIRST_EXPERIMENTS.md` (the runbook). This file changes when a decision changes and says so in the change log. |
| **What it is** | Diffusion-model tools for ATK, in one engine repo with an ATK adapter — the pattern of Aura, CyberWolf and atkdsp. Three kinds of thing: small domain models for RF, audio and images; a text-diffusion backend; and two scoring primitives (novelty and redundancy) any workspace can call. Plus the RF data foundation that every learned RF tool stands on — receiver profiles, the sample-rate law, synthetic data at the profile's rate, and TorchSig as the detector. |
| **Who** | Bill — final say. Designed with Claude, 2026-10-08. Bill: *"Keep that which makes sense to you."* Everything below is what made sense; Bill may veto any line. |
| **Code** | `D:\Analyst_Toolkit\ATK_Diffusion_Toolkit` — this repo. |
| **Data** | `D:\Analyst_Toolkit\rf_data\<receiver-profile>\…` (§3) — beside the code on D:, never in a user profile, shared with ATK's RF path. ✅ Bill, 2026-10-08 (D2). |
| **Papers** | `D:\Analyst_Toolkit\new papers` — mapped to tracks in §9. |

## Contents

1. Why, and what diffusion actually does
2. Principles
3. The RF data foundation — receiver profiles, the sample-rate law, storage
4. Tracks — what is kept, and why
5. Not kept — and why
6. Build order
7. Evaluation
8. Licenses and dependencies
9. The papers, mapped
10. Decisions
11. Change log

---

## 1. Why, and what diffusion actually does

A diffusion model learns to remove noise, step by step, from something
corrupted. Everything it is good for in ATK follows from that one fact,
and the same mechanism keeps landing on several workspaces:

| Mechanism | What it gives an analyst |
|---|---|
| **Denoise** | A clean estimate of a weak signal — on the waterfall, in audio, in a scan, on a range-Doppler map |
| **Inpaint** | Fill a gap conditioned on everything around it — IQ dropouts, missing pulses, track holes, torn pages, audio dropouts |
| **Reconstruction error** | *What is new* — the part of a capture or a document that the model of "normal" could not predict |
| **Masked prediction** | *What is redundant* — the part that was predictable from what we already had |
| **Posterior sampling** | The ambiguity made visible — a cloud of fixes, several readings of damaged text, a coverage map with uncertainty |
| **Generation** | Realistic training data for detectors; variation beyond parametric impairments |
| **Global revision (text)** | Edit a report, a log or a translation in place, all of it at once, rather than regenerate |

The hazard is the same fact seen from the other side: a model good at
making signals out of noise will sometimes make a signal out of nothing.
In a companion that is a false memory; here it is a false contact. §2.1 is
the answer.

## 2. Principles

1. **Every learned output is a reconstruction or a proposal — never
   evidence.** It carries ATK's *Invented* tier (the forensics tier system
   built for SAM), is labeled as such wherever it is shown, and is
   clickable to the raw capture, audio, scan or text it was reconstructed
   from. The raw data stays the record. A proposal on the waterfall is a
   box marked *Proposed* until a decoder confirms it (§4.B).
2. **Receiver profiles are law** (Bill, 2026-10-08: *"anything RF related
   should have separate storage areas based on the SDR utilized"*). Every
   RF capture, dataset and model is keyed by its receiver profile (§3). A
   model is offered only for a capture whose profile matches; a mismatch is
   refused in plain words, never adapted silently.
3. **The sample-rate law** (Bill: *"OmniSIG only works if the sample rate
   is identical in training … as in the field"*). Training, synthetic
   generation and field use happen at one exact rate per profile.
   Resampling is allowed only as an explicit, logged step, and the model
   it feeds is trained at the resampled rate and says so.
4. **Never for identifying people.** No face, plate or voice
   "enhancement". The model would invent a face. Ruled out by name.
5. **Never medical.** Medical Support stays out.
6. **Measured, not asserted.** Every track has a first experiment with a
   number, and the RF tracks have one shared number — the domain gap (§7).
7. **Its own engine, ATK's airlock.** Optional subsystem, fetched by
   `get_*.bat`, failing loudly and costing one feature, never the app. Own
   venv (ATK's `envs\` pattern); nothing on PATH; nothing in AppData.
8. **Licenses respected at arm's length** (§8).
9. **Bill-proof.** One switch per tool; status in plain words; nothing
   happens silently.

## 3. The RF data foundation

Nothing learned about RF survives a change of receiver or rate. Bill,
2026-10-08: *"the bladeRF files won't work for the RTL-SDR or KrakenSDR or
HackRF files."* So before any model, the data has a shape.

### 3.1 The receiver profile

A profile is the tuple that makes two captures comparable:

```
<family>_<rate>_<datatype>[_<variant>]
  family    bladerf1 · hackrf · rtlsdr · krakensdr · kiwisdr · spyserver · sigmf-import
  rate      the exact sample rate, e.g. 2400000
  datatype  the datatype on disk — cu8 (RTL, Kraken) · ci8 (HackRF) · ci16 (bladeRF: 12-bit samples in SC16 Q11) · cf32
  variant   optional: a gain preset, a front end, a channel on the Kraken
```

Examples: `rtlsdr_2400000_cu8`, `bladerf1_4000000_ci16`,
`krakensdr_2400000_cu8_ch0`, `hackrf_8000000_ci8`. (Corrected 2026-10-09:
RTL and Kraken samples are unsigned offset-binary bytes, `cu8`, which is what
ATK's recorder has always written; a profile naming a datatype the files do
not have would be the first lie in the chain.)

The profile is **not** just a folder name. It is written into every
capture's SigMF metadata (`core:sample_rate`, `core:datatype`, `core:hw`,
plus `atk:receiver_profile`), into every dataset's manifest, and into
every model's card. The adapter refuses a model whose card does not match
the capture's profile — *"this detector was trained for the RTL-SDR at 2.4
MS/s; this capture is a bladeRF at 4 MS/s"* — and offers the matching one
if it exists.

**What a profile captures that a sample rate alone does not:** ADC bit
depth and its quantization noise, the front end's noise figure and
response, the DC spike and IQ imbalance signature, the gain structure. A
detector trained on 12-bit bladeRF data has never seen 8-bit RTL noise.
That is the receiver's own fingerprint (the mirror of §4.C), baked into
every sample.

### 3.2 Storage layout

```
D:\Analyst_Toolkit\rf_data\
  README.txt                       what this is; why profiles never mix
  profiles\<profile>.json          the receiver's measured impairments (§3.4)
  <profile>\
    captures\                      SigMF pairs (.sigmf-data + .sigmf-meta), as recorded
    cabled\                        calibration captures: known TorchSig signals played through a cable (§3.5)
    synthetic\<dataset>\           TorchSig datasets generated at this exact rate
    datasets\<dataset>\            built training/validation/test splits, with manifest.json
    models\<model>\                weights + model card (profile, rate, dataset hashes, metrics, domain gap)
    runs\                          training and evaluation logs
  shared\                          nothing profile-specific: class lists, label schemas, tools
```

Rules: the same rules as the Palimpsest home, because the reason is the
same — a dataset that took a night to build is not deleted by an update.
Install, update and uninstall never touch `rf_data\`. Never under a sync
tool. Every file ATK writes here is hashed in a write log; a capture that
changed after it was recorded is named, not used.

ATK's existing `data\iq` and the SigMF exporter (the 2026-09-11 BLUE/SigMF
work) feed this: a recording made in ATK lands in the right profile folder
by construction, because the recorder knows the device and the rate.

### 3.3 The sample-rate law, in practice

- Each profile has **one** rate. A receiver that is used at two rates is
  two profiles.
- Synthetic data is generated at that rate (TorchSig takes the rate as a
  parameter; it is set once, from the profile, never typed).
- A capture at another rate is not silently resampled. "Resample to
  profile X" is a tool with a log line, a new SigMF file whose metadata
  says `atk:resampled_from`, and the understanding that a model trained on
  resampled data is a model of resampled data.
- The waterfall's own decimation for display is display-only and never
  reaches a model.

### 3.4 The receiver impairment model

For each profile, measured once from the actual device with no antenna
(terminated): noise floor and its spectral shape, the DC spike, IQ
imbalance, ADC effective bits, spurs. Stored in `profiles\<profile>.json`
and applied as TorchSig impairments on top of its channel models, so
synthetic data sounds like *this* receiver and not a textbook one.
Re-measured when the device, firmware or gain preset changes; the file
carries the date and the device serial.

### 3.5 The cabled calibration loop (the bladeRF transmits)

Bill owns a bladeRF 1.0, which transmits. So the "cabled" condition
TorchSig models can be made real:

1. TorchSig generates a known signal set at the receiver-under-test's rate.
2. The bladeRF plays it through a cable and a fixed attenuator into the
   RTL-SDR, the Kraken (per channel), the HackRF, or the bladeRF's own RX.
3. The receiver records; the recording lands in `<profile>\cabled\` with
   the ground-truth labels and the attenuation in its metadata.

Result: real-receiver-impaired training data with perfect labels — every
impairment of that receiver, none of the guesswork. This is also the
**domain-gap** set (§7): train on synthetic, test on cabled, and the gap
is a number. Over the air is not needed for this and is not done casually:
cabled with attenuators radiates nothing; OTA only inside a shielded
enclosure or on a band and power Bill is licensed for.

### 3.6 Environment profiles — training for a region you have not been to

Bill, 2026-10-08: *"generative AI can be used to make realistic signal
samples for training. That way you could train without going out to that
exact region with your equipment and loitering when you don't need to."*
Receiver and rate say *what hears*; this is *where*. In relief or contested
areas, loitering with equipment is a safety and visibility problem, not a
time cost, so the aim is: train for the region before anyone goes, and
adapt on site in minutes.

- **An environment profile** describes a region's spectrum without a
  capture from it: the ITU region and the national frequency allocation
  table; the cellular band plan by country and operator (MCC/MNC — ATK
  already speaks these identifiers); known broadcasters and their
  allocations; the terrain class (urban, rural, mountain, coastal) and the
  channel models that go with it; the expected interference population.
  Stored as `environments\<region>.json` beside the receiver profiles.
- **The scene composer** lays TorchSig (and, if it earns its place,
  RF-Diffusion) signals into a wideband scene according to that profile —
  the right bands occupied by the right kinds of signal, with the terrain's
  channel impairments — and then passes the scene through the *receiver*
  profile's impairments (§3.4). A dataset is keyed by both:
  `<receiver-profile>\synthetic\<region>\…`.
- **Adapt, do not loiter.** On arrival, a short capture — minutes, from a
  vehicle, no antenna farm — is the adaptation set. The detector is
  fine-tuned on it (or its thresholds re-fitted), and the log records how
  many minutes it took to close the gap to the acceptance line.
- **The honest limit.** A synthetic region is a prior, not the place. Every
  environment profile is scored by the same domain gap (§7): pre-deployment
  synthetic versus the first minutes on site. The number to drive down is
  *minutes on site to acceptable* — that is what "not loitering" means,
  measured. RF-Diffusion's role here is the one it was built for: learning
  what *real* looks like from a little real, so that the little real goes
  further.

### 3.7 Products — results live outside the install, and the map can serve them

Bill, 2026-10-08: *"as long as we can save results outside of the install
folder and import them into new installs."* Everything a track produces —
radio maps, coverage rasters, emitter tracks, fingerprint libraries,
position posteriors, trained models — is a **product**, and products live
in `rf_data\products\<kind>\`, never in ATK or this repo. A new install
points at `rf_data\` and has them. Products are written in open geospatial
formats so anything can read them: rasters as **GeoTIFF** (cloud-optimized,
with the CRS and the model card in the tags), vectors as **GeoJSON**
(emitters, tracks, posterior contours), tables as CSV/Parquet, models as
safetensors with a card. ATK's own Leaflet map reads these directly
through its offline tile server. **GeoServer is optional**, not required:
because the formats are open, a GeoServer pointed at `rf_data\products\`
can serve them as WMS/WMTS to ATK's map or to anyone else's — useful when
more than one machine needs the same layers; a single laptop never needs
it.

```
D:\Analyst_Toolkit\rf_data\products\
  coverage\<run>\        predicted reach: GeoTIFF + the parameters (radio, power, antenna, model)
  radiomaps\<run>\       measured/learned radio maps: GeoTIFF + sample points (GeoJSON)
  emitters\              the fingerprint library and the RF social graph (GeoJSON + Parquet)
  tracks\<run>\          emitter tracks, posteriors, synthetic-aperture solutions (GeoJSON)
  position\<run>\        where-am-I posteriors (GeoJSON contours)
  hf\                    band-openness matrices by hour (Parquet) + map overlays
```

## 4. Tracks — what is kept, and why

Each track names its first experiment. Priorities: **A** is prerequisite;
**B** and **D2** are first wins; the rest in the order of §6.

### A. The RF data foundation — prerequisite

§3, built as code: profiles, the layout, the SigMF fields, the refusal, the
resampler with its log, the impairment measurement, the cabled loop, and a
dataset builder that generates TorchSig data at the profile's exact rate
with the profile's impairments. TorchSig itself (MIT) in its own venv on
Windows; its Ubuntu/Docker recommendation is for its large-dataset
workflow, not a requirement — verified on Bill's machine as the first
step, with the fallback that generation runs in WSL if a dependency will
not build.

*First experiment:* generate one narrowband classification dataset and one
wideband detection dataset for `rtlsdr_2400000_ci8`; record one cabled
set; the mismatch refusal fires on a bladeRF capture. Then the environment
version (§3.6): compose a scene for Bill's own region from its allocation
tables, train on it, and measure how many minutes of a real capture from
his yard close the gap — the first *minutes-to-acceptable* number.

- **A6 Paired captures → a receiver-to-receiver translator.** The cabled
  loop played into *two* receivers gives paired captures: identical input,
  different receivers. That is the training data for a diffusion model
  that translates a capture from one profile into what another receiver
  would have heard. The profile rule stands — files never mix — but a
  dataset collected on the bladeRF can be *translated* into an RTL-SDR
  dataset, labeled `atk:translated_from`, and judged by the domain gap
  like any other synthetic data. If it holds, every hour of collection
  counts for every receiver Bill owns.
  *First experiment:* translate bladeRF → RTL-SDR; train a detector on the
  translated set alone; test on real RTL cabled captures; compare with a
  detector trained on synthetic-plus-impairments.

  **Transmit safety on the loop** (Bill, 2026-10-08: *"we'd have to use
  either the bladeRF or HackRF as the transmitter … careful about transmit
  power to avoid blowing a receiver"*). Rules, enforced by the loop tool,
  not by memory:
  - The transmitter is the bladeRF or the HackRF — the only two that can.
    Never over the air for this; cabled, with a DC block.
  - The tool asks for the TX power setting and the attenuator value, looks
    up the receiver's maximum safe input from `profiles\<profile>.json`
    (entered once from the data sheet, with the date), and computes the
    expected input. Target **about −40 dBm** at the receiver — linear,
    well inside the ADC. It refuses to start above a hard ceiling with
    **20 dB of margin** under the data-sheet maximum. Ballpark: a HackRF
    at its highest TX gain wants 50–60 dB of fixed attenuation; a bladeRF
    1.0 at full TX, 40–50 dB — confirm against the sheets, never against
    memory.
  - First run is always at minimum TX gain; the receiver's own level
    reading is checked against the computed value before gain goes up.
  - The Kraken is fed through a splitter, and the splitter's loss is in
    the arithmetic.
  - Every run writes TX power, attenuation, splitter loss and the
    computed input into the capture's SigMF metadata.

### B. Detection and recognition — TorchSig as the AI detector

- **B1 Wideband detector on the waterfall.** A TorchSig-trained detector
  proposes boxes (class, confidence) on the live or recorded waterfall.
  ATK's PTT scanner already runs *energy proposes, DSD confirms*; this
  becomes *energy or AI proposes, the decoder confirms*. A proposal is a
  box marked *Proposed* until a decoder (DSD, the pager decoder, multimon)
  confirms it; a signal with no decoder stays *Proposed* with its class
  and confidence shown. Never a detection without a confirmation where a
  confirmation exists.
- **B2 Modulation recognition** on a selected VFO (the AMR survey in §9
  is the map of methods; TorchSig's classifiers are the implementation).
- **B3 The diffusion denoiser as pre-detector.** "Erasing Noise in Signal
  Detection with Diffusion Model" (§9) gives the theory: a denoising
  diffusion model as a detector that beats maximum likelihood, with the
  optimal denoising timestep a function of SNR. The two cosmic-ray papers
  do the same job for weak pulses under RFI. A small U-Net on spectrogram
  tiles from Bill's own captures, trained per profile.
  *First experiment — the weak-burst test:* inject a weak burst into real
  noise from each profile — bursts Bill can actually make or capture: a
  bladeRF-generated burst of a waveform the denoiser was not trained on, a
  handheld's PTT key-up, a pager burst, an ADS-B squitter, a LoRa chirp —
  and ask at what SNR the diffusion denoiser recovers it where a Wiener
  filter, a median filter and a wavelet denoiser miss it, with the waveform
  unknown to all four. The matched filter is optimal when the waveform is
  known; the analyst's case is that it is not. (Bill, 2026-10-08: no
  access to CSAR survival radios in twenty years, so nothing here depends
  on samples he cannot get.)
- **B4 Generative augmentation.** RF-Diffusion's time-frequency diffusion
  (GPL-3 — run as a separate process, or reimplement TFD from the paper,
  §8) to add realism beyond TorchSig's parametric impairments. Kept only
  if it closes the domain gap (§7): train with and without, test on cabled.

- **B5 Point-and-ask on the waterfall, and teach-it-a-signal** (Bill:
  *"we definitely have to do"*). Two halves. *Ask:* select a VFO, and the
  crop of the waterfall around it goes to the primary model (Gemma 4 takes
  images) together with the bench's measurements — center, bandwidth,
  symbol rate, PRI, the detector's class and confidence, the fingerprint
  if known — and the analyst asks in words. RF-GPT (§9) is the proper
  version: a spectrogram encoder trained on profile data and aligned to
  the LLM; the cheap version ships first and is measured against it.
  *Teach:* mark a signal, name a class (new or existing); the examples go
  to `<profile>\captures\labeled\<class>\`; **Teach** fine-tunes that
  profile's detector and classifier on them (few-shot; synthetic
  augmentation of the class if its modulation is one TorchSig can make),
  updates the model card, and the LLM's class list follows. A class with
  too few examples says so rather than pretending.
  *First experiment:* ten signals from Bill's own captures; blind
  point-and-ask accuracy with and without the bench measurements; then
  teach one new class from five examples and measure it on the next day's
  captures.
- **B7 Cyclostationary processing and the signal cut** (Bill,
  2026-10-08). A cyclic-feature proposer that finds known signals below
  the noise floor; the spectral correlation function as the classifier's
  second input and as a measurement; and **the right-click signal cut** —
  cut, analyze, clean, route to any tool that takes that class
  (demodulator, decoder, DF, the bench, point-and-ask, teach,
  fingerprint) or just save `original` and `cleaned` together for offline
  work. Low-SNR escalation over an IQ ring buffer, and the three
  classical filters that use what the cyclic detector found — matched,
  FRESH, and SCORE beamforming on the Kraken — each with measured gain.
  `DETECTION_DESIGN.md` §4.1–4.3.
- **B6 The self-hunting receiver** (Bill: *"I definitely want to do"*).
  DARPA RFMLS task 4 and the agentic-RF paper (§9): an agent drives the
  SDR — gain, center frequency, bandwidth, dwell — to maximize finding
  what the analyst said matters, in words (*"anything narrowband and
  bursty between 400 and 470"*). Goal-driven attention over a wide band;
  the scanner becomes a hunter. Built on the receiver control ATK already
  has: a policy (rules first, then the primary model as the agent choosing
  from a bounded action set), every retune logged with its reason, the
  detector (B1) as its eyes, the watchlist and the social graph (C3) as
  its memory. It never transmits.
  *First experiment:* a bladeRF at low power on a cable loop into the
  hunter's receiver plays a scripted sequence of bursts at random times
  and frequencies; time-to-find and fraction found, hunter versus a fixed
  scan.

### C. Specific emitter identification — the watchlist for radios

Three of Bill's papers and DARPA RFMLS task 1 are about recognizing a
*specific* transmitter from its hardware imperfections, not its software
identity. For relief work that is "is this our radio?", "is this the
beacon we heard yesterday?" — the RF side of ATK's watchlist.

- **C1 RF fingerprinting per receiver profile** (DeepRadioID; the generic
  RFF framework). The known confound is that the receiver's fingerprint
  and the channel are mixed into the transmitter's — DeepRadioID's
  channel-resilience is the method to follow, and profiles keep the
  receiver constant.
- **C2 Diffusion denoising for low-SNR fingerprinting** (the Liverpool
  paper): denoise first, then fingerprint, measured against fingerprinting
  the raw capture.

*First experiment:* two handheld radios of the same model, cabled; can the
system tell them apart at the SNRs the field gives?

- **C3 The RF social graph** (Bill: *"I love the RF social graph idea"*).
  Once C1 exists, every emitter seen is fingerprinted, tracked over time
  and place, and put on the map — and then Network Link does for radios
  what it does for people: which emitters co-occur, which share a
  schedule, which move together, which answer which. Link charts for
  transmitters; entity cards for radios; the watchlist as a living record.
  Lives in `rf_data\products\emitters\`; exported like any other graph
  (i2, GeoJSON).
  *First experiment:* Bill's own handhelds on a scripted schedule over a
  week; does the graph recover the schedule and the pairs?

### D. Repair — fill what is missing, labeled as reconstruction

- **D1 IQ dropout repair before DSD.** USB glitches break decoder sync and
  the voice is lost. Inpaint the gap in the IQ before the decoder sees it;
  the repaired span is marked in the processing mark. Directly attacks a
  known DSD-trust problem.
- **D2 Speech enhancement before Whisper.** Diffusion speech enhancement
  models exist off the shelf (SGMSE-class). A *clean before transcribe*
  step on the DSD and PTT audio paths. The largest practical win in this
  plan and the cheapest; a first-wins item.
  *First experiment:* word error rate on Bill's transcripts with and
  without, on the same clips.
- **D3 Pulse-train completion and deinterleaving** for the Signals bench:
  dropped pulses from fading wreck PRI and stagger analysis; inpaint the
  train and flag inferred pulses. The Fraunhofer blind-source-separation
  paper (§9) is the deinterleaving method for overlapping emitters.
- **D4 Track inpainting** — ADS-B coverage holes, bearing gaps, GPS
  telemetry gaps, conditioned on the offline road graph for ground tracks.
  Also "where does it reappear."
- **D5 Forensics repair** — document restoration before OCR (deblur,
  denoise, de-skew) and audio inpainting in the Media Lab, both Invented
  tier through the existing tier system.

### E. Posteriors and maps — show the ambiguity

- **E1 Geolocation as a cloud.** Sample plausible emitter positions given
  the bearings and terrain, with the lobes, instead of one dot with an
  ellipse. "Soft range information" (§9) is the same idea for ranging:
  all probable values, not one.
- **E2 Radio maps from sparse measurements.** ControlRadio and Diffusion²
  (§9) generate coverage maps from a 3D or map environment and sparse
  samples. For AURA this is the question itself — *where can we reach?* —
  with OSM buildings and terrain as the environment and field-strength
  samples from ATK's own receivers as the conditioning.

- **E3 Where am I, from the spectrum** (Bill: *"I love the map
  inversion"*). Invert the radio map. Cell identifiers (which ATK already
  speaks), FM and broadcast allocations, measured strengths and the
  offline map; a model trained on Bill's own drive data learns the
  fingerprint of *place* and returns a position **posterior** — contours
  on the map, not a dot — with no GPS. GPS-denied navigation for a relief
  team under jamming or indoors. The same machinery as E2 run backwards;
  products in `rf_data\products\position\`.
  *First experiment:* drive data with GPS as truth; hold out a route;
  predict position from RF alone; the error distribution, and whether the
  posterior's stated probability is honest.
- **E4 The Kraken as a moving synthetic aperture** (Bill: *"I love …
  kraken as a moving synthetic aperture"*). Five coherent channels is a
  small array; driven with GPS- and PPS-timestamped captures it becomes a
  much larger one for emitter localization — the passive-emitter cousin of
  SAR. A known technique; every piece exists (coherent array, GPS
  telemetry, the geolocation refinement already built); the inverse
  problem is where the diffusion posterior sampler earns its place.
  Products in `rf_data\products\tracks\`.
  *First experiment — no transmitter needed:* localize a known broadcast
  tower (position public) while driving a loop; error versus truth, and
  versus the stationary DF fix.
- **E5 Predicted reach on the map — terrain propagation with DTED Level
  1** (Bill: *"if I have the DTED level 1 loaded, we can use space loss
  and other propagation models to show on the map the likely reach given
  the radio, power levels, antenna patterns"*). Physics first: free-space
  and two-ray for a sanity line, then a terrain-aware model — the NTIA
  ITM (Longley-Rice) library, open source, with DTED1 read through GDAL —
  taking the radio's power, the antenna pattern from ATK's own antenna
  designer, heights, frequency and the receiver's sensitivity, and
  painting the likely reach as a GeoTIFF in `rf_data\products\coverage\`.
  Then the learned part, which is E2 done honestly: the diffusion radio
  map learns the **residual** between the physics prediction and real
  measurements, so it only fills what physics cannot, and the map shows
  physics, measurement and learned correction as three layers an analyst
  can toggle. For AURA this is the planning tool: *where will this radio
  reach from here?*
  *First experiment:* predict coverage of a known broadcaster from its
  published parameters; drive and measure; the residual map, before and
  after the learned correction.

### F. Text diffusion — DiffusionGemma through the host-side loop

The backend is the host-side diffusion loop Palimpsest's plan §8.6
specifies — written once, shared. Blocked on llama.cpp PR #24427 merging
into the ordinary API; experiments run in PyTorch meanwhile.

- **F1 "What's new."** The analysis step Bill felt was there. Redundancy
  is predictability: mask a sentence, let the model fill it from
  everything already in the project; perfect reconstruction means
  redundant, failure means new. Two tools: a *novelty filter* on ingest
  (rank fifty documents by what each adds) and a *new-facts highlight*
  inside a document (the sentences that resisted reconstruction).
- **F2 Composing and revising.** Athena's Composer as a diffusion model —
  the specialists' outputs as context, the synthesis written as one canvas
  with global coherence. In-place report revision when new data arrives.
  The hour log's revision step (§4.H).
- **F3 Structured extraction, faster.** Network Link's chunked JSON
  extraction as a canvas, conditioned on the Pass 0 brief; measured
  against the autoregressive extractor on the same documents for speed and
  accuracy.
- **F4 Style fidelity as predictability** (the Writing Workshop's
  fingerprints become a number) and **terminology-consistent translation
  post-edit** (bidirectional attention keeps a term rendered the same way
  across a document).

### G. Passive radar — handed to atkpr, not built here

Five of the papers are passive radar: the reference-free roadside
receiver, SkyWatch's multistatic network, uncalibrated MIMO detection,
uncertainty-aware Wi-Fi fusion, and the statistical analysis of ECA with an
imperfect reference. They belong to `atkpr`. The one diffusion tie: a
learned cleanup of ECA residue on the range-Doppler map, trained on the
residue model that last paper derives. Noted here; built there.

### H. Asked for, not diffusion: the hour log

Bill, 2026-10-08: *"I need an hour by hour automatic log producer that is
exportable … over the course of years, they always had me log what I did
over that hour."* Most of it is plumbing, and it belongs in ATK, not this
repo: every hour, read the O.W.L. ledger, worker logs, chat turns, RF
sessions and extraction runs, and write one entry derived **only from
recorded events**, each line clickable to its event, *"no recorded
activity"* when there was none, Bill's own annotations beside, exportable
to CSV, DOCX and PDF. Diffusion's part is F2: when an hour's facts complete
late (a transcript finishes at :20), revise the entry in place rather than
append a correction. Build the honest log first; the model polishes it.

### I. Passive sensing — presence, motion and vital signs

Bill likes both halves; the hardware decides how.

- **I1 Vital signs from Wi-Fi CSI** — breathing rate, heart rate, apnea
  events, contactless (the two papers Bill sent, §9). The 2019 work needs
  a discontinued Intel 5300 NIC; **PulseFi (2025)** does it with
  amplitude-only CSI from an **ESP32** (or a Raspberry Pi 4 with Nexmon)
  and a small LSTM, on 118 participants. So the ATK path is an ESP32 pair
  — transmitter and receiver, a few dollars each — streaming CSI over USB
  serial as a sensor; no SDR, and no laptop Wi-Fi card (Windows cannot
  extract CSI from one). PulseFi's pipeline as published: amplitude, DC
  removed, band-pass 0.1–0.5 Hz for breathing and 0.8–2.17 Hz for heart,
  Savitzky-Golay, windows, LSTM. Fresnel-zone placement from the 2019
  paper for where to put the pair. **Labeled plainly as a measurement
  with research-grade accuracy, never a diagnosis and not a medical
  device**; it can sit beside Medical Support without being part of it —
  a casualty's breathing, watched from across the room.
  *First experiment:* Bill's own breathing rate against a count, three
  postures, one ESP32 pair.
- **I2 Presence and motion through walls** — the passive-radar side: FM or
  Wi-Fi as the illuminator, the Kraken or the ESP32's CSI as the receiver,
  motion behind a wall or rubble as the target. Physics pushes back
  hardest here; it waits for the weak-burst denoiser (B3) and the vital
  signs pipeline (I1), whose motion detection is the same measurement
  with a smaller target.
  *First experiment:* a person walking in the next room, detected from
  CSI alone; then from the Kraken with an FM illuminator.

### J. HF propagation now — from the Kiwi network

Bill: *"I also like the HF propagation now concept."* Not diffusion, but
AURA-shaped: ATK already talks to KiwiSDRs. Decode WSPR and beacon
receptions from Kiwis in the regions that matter (open decoders; local
decoding from the Kiwi's audio stream, nothing sent anywhere) into a
**band-openness matrix** by hour — which HF bands are open *right now*
between here and there — painted on the map as reach arcs, beside the
VOACAP-style prediction for the same hour so the analyst sees measured
against predicted. *"Which band do I use to reach the team in the next
valley"* answered from measurement. Products in `rf_data\products\hf\`.
*First experiment:* one evening, Virginia to three chosen Kiwis; measured
openings versus the prediction.

### R. Research track

- **A beacon co-designed with its detector.** DARPA RFMLS task 2 and
  DeepSig's channel-autoencoder work: a waveform learned to be easy for
  *our* detector to find at low SNR through *this* receiver. For AURA, a
  relief beacon that the field laptop hears first. Mad science, on brand.
- **Generative classification** for modulation recognition (a denoiser
  per class; classify by which denoises best) — robust at low SNR; niche
  beside TorchSig's classifiers.
- **CyberWolf anomaly detection** by reconstruction error on flow features
  — only under CyberWolf's *context, never suppression* rule, and only
  after its false-positive history is understood to be the detectors being
  right.

## 5. Not kept — and why

- **Identification of people** by enhanced faces, plates or voices. The
  model invents the face. Never.
- **Medical facts.** Never.
- **Guessing redacted text.** Technically a masked-prediction task;
  ethically a tool for defeating redactions, and of doubtful analytic
  value. Deliberately not built.
- **Chat inline editing.** Trivial once F exists; not a track.
- **Replacing the Writing Workshop's multi-pass drafting.** It works; a
  diffusion version is a later experiment, not a track.

## 6. Build order

Each phase is useful alone and ends with a number.

- **Phase 0 — Foundation (A).** Profiles, layout, SigMF fields, refusal,
  resampler, impairment measurement, TorchSig in its venv, the dataset
  builder, the cabled loop. *Exit:* datasets for two profiles; one cabled
  set; the refusal fires; the domain-gap harness runs.
- **Phase 1 — First wins (D2, B3).** Speech enhancement before Whisper;
  the hook-burst denoiser experiment. *Exit:* WER with and without; SNR
  numbers against three classical denoisers.
- **Phase 2 — The detector (B1, B2, B4).** Trained per profile on
  synthetic + cabled; *AI proposes, decoder confirms* on the waterfall;
  augmentation kept only if it closes the gap. *Exit:* precision/recall on
  cabled real captures; the domain gap with and without augmentation.
- **Phase 3 — Repair (D1, D3, D4, D5).** *Exit:* DSD decodes recovered
  across injected dropouts; PRI analysis correct with inferred pulses
  flagged; OCR accuracy on degraded scans before/after.
- **Phase 4 — Emitter identification (C).** *Exit:* same-model radios told
  apart at field SNR, per profile, with the channel-resilience test.
- **Phase 5 — Posteriors and maps (E).** *Exit:* the cloud contains the
  true fix at the stated probability; radio-map error against held-out
  measurements.
- **Phase 6 — Text diffusion (F)** — when the llama.cpp PR merges;
  PyTorch experiments earlier. *Exit:* novelty-filter ranking agrees with
  an analyst's blind ranking; extraction speed and accuracy against the AR
  extractor.
- **Interleaved, by Bill's choice (2026-10-08), each after its
  prerequisite:** B5 point-and-ask (after Phase 0 — the cheap version
  needs only the primary model); E5 predicted reach (after Phase 0 — it is
  physics and DTED1, no training); A6 the translator (with Phase 2's
  cabled captures); B6 the hunter (after Phase 2's detector); C3 the
  social graph (after Phase 4); E3 where-am-I and E4 the synthetic
  aperture (with Phase 5); I1 vital signs (whenever the ESP32 pair
  arrives — independent of everything else); I2 after B3 and I1; J HF now
  (any time — Kiwi access and a decoder); **B7** cyclostationary
  processing and the signal cut — the classical parts (FAM/SSCA, the
  cyclic proposer, the cut, the three filters) **inside Phase 0**, since
  they need no model and are useful on day one; the SCF branch of the
  classifier with Phase 2. **On screen:** `DETECTION_DESIGN.md` §12.1 —
  sub-tabs with a box each, the waterfall untouched but for boxes and a
  right-click.

**Where the plans live.** This file is the toolkit's plan of record;
`DETECTION_DESIGN.md` is the detector in detail and carries its own build
steps (§12) and decisions. The ATK-side work both require — tabs, the
right-click, the subsystem airlock, `rf_data` in the recorder, the hour
log, map layers from products — is listed in ATK's own
`FUTURE_PLANS.md` under 2026-10-08. Palimpsest's plan is
`Palimpsest\PALIMPSEST_PLAN.md`; the host-side diffusion loop shared with
it is its §8.6. Athanor's spike list carries S7 and S8. Passive-radar
inputs (§4.G) are noted here and belong in `atkpr`, which is not yet
written to.
- **Research track (R)** — as time allows.

## 7. Evaluation

- **The domain gap** — the one number every RF track reports: trained on
  synthetic at the profile's rate, tested on cabled real captures from
  that receiver. Augmentation, impairment models and profile discipline are
  all judged by whether they shrink it.
- **Minutes to acceptable** — the environment version of the gap (§3.6):
  trained on a composed region, how many minutes of real on-site capture
  bring the detector to the acceptance line. The operational number; the
  one that says whether "do not loiter" is true.
- **Classical baselines first.** Every denoiser is scored against Wiener,
  median and wavelet; every detector against the energy detector ATK has;
  every inpainter against interpolation. A learned tool that does not beat
  the classical one is not shipped.
- **Hallucination rate.** For every reconstruction tool: how often does it
  produce a signal, pulse, word or character where the ground truth has
  none? Reported with every result; the number that keeps §2.1 honest.
- **Before/after on Bill's own captures and transcripts**, blind where a
  judgement is involved.
- **Published with the code, failures included.**

## 8. Licenses and dependencies

| Component | License | How it is used |
|---|---|---|
| TorchSig | MIT | Vendored, pinned, in its own venv |
| RF-Diffusion (mobicom24) | GPL-3 | A separate process at arm's length, fetched on enabling — or the time-frequency diffusion reimplemented from the paper under this repo's license |
| DiffusionGemma 26B-A4B | Apache 2.0 | The text backend, through the host-side loop (Palimpsest §8.6) |
| Speech enhancement model (SGMSE-class) | per model | Weights pinned by hash; license shown |
| DeepSig OmniSIG | commercial (~$100k; Bill) | **Not used.** The white papers are read for ideas; nothing of theirs is in this repo |
| PyTorch, CUDA | BSD / NVIDIA | In the venv; GPU shared under ATK's GPU lease and the one AI queue |

This repo's own license: **all rights reserved** (D1). Every dependency
is therefore checked for copyleft before it is used; `DETECTION_DESIGN.md`
§8 lists the detector-side choices.

## 9. The papers, mapped

`D:\Analyst_Toolkit\new papers` — Bill: *"not all … will be relevant, but
some aspect in each may have value, if repurposed."*

| File | Paper | Feeds |
|---|---|---|
| 2404.09140 | RF-Diffusion: time-frequency diffusion for raw RF generation (MobiCom '24) | B4 — augmentation; the TFD method |
| 2501.07030 | Erasing Noise in Signal Detection with Diffusion Model — SNR ↔ timestep | B3 — the theory for the denoiser-as-detector |
| 2602.03818 | DL denoising of radio signals for UHE cosmic-ray detection | B3 — weak pulses in noise; the weak-burst analogue |
| 2605.28457 | Noise suppression and RFI rejection for self-triggered radio detectors | B3 — triggering under non-stationary interference |
| 2503.05514 | Noise-robust RF fingerprint identification using denoise diffusion | C2 |
| 1904.07623 | DeepRadioID — channel-resilient radio fingerprinting | C1 — the channel confound and its answer |
| 2510.09775 | A generic ML framework for RF fingerprinting / SEI | C1 — the framework |
| DARPA RFMLS (page) | Fingerprinting, fingerprint enhancement, spectrum awareness, autonomous configuration | C, R — the beacon co-design; goal-driven attention over wide bands |
| 20205000256 | DeepSig — "Machine Learning Remakes Radio": channel autoencoders | R — AI-designed waveforms; background on DeepSig's approach |
| 2607.23014 | AI-empowered communication and radar modulation recognition — survey | B2 — the map of AMR methods |
| 2509.15603 | Blind source separation of radar signals in the time domain (Fraunhofer) | D3 — deinterleaving |
| 2305.13911 | Soft range information from RF data | E1 — all probable values, not one |
| 2608.09357 | ControlRadio — controllable diffusion for radio-map generation | E2 |
| Park, CVPR 2026 | Diffusion² — 3D environments into RF heatmaps | E2 |
| 1910.10817 | Passive radar at the roadside unit — a reference-free receiver | G → atkpr |
| 2305.18562 | SkyWatch — passive multistatic radar network | G → atkpr |
| 2402.16675 | Integrated MIMO passive radar detection, uncalibrated receivers | G → atkpr |
| 2407.04733 | Accurate passive radar via uncertainty-aware fusion of Wi-Fi sensing | G → atkpr; the fusion idea also fits E1 |
| 2601.20817 | Statistical analysis of ECA with an imperfect reference | G → atkpr; the residue model for the one diffusion tie |
| 2406.01622 | Survey of diffusion probabilistic models (biomolecules) | Theory reference; discrete-sequence diffusion is the bridge to F |

`new papers\additional` (added 2026-10-08):

| File | Paper | Feeds |
|---|---|---|
| 2508.19552 | CSRD2025 — open-source end-to-end simulation platform and synthetic radio dataset for spectrum sensing | A, §3.6 — a second generator beside TorchSig for composing wideband scenes; compare both on the domain gap |
| 2501.00282 | ReFormer — radio fakes by autoregressive generation over learned discrete RF tokens | B4 — augmentation; and the **RF tokenizer** idea (R) |
| 2602.14833 | RF-GPT — an RF language model: a visual encoder on spectrograms feeding an LLM | R — *point-and-ask* on the waterfall; the cheap version is Gemma 4's own image input |
| 2603.20692 | Agentic physical-AI for self-aware RF systems — agents per transceiver component | R — the self-configuring receiver (DARPA RFMLS task 4) |
| 2607.10930 | The Singularity Space — diffusion over pole-residue (complex-plane singularity) signal representations; keeps transients sharp | B3, D3 — a pulse-friendly representation for the denoiser and for pulse-train completion |
| 2509.15258 | Generative AI meets wireless sensing — towards a wireless foundation model (the RF-Diffusion group) | Background and the long view; the passive-sensing track (I) |
| 2201.00680 | Comprehensive survey of RF fingerprinting | C — the reference survey |
| 2505.03556 | Survey of large AI models for future communications | Background |

`new papers\additional\more` (added 2026-10-08):

| Paper | Feeds |
|---|---|
| 1908.05108 — Wi-Fi-based real-time breathing and heart rate monitoring during sleep (Hefei; Intel 5300 CSI; Fresnel-zone placement; 96.6% breathing, 94.2% heart rate) | I1 — the placement theory |
| 2510.24744 — PulseFi: low-cost cardiopulmonary and apnea monitoring from CSI (ESP32 / Raspberry Pi 4; amplitude-only CSI; LSTM; 118 participants) | I1 — the pipeline and the hardware |

Sent in chat, 2026-10-08 evening — Bill may add them to the folder:

| Paper | Feeds |
|---|---|
| 1609.09077 — RFI mitigation with a U-Net (ETH, radio astronomy) | B3, the signal cut's *clean* step — the canonical small-U-Net RFI mask and denoiser |
| 1701.00458 — Deep-HiTS: rotation-invariant CNN for transient detection in difference images | F1's signal twin / what-changed — transient detection on *difference* waterfalls between visits |
| 2002.05770 — Harvesting ambient RF for presence detection through deep learning | I2 — presence from raw ambient RF with an SDR, the non-CSI half |
| 2206.06637 — RF-Next: receptive field search for CNNs (PAMI) | The 1D classifier's receptive fields, searched rather than hand-set (`DETECTION_DESIGN.md` §4) |
| 2302.09854 — Faster R-CNN spectrum sensing and signal identification in cluttered RF (1D FRCNN) | B1 — a license-clean 1D region-based proposer; torchvision has Faster R-CNN (BSD) |
| 2305.09594 — HiNoVa: open-set detection for RF device authentication (LoRa, Wi-Fi) | C1 and the UNKNOWN threshold (`DETECTION_DESIGN.md` §4) |
| 2510.09663 — Adversarial-resilient RF fingerprinting: CNN-GAN for rogue transmitter detection | C — rogue detection; GAN augmentation for fingerprints |
| 2604.14987 — AI-enabled covert channel detection in RF receivers (Sorbonne) | A new cut analysis: *is something hidden inside this nominal signal?* — in-signal anomaly, R for now |
| 2607.16455 — Compact CNNs for AI-based drone detection (Ludovika, EW) | B — drone links on the class list; compact models for the CPU budget |
| 2607.24669 — Heterogeneous NN accelerator for multitask RF recognition: AMR + covert channel + GNSS jamming; LSDec learnable streaming decimator | B — multitask head; GNSS jamming class; the learnable decimator as a research counterpoint to fixed canonical rates |
| 2609.19279 — Radio-Frequency Convolutional Neural Networks (Duke / MIT) — CNN inference carried out in the RF domain | R — edge compute; reference |

Plus TorchSig itself (torchdsp/torchsig, MIT): 60+ signal types across FSK,
QAM, PSK, ASK, OFDM and analog families; perfect / cabled / wireless
impairment models; one dataset architecture for classification
(`num_signals_max = 1`) and detection; `DatasetCreator` to disk,
`StaticTorchSigDataset` back. Its rate is a parameter — set from the
profile, never typed.

## 10. Decisions

- **D1 — This repo's license.** ✅ **All rights reserved** (Bill,
  2026-10-08) — like ATK: source may be published for review; use by
  permission. Consequence: no AGPL or GPL component may be linked in;
  anything GPL runs at arm's length as a separate process, and
  Ultralytics YOLO (AGPL-3) is excluded outright (`DETECTION_DESIGN.md`
  §8).
- **D2 — The data root.** ✅ `D:\Analyst_Toolkit\rf_data\`, shared by ATK's
  RF path and this toolkit (Bill, 2026-10-08: *"I like the rf_data folder
  idea"*). Layout and the profile rule in §3.
- **D3 — TorchSig is the detector; DeepSig is not used.** ✅ Bill,
  2026-10-08.
- **D4 — Receiver profiles key everything RF; the sample-rate law.** ✅
  Bill, 2026-10-08; made concrete in §3.
- **D5 — The cabled calibration loop with the bladeRF.** ✅ Claude,
  2026-10-08, from Bill's hardware; Bill may veto.
- **D6 — What is kept and what is cut.** ✅ §4 and §5, Claude by Bill's
  delegation (*"Keep that which makes sense to you"*); Bill may veto any
  line.
- **D7 — The hour log is an ATK feature**, built in ATK with F2 as its
  revision step. ✅ Claude, 2026-10-08.
- **D8 — Environment profiles and the scene composer; adapt, do not
  loiter.** ✅ Bill, 2026-10-08 (the idea); the profile, composer and the
  *minutes-to-acceptable* measure, Claude, same day.
- **D9 — Bill's picks from the second round.** ✅ 2026-10-08: A6 the
  paired-capture translator; B5 point-and-ask with teach-it-a-signal; B6
  the self-hunting receiver; C3 the RF social graph; E3 where-am-I; E4
  the Kraken synthetic aperture; E5 predicted reach with DTED1 and
  propagation models; I1 vital signs from Wi-Fi CSI and I2 presence and
  motion; J HF propagation now. Not picked, left on R: the RF tokenizer,
  what-changed-since-last-visit (folded into F1's signal twin), generative
  classification, CyberWolf anomaly, the beacon co-design.
- **D10 — Transmit safety on the cabled loop.** ✅ Claude, 2026-10-08,
  from Bill's warning: bladeRF or HackRF only; cabled with a DC block;
  target −40 dBm; a 20 dB margin under the data-sheet maximum; the tool
  computes and refuses; first run at minimum gain (§4.A6).
- **D11 — Products outside the install; GeoServer optional.** ✅ Bill's
  requirement, 2026-10-08; §3.7.

## 11. Change log

- **2026-10-08** — Created from the design conversation (Bill and Claude):
  the mechanism table; principles including the Invented tier, receiver
  profiles, the sample-rate law and the two nevers; the RF data foundation
  with the storage layout, impairment model and cabled loop; tracks A–H
  and R; the cuts; build order; the domain gap and hallucination rate as
  the measures; licenses; the nineteen papers mapped; decisions D1–D7.
- **2026-10-08** — §3.6 environment profiles and the scene composer, from
  Bill's point that generated samples let a detector be trained for a
  region before anyone goes there; *adapt, do not loiter* as the rule and
  *minutes to acceptable* as its number (§7); D8.
- **2026-10-08** — D2 decided (`rf_data\`). The weak-burst test is framed
  on signals Bill can make or capture; nothing depends on CSAR radio
  samples he has not had access to in twenty years. Eight papers from
  `new papers\additional` mapped (§9).
- **2026-10-08** — Bill's picks written in as tracks with first
  experiments (D9): A6, B5, B6, C3, E3, E4, E5, I1, I2, J. §3.7 products
  outside the install, open formats, GeoServer optional (D11). Transmit
  safety on the cabled loop (D10). Two CSI vital-sign papers mapped; the
  ESP32 is the hardware for I1. Build order interleaved by prerequisite.
- **2026-10-08** — D1: all rights reserved; copyleft components excluded
  or kept at arm's length. The two CSI papers are in
  `new papers\additional\more`. The detector's design begins:
  `DETECTION_DESIGN.md`.
- **2026-10-08** — Cyclostationary processing and the signal cut (B7;
  `DETECTION_DESIGN.md` §4.1–4.2, D9–D10). Canonical rates per profile by
  integer decimation (detection D2). FCOS confirmed in place of YOLO
  (detection D3). Drone links and GNSS jamming on the class list. Eleven
  more papers mapped (§9).
- **2026-10-08** — Low-SNR escalation over an IQ ring buffer, and
  filtering with the cyclic detector's findings: matched filter
  parameters, FRESH (clean and separate), SCORE on the Kraken; the
  cycle-conditioned denoiser on R (`DETECTION_DESIGN.md` §4.3, D11).
- **2026-10-08** — Bill: build all three filters; sub-tabs with a box to
  turn each on (`DETECTION_DESIGN.md` §12.1, D12). B7's classical parts
  placed in Phase 0. *Where the plans live* added to §6; `README.md` as
  the index; ATK's `FUTURE_PLANS.md` and Athanor's plan updated to carry
  their sides.
- **2026-10-08** — Box styling on the waterfall by detection source
  (`DETECTION_DESIGN.md` §12.2, D13): cyan for cyclostationary, magenta
  for AI, grey for energy, solid in the technology colour when confirmed.
- **2026-10-09** — **Built**: the engine for every track in §4 (A, A6, B1–B7,
  C1–C3, D1–D5, E1–E5, F1–F4, H's revision step, I1–I2, J, R) with 1,195
  tests, and ATK's side (FUTURE_PLANS 2026-10-08 items 1–10). TorchSig
  2.2.0 verified on Python 3.11 (ATK's interpreter) generating at the
  profile's exact rate; a native generator beside it. Status and measured
  numbers: `docs/BUILD_STATUS.md`; the runbook: `docs/FIRST_EXPERIMENTS.md`.
  Corrections made while building: profile datatypes are the ones on disk
  (§3.1); the RTL's canonical rates are 48 k / 480 k / 2.4 M by the stated
  rule (`DETECTION_DESIGN.md` §4); the cyclostationary box colour moved from
  cyan to vermilion by §12.2's own rule (ATK's LTE cells are cyan); tiers use
  ATK's vocabulary — linear cleans are CLEANED, fills INFERRED, generative
  outputs INVENTED; `text/loop.py` is MIT (Palimpsest's loop, shared).
  Added: `products reach` on the command line (E5 on real DTED without
  writing Python); itmlogic (MIT) installed with the core half.
  Found while checking ATK: PySide6 6.12.0 aborts ATK on Python 3.11 after
  about a thousand signals — install.bat now holds PySide6 below 6.12 and
  self-tests what it installed.
