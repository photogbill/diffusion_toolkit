# AI signal detection — design of record

> Bill, 2026-10-08: *"the first thing I want to do is plan out the ideal
> AI Signal Detection using AI configuration. What methods speak for you?"*

| | |
|---|---|
| **Status** | Design 2026-10-08; **built 2026-10-09** — every build step in §12 and the on-screen design of §12.1–12.2 (in ATK). The learned parts are trained on Bill's GPU from his datasets; `docs/BUILD_STATUS.md` says what is measured. Companion to `ATK_DIFFUSION_PLAN.md` (tracks A, B, C3, B5, B6); this file is the detector in detail. §4.5 (audio ID) and §4.6 (radio-astronomy search methods) were planned 2026-10-28 and are not built. |
| **Goal** | On the live or recorded waterfall, at a receiver profile's exact rate, find every signal (where, when), say what it is (what), measure it (how), and hand what can be decoded to a decoder — on a laptop whose GPU the language model is also using, with an honest *unknown* and an honest confidence. |
| **Constraints** | Receiver profiles and the sample-rate law (plan §3) · the Invented tier: a detection is *Proposed* until confirmed (plan §2.1) · runs beside the cognitive core, so real-time inference is **CPU** (ATK's embedders-on-CPU rule, extended) and the GPU is for training · all rights reserved: no copyleft linked in (§8). |

## Contents

1. The shape — two proposers, a classifier, a confirmer
2. The front end — per-profile, noise-floor-relative
3. Stage 1 — proposers
4. Stage 2 — the cutout and the classifier; *unknown*; teach
4.1 Cyclostationary processing — a detector, a representation, a measurement
4.2 The signal cut — right-click to everything
4.3 Low-SNR escalation — the buffer, and filtering with what the cyclic detector found
4.4 Sifting — the deep scan for signals below the noise floor (LPI / LPD)
4.5 Audio ID — identify by how it sounds (planned)
4.6 Weak-signal search methods from radio astronomy (planned)
4.7 Below the floor — the detection_design folder (2026-10-30; extensions and one correction to §2)
5. Stage 3 — tracks and confirmation
6. Training — pretrain on real, supervise on synthetic, fine-tune on cabled
7. Deployment — ONNX on the CPU
8. Components and licenses
9. Classes, v1
10. Labels — SigMF annotations
11. Evaluation
12. Build steps
12.1 In ATK — on screen
12.2 The boxes — source is the edge, class is the caption, confirmation is the weight
13. Decisions

---

## 1. The shape

A single end-to-end network that takes wideband IQ and emits labels is the
seductive design and the wrong one here: it entangles *where* with *what*,
it cannot say "unknown" honestly, it needs retraining to learn a new class,
and nothing outside it can check it. The design that holds up in the field
— and the one TorchSig's dataset architecture is built for — separates the
questions, puts classical DSP on both sides of the learned parts, and lets
the decoders have the last word.

```
 IQ at the profile's rate
   │
   ▼
 FRONT END   STFT with the profile's fixed parameters; dB above the measured noise floor
   │
   ├──────────────► ENERGY PROPOSER   CFAR on the PSD — classical, always on, no training
   ├──────────────► CYCLIC PROPOSER   cyclic-feature detector on known cycle frequencies — below the floor
   │                                              │
   ▼                                              │
 AI PROPOSER  2D detector on the spectrogram tile → boxes (t, f) + family + confidence
   │                                              │
   └──────────────┬───────────────────────────────┘
                  ▼
 CUTOUT       shift · filter · decimate each box to a canonical narrowband rate (logged)
                  │
                  ▼
 CLASSIFIER   1D network on IQ → modulation/protocol + an embedding
                  │           prototypes → nearest class or UNKNOWN (open set)
                  │           classical measurement: bandwidth, symbol rate, PRI, duty
                  ▼
 TRACKER      boxes across frames → one object per signal over time
                  │
                  ▼
 CONFIRMER    a decoder (DSD · pager · multimon · ADS-B · …) confirms what it can
                  │
                  ▼
 Proposed → Confirmed → the watchlist, the social graph, point-and-ask, the hunter
```

Three proposers because they fail differently: energy catches anything
with power and nothing structured below the floor; the cyclic-feature
detector catches signals *below* the floor whose symbol or chip rate it
knows, and nothing it does not; the learned detector catches structure at
low SNR and can hallucinate. Where they disagree, all are shown. A decoder's confirmation is the only thing that turns *Proposed*
into *Confirmed*; a signal with no decoder stays *Proposed* with its class
and confidence, forever if need be.

## 2. The front end — per-profile, noise-floor-relative

- **STFT parameters are part of the profile.** FFT size, hop, window and
  the tile duration are fixed per receiver profile and written into the
  model card. A model never meets a spectrogram it was not trained on the
  geometry of. (At 2.4 MS/s with a 1024-point FFT a bin is 2.34 kHz; at
  20 MS/s it is 19.5 kHz — the same signal is a different picture. This is
  the sample-rate law seen from the detector's side.)
- **Inputs are dB above the measured noise floor**, not absolute dB. The
  floor comes from the profile's impairment measurement (plan §3.4) and is
  tracked slowly at run time. This one choice removes gain settings, LNA
  state and receiver sensitivity from what the model has to learn, and is
  the largest single domain-gap reducer there is. Absolute level is kept
  beside the tile for the measurement stage, never fed to the detector.
- **Tiles.** A tile is a fixed window of time × the full span (for example
  1 s × 2.4 MHz at 2.4 MS/s → ~1024 bins × ~2300 frames with a 1024-point
  FFT and no overlap; downsampled in time to a fixed 640-ish rows). Tiles
  overlap by a fraction so a burst on a boundary is seen whole by one of
  them.
- **Phase is kept.** The spectrogram is for *where*; the cutout for *what*
  works on the IQ, which still has the phase. Magnitude-only detection,
  IQ classification.

## 3. Stage 1 — proposers

**Energy (CFAR).** Cell-averaging CFAR on the per-frame PSD, in dB above
floor, with a per-profile false-alarm rate. Cheap, classical, explains
itself. Its boxes are time runs of contiguous bins above threshold.

**Cyclostationary (cyclic feature).** Noise is not cyclostationary;
modulated signals are — their symbol rate, chip rate, hop rate and carrier
show up as discrete cycle frequencies α in the spectral correlation
function (SCF), where noise averages to nothing. A cyclic-feature detector
scanning the α of the known classes (P25 and DMR at 4800 sym/s, POCSAG at
512/1200/2400, LTE's 14 kHz OFDM symbol rate — its cyclic prefix, seen at a lag
of 1/15 kHz (corrected 2026-10-09; it is not the 15 kHz subcarrier spacing) —
ADS-B's 1 Mb/s, and so on
from the class table) finds those signals **below the energy floor** —
the classical result that energy detection cannot match. Cost: a FAM or
SSCA pass per tile on the CPU, bounded by scanning only listed α. It
cannot find what it has no α for; that is what the other two are for.
§4.1 has the processing in full.

**The learned detector.** A 2D object detector on the tile that emits
boxes with a coarse **family** (FM-like · AM-like · FSK · PSK/QAM · OFDM ·
burst · spread/noise-like · unknown) and a confidence. Family, not
modulation: the spectrogram cannot tell QPSK from 8PSK and should not
pretend to; the IQ classifier does that. Architecture: an anchor-free
single-stage detector — **FCOS** or **RetinaNet** from torchvision (BSD),
or **RT-DETR** (Apache 2.0) if accuracy on small, thin boxes needs it —
with a small backbone (ResNet-18-class) so it runs on the CPU (§7).
Boxes are axis-aligned in (time, frequency), which is what signals are.

**Low-SNR mode (optional, plan B3).** When the analyst asks for it, or the
hunter (plan B6) is in a quiet band, the tile first passes through the
diffusion denoiser at the SNR-matched timestep ("Erasing Noise in Signal
Detection"), and the detector sees the denoised tile beside the raw one.
Not the default path: latency, and a denoiser can invent structure; every
box from this path carries the *denoised* flag.

## 4. Stage 2 — the cutout, the classifier, *unknown*, teach

**Cutout.** For each box: frequency-shift to baseband, low-pass to the
box's bandwidth with a guard, decimate to a **canonical narrowband rate**.
Bill, 2026-10-08: *"it will vary by hardware, I think that's unavoidable
… it might not be ideal to have a one size fits all solution."* Agreed,
and the clean rule is: **canonical rates are integer decimations of the
profile's own rate** — never a fractional resample — three per profile,
one per bandwidth class (voice-class · wideband digital · spread/OFDM).
Stated as a rule so it can be checked: for each class, the LOWEST exact
integer-decimated rate that is at least the class minimum — 48 kHz,
480 kHz, 2 MHz. A bladeRF at 4 MS/s has 50 k / 500 k / 2 M; a HackRF at
20 MS/s has 50 k / 500 k / 2 M by /400, /40, /10; an RTL at 2.4 MS/s has
48 k / 480 k / 2.4 M (corrected 2026-10-09 from a sketched 48 k / 240 k /
1.2 M: 240 kHz cannot hold a 250 kHz LoRa chirp with a guard, and 1.2 MHz
cannot hold a 2 MHz ADS-B signal). The rates fall out of the hardware, the classifier is per
profile anyway, and nothing is shared across profiles by assumption —
only through the translator (plan A6), measured. The decimation factor
and canonical rate are written into the cut's metadata.

**Classifier.** Two inputs, one network: a 1D convolutional branch on a
fixed-length IQ window (TorchSig's narrowband families are the reference
implementations; a plain 1D ResNet with receptive fields chosen by search
rather than habit — RF-Next, §9 of the plan) and a 2D branch on the cut's
**spectral correlation function** (§4.1), fused before the head. Three
outputs: the modulation/protocol class, a **normalized embedding** of the
signal, and regressed **cycle parameters** — symbol rate, carrier offset —
supervised directly, since synthetic data knows them exactly. The SCF
branch is what holds accuracy up at low SNR; the IQ branch is what tells
QPSK from 8PSK.

**Open set — honest *unknown*.** Classification is by **nearest prototype**
in embedding space: each known class has one or more prototypes (the mean
embedding of its examples, per profile); a cutout whose distance to the
nearest prototype exceeds that class's threshold is **UNKNOWN**, and the
threshold is set on held-out data so that unknown-signal rejection is
measured, not hoped. The field always has signals that were not in
training; a detector that forces a class is a detector that lies.

**Teach-it-a-signal is a prototype, not a retrain.** The analyst marks a
signal and names a class; its examples are embedded and become a prototype
(or extend one). No training run, no GPU, seconds. The model card lists
the class as *taught* with its example count; a class with fewer examples
than the floor says so, and its threshold is wider. When taught classes
accumulate, a scheduled fine-tune of the embedding network on the whole
labeled set folds them in properly, with the prototype path as the
fallback in between.

**Measurement, classical.** Occupied bandwidth from the cutout's PSD;
symbol rate, chip rate and carrier offset from the cyclic domain profile
(§4.1); burst length, PRI and duty from the track (§5); SNR in dB above
floor. These are numbers the bench can show and an analyst can check; they
are not the model's opinion. Where the classifier's regressed cycle
parameters disagree with the classical ones, the classical ones are shown
and the disagreement is logged.

## 4.1 Cyclostationary processing — a detector, a representation, a measurement

Bill, 2026-10-08: *"incorporate cyclostationary detection and processing
into some of the training data, and give access to run signal cuts of RF
through cyclostationary / AI trained on it."* It earns three places.

- **The processing.** The spectral correlation function S(f, α) estimated
  by the FFT accumulation method (FAM) or the strip spectral correlation
  analyzer (SSCA) — both classical, both O(N log N), both CPU — on a cut
  at its canonical rate; and from it the **cyclic domain profile**, max
  over f of |S(f, α)|, a one-dimensional curve over α whose peaks *are*
  the symbol rate, chip rate, hop rate and (at 2·f_c for BPSK-class) the
  carrier offset. Implemented once in numpy (no copyleft dependency
  exists worth taking), with the FAM parameters (N, N′, window, α
  resolution) part of the profile so a model never meets an SCF of a
  geometry it was not trained on.
- **As a detector** (§3): scanning the listed α of known classes finds
  them under the noise.
- **As a representation for the AI** (§4): the SCF image is the second
  input to the classifier, and the cycle parameters are regressed targets.
  Synthetic training data therefore carries, for every signal, its true
  symbol rate and carrier offset in the labels (`atk:symbol_rate`,
  `atk:carrier_offset_hz`), and the dataset builder computes and caches
  the SCF of every cut so training does not recompute it. The SCF is
  famously robust to noise and to the receiver's gain — exactly the
  invariances the domain gap wants.
- **As a measurement** (§4): the numbers an analyst can read off a plot.
  The cut viewer shows the SCF as an image and the cyclic profile as a
  curve, with the peaks labeled.
- **What it does not do.** It needs a few hundred symbols to resolve a
  cycle frequency, so very short bursts fall to the learned detector; it
  scales with the α list, so an unknown rate is found only by a full scan
  (offered as *deep scan*, slow, on a cut, never on the live tile); and an
  SCF can be cleaned of nothing — it is an analysis, not a filter.

## 4.2 The signal cut — right-click to everything

Bill, 2026-10-08: *"with a right click from the waterfall, and then
process the result via any tools that can process that signal type, be it
demodulator, decoder, DF, or even just saving the original and the cleaned
up one in the same folder for offline analysis."* This is the workflow
the whole detector exists to serve; built in ATK, with this repo as its
engine.

**Right-click a box, a track or the VFO → Cut signal…** and then, in
order, each step visible and skippable:

1. **Cut.** The IQ for the box's time × frequency with margins; shift,
   filter, integer-decimate to the profile's canonical rate for the box's
   bandwidth class; saved as SigMF `original` with full provenance —
   profile, source capture and sample range, box, decimation factor, who
   cut it and when. Nothing about the original is ever changed; the
   write log hashes it.
2. **Analyze.** The SCF and cyclic profile (§4.1); classical measurements;
   the classifier's class, confidence and embedding, or UNKNOWN; the
   fingerprint if a library exists (plan C). All of it into
   `analysis.json`, the SCF into `scf.png` and `scf.npy`.
3. **Clean** (optional, chosen by the analyst). An RFI mask and
   interpolation (the U-Net approach, §9 of the plan — ETH's RFI
   mitigation), a Wiener filter, or the diffusion denoiser (plan B3) —
   producing SigMF `cleaned`, **Invented tier**, with the method, its
   parameters and the model's hash in the metadata. `original` and
   `cleaned` sit in the same folder, always; a decoder run on `cleaned`
   labels its output as decoded from a reconstruction.
4. **Route.** The menu offers every tool that accepts the cut's class —
   and only those:
   - a **demodulator** (ATK's NFM / AM / SSB) → audio, the headphone
     monitor, Whisper;
   - a **decoder** — DSD for P25/DMR/NXDN, the pager decoder, multimon,
     ADS-B → confirmation (§5) and content;
   - **DF** — the box's time and frequency to atkdf for a bearing on the
     Kraken (if the cut came from a Kraken capture, DF runs on the cut
     itself);
   - the **Signals bench** — pulse, PRI, intra-pulse;
   - **point-and-ask** (plan B5) with the analysis attached, and
     **teach** (name the class);
   - **fingerprint** (plan C) — add to the library or match against it;
   - **save only** — the folder is the product.
   The route taken, the input chosen (`original` or `cleaned`) and the
   tool's result are appended to the cut folder, so a cut tells its whole
   story.
5. **Offline.** The folder is self-contained — SigMF pairs, `analysis.json`,
   the SCF, `report.md` written from the facts — under
   `rf_data\<profile>\cuts\<utc-stamp>_<center-hz>\`, readable without
   ATK, importable into a new install, and the unit the RF social graph
   and the watchlist point at.

Also reachable from the hunter (plan B6), which cuts what it finds, and
from a recorded capture's waterfall as well as the live one.

**The `video` route (planned 2026-10-30; `ATK\docs\SIGDIGGER_PARITY_PLAN.md`
§V).** Status: PLAN.
- **What it adds:** a route that turns an analog-video cut into a picture.
- **Toolkit side:**
  - `drone_fpv_analog` gets `decoders=("video",)` in `detect/classes.py`;
  - `cut/route.py` gets `ROUTE_WORDS["video"] = "the analog video picture
    — a locked picture CONFIRMS the class"`.
- **ATK side:**
  - `ROUTE_LABELS["video"] = "Analog video picture"`;
  - the handler opens ATK's video viewer (`atk\ui\video_view.py`, on
    atkdsp's video kernel) on the cut's **original** record. It never uses
    the cleaned one unless the operator picks it, and the cleaned-input
    warning applies as for every route.
- **Confirmation:** a picture with field lock and the right line count is a
  decode, so it **CONFIRMS** the class under the confirmer rule. H-lock with
  no fields confirms nothing, and the route's history line says why.
- **Not in this change:** a generic "analog video transmitter" class for any
  band (0.9 / 1.2 / 2.4 GHz cameras, Bug Sniffer's VIDEO). It would change
  the trained class set and needs its own data. The native synthesiser's
  `fpv_analog` generator is the starting point if Bill wants it.

## 4.3 Low-SNR escalation — the buffer, and filtering with what the cyclic detector found

Bill, 2026-10-08: *"if a different detector is running, and the SNR is
below a certain threshold, it uses a buffer to run cyclostationary
detection. BUT, can it use that information to improve SNR by filtering
out noise as well?"* Yes, and yes.

**The buffer.** The PTT scanner's pre-roll ring, generalized: the last
*N* seconds of raw IQ at the profile's rate (a dial; 2–10 s; ~5 MB/s at
2.4 MS/s, 40 MB/s at 20 MS/s). **Escalation triggers:** a proposer's
confidence in the ambiguous band; a region's SNR estimate below the
profile's threshold; the hunter (plan B6) dwelling in a quiet band; or the
analyst asking. On a trigger, the cyclic detector runs over the buffered
seconds, not the one-second tile. This is the one proposer whose depth
grows with observation time — the SCF estimate's variance falls as
1/(T·Δf) — so ten seconds of buffer finds what no tile can. A cycle
frequency found this way becomes a box with its integration time on it.

**Filtering with it — three classical mechanisms, in the order of what
they give.** Every one produces a `cleaned` SigMF, Invented tier, with
the method, α, and the **measured** before/after SNR in its metadata;
nothing is claimed that is not measured on the cut.

1. **Matched filtering, because now the baud is known.** The cyclic
   profile gives the symbol rate and the carrier offset precisely — the
   parameters a matched filter and timing recovery need — and the matched
   filter is the maximum-SNR linear receiver per symbol. The unglamorous
   answer that delivers most of the gain for most signals:
   cyclostationary → parameters → the right demodulator settings. The
   route step (§4.2) passes them to the demodulator and the decoder.
2. **FRESH filtering — the cyclic Wiener filter (Gardner).** A
   cyclostationary signal is correlated with frequency-shifted copies of
   itself at its cycle frequencies; noise is not. A FREquency-SHift filter
   combines the signal with copies shifted by ±α and filters them jointly
   (a linear periodically time-varying filter), so the signal's redundant
   spectral components add coherently and the noise does not. Blind
   adaptive forms need only α — which the detector just produced. Sized
   honestly: against white noise the gain is bounded by the signal's
   **spectral redundancy** — the excess-bandwidth roll-off for PSK/QAM,
   more for BPSK/ASK with conjugate cyclostationarity, more again for
   spread spectrum — so a few dB for a typical narrowband digital signal.
   Where it is dramatic is **co-channel interference**: two signals on
   top of each other with different baud rates separate cleanly, each
   having cycle frequencies the other lacks; nothing time-invariant can
   do that. Offered as *separate* as well as *clean*.
3. **SCORE on the Kraken — blind cyclostationary beamforming (Agee,
   Schell, Gardner).** Five coherent channels make this the big one. The
   SCORE family adapts the array weights to maximize the output's
   self-coherence at α — no direction of arrival, no array calibration,
   no training sequence, only the symbol rate. The array steers itself
   onto the signal with that baud and nulls what does not share it: up to
   ~7 dB of array gain for five channels plus interference nulling that
   can reach tens of dB. A real SNR improvement from information the
   cyclic detector produced, on hardware already owned. Lives beside DF in
   atkdf's territory; the cut carries all five channels when it came from
   the Kraken.

**Then the learned one** (research track): the cyclic parameters as
*conditioning* for the diffusion denoiser (plan B3) — reconstruct the
signal consistent with this symbol rate — a learned FRESH filter, judged
against the classical three on the same cuts.

**After cleaning**, the learned detector and classifier re-run on the
cleaned cut, flagged *post-clean*; a classification that appears only
after cleaning is shown with that flag, never silently promoted.

## 4.4 Sifting — the deep scan for signals below the noise floor (LPI / LPD)

Bill, 2026-10-12: *"an automated 'sifting' mode that is specifically for
looking for and automatically recording signals below the noise floor. Its
an LPI/LPD detector. The Operator would set a bandwidth and a list of
frequencies to hop through, and it would take long listens looking for
signals … As our deep scan I guess. I assume we would need it to
automatically engage a longer buffer to ensure it could record any
detections."* This is the *deep scan* §4.1 promised ("an unknown rate is
found only by a full scan") and never built. Engine
`atk_diffusion/cyclo/sift.py`; in ATK, RF / Radio → **Sift**
(`atk/ui/sift_page.py`, threads in `atk/core/sift_worker.py`).

**What is different from the cyclic proposer.** The proposer (§3) tests
the symbol rates and prefix lags the class table lists, and nothing else.
Sifting is **blind**: every cycle frequency from a few hertz to the
channel's width, at several channel widths, conjugate and non-conjugate.
It promises "a modulated signal with a clock at α is here, at this
frequency" — the only finding an LPI search can promise — and leaves the
naming to the cut (§4.2) and the classifier.

**The five decisions.**

1. **Record first, decide after.** A detection exists only at the END of a
   dwell, so a buffer engaged on a detection is always too late: the
   buffer IS the recording. Every dwell is written to disk as it arrives,
   in the radio's own bytes, under `rf_data\<profile>\runs\sifts\<run>\`
   (never C:\, never the user profile); the search runs on the file; a
   dwell with nothing new is deleted. One with something NEW is **held**
   while the same stop is listened to again; both are **kept** — as SigMF
   pairs in `captures\sift\<run>\`, one annotation per feature — only if
   the second look reproduces it. Two independent looks square the
   false-alarm probability and give a longer recording of anything real.
   The second look asks only "is THAT line still here?": inside each held
   line's window its threshold is set by the cells in the windows alone
   (thousands, not millions), and a carrier may have moved up to 0.5 ppm
   between looks (both oscillators drift; an α bin is a few parts per
   billion at RF). The disk budget covers every sift with that radio, and
   a first look and its confirmation are kept together or not at all.
   A full queue or a gap reported by the radio ENDS the dwell (said, and
   the stop done again); bytes are never dropped from a recording's middle.
2. **Segmented integration, a derived threshold.** Coherent integration
   pays only while a feature stays in one α bin (1/T); clock and LO drift
   end that within seconds, so a dwell is cut into coherent segments
   (default 2 s) whose lag-domain statistics are **summed**. Per segment,
   per α bin, per lag the statistic is exponential under the null, so the
   sum of K is **Gamma(K, 1)** after the local-median (CFAR-in-α)
   normalisation, and the threshold is `gammainccinv(K, Pfa / trials)` with
   every channel, resolvable α cell and lag counted (Bonferroni). Measured
   on noise through the real channelizer: mean 4.999 and variance 4.999
   for K = 5, tail rates 9.98·10⁻³ / 1.05·10⁻³ / 1.21·10⁻⁴ at nominal
   10⁻² / 10⁻³ / 10⁻⁴. Coherently, each doubling of dwell buys ~1.5 dB;
   past the segment length ~0.75 dB.
3. **Several channel widths.** Cyclic sensitivity is set by IN-BAND SNR,
   so a narrow signal searched inside a wide channel loses √(W/B). The
   stop is resampled once to 2.5× its bandwidth (a rational polyphase
   resampler, so the cost follows the bandwidth, not the radio: a bladeRF
   at 4 MS/s and a HackRF at 20 MS/s both search a 1 MHz stop at
   2.5 MS/s), the widest level filtered directly and the narrower ones cut
   by the proposer's WOLA channelizer at widths a factor of 4 apart (the bandwidth, ÷4, ÷16, …,
   down to 12.5 kHz), half-overlapping, each channel sampled at 2.5× its
   width — the margin that keeps a transition-band carrier's conjugate
   line from aliasing into the searched |α| ≤ width (measured otherwise:
   a BPSK at +93.75 kHz reappeared in the channel at 0 Hz, aliased to
   −145.8 kHz). Conjugate lines are matched by their **absolute carrier**
   (channel centre + α/2), never by α, which depends on the channel. A
   rate is merged within one width first (overlapping channels: one run,
   their union its place) and only then across widths, so a wide channel
   cannot chain separate emitters together; a harmonic is folded into a
   rate only when it is seen inside that rate's place.
4. **The environment is the real false-alarm source.** Below the floor a
   blind search finds every periodic thing nearby. The threshold controls
   chance only; every feature is therefore classified, first match wins:
   **receiver** (learned with the antenna replaced by a 50 Ω terminator,
   per receiver profile, matched by offset from the tuned centre; saved
   only by a calibration that finished every dwell — a cancelled one keeps
   the stored calibration) · **dc** (a carrier line at the tuned centre —
   the radio's DC offset looks the same) · **local** (the same α at three
   or more stops further apart than the bandwidth: one wideband emitter
   fills many narrow channels of ONE stop and is not local by itself) ·
   **background** (seen at that stop during the baseline pass; carriers
   matched within 2 ppm, hours of drift) ·
   **visible** (channel SNR ≥ 0 dB and measurably so — at or above the
   floor is what the waterfall shows; a perfect radiometer over a dwell of
   white noise "measures" −15 dB, which no waterfall shows) · **explained**
   (α matches a class-table rate or harmonic) · **new**. Only NEW is
   recorded by default; nothing is hidden — every class is listed on the
   Sift page in its own colour (gold for new and kept, muted for what the
   place and the radio explain).
5. **False recordings per day, not a Pfa.** The operator says how many
   false recordings a day chance may cause; with confirmation, F = D·p₁·p₂
   for D dwells a day, split evenly, the first look capped at 2 % (each
   false first look costs one confirmation dwell).

**Before it starts, it says what it will do** (`SiftPlan.words`): the
revisit time; the segments and widths; the sensitivity — a measured κ
scaled by the algebra (K·κ·L·η = t − K + z₀.₉√K): for 1 MHz stops of a
4 MS/s bladeRF, 30 s dwells, a QPSK-like (RRC 0.35) symbol-rate line
filling its channel is found to about −20 dB in-band SNR in a 1 MHz
channel and −11 dB in a 15.6 kHz one, and a BPSK-like carrier line about
6 dB deeper; the expected false recordings and confirmation dwells a day;
the disk a dwell takes and the memory the analysis needs (every width the
machine searches at once, counted together — an upper bound); and, measured on
the machine (one segment of noise through the real search), the analysis
time — when it is longer than the dwell, it sets the pace and the plan
says so.

**The schedule.** Stops in order; a NEW finding queues a confirmation of
that stop at the front; a near miss (family p < 0.1) earns one relook per
stop per pass; the hopper waits while two dwells wait for analysis. The
first pass (configurable, 0–5) learns the place. **Calibrate** listens
with the terminator fitted at up to three of the stops. A retune the RF
panel refuses is said and skipped (a whole list refused ends the run); a
driver that confirms a retune late has the dwell restarted and its wait
raised; the radio's stream taken away (Stop on the Spectrum page) ends the
run within a second; and a result that arrives after its run ended counts
against nothing. Every hop,
dwell, analysis and decision is a line in the run's `sift_log.jsonl`, and
ATK's own log (`atk.sift`) carries the same at INFO.

**What it cannot do** (`sift.LIMITS`, said on the page): a burst shorter
than a few hundred symbols leaves no line; a frequency hopper is seen only
in the fragments that land at a stop while it is listened to (bandwidth
beats a hopper, not dwell); a waveform built without cyclic features —
dithered timing, noise-like spreading — can hide from it, and against that
only a carefully calibrated radiometer helps. A signal filling only part
of a channel scores less than the reference (measured: a third of a
channel, about half the matched κ); the levels exist so some channel is
within a factor of 4 of every signal.

**Reviewed** (2026-10-12) by an agent that had not seen it written: nine
defects, each fixed and pinned by a test (`tests/test_cyclo_sift.py` §9,
ATK's `tests/test_sift_page.py`). The one physics limit it measured stays:
a carrier drifting WITHIN a dwell loses margin (13.9 → 4.8 for 4 Hz over
4 s) — shorter coherent segments are the answer, and the plan says so.

**Next, not built:** accumulating a persistent emitter's statistics across
revisits (incoherent, so hours of total listening count); drawing a kept
finding's box on the waterfall when the radio is back on its stop; and on
the KrakenSDR, SCORE (§4.3) pointed at a finding's α for a bearing on an
emitter the waterfall cannot show.

## 4.5 Audio ID — identify by how it sounds (planned, optional, last to run)

Bill, 2026-10-28: *"I think it should be an optional method, only to be
used with signals with clear audio characteristics as it can help identify
a noisy signal via sound. But its definitely one of the last detectors to
hit."* His long-held idea behind it: operators identify a great many
signals by ear, so an AI that listens should augment the
spectrogram-based detectors.

**Status: planned 2026-10-28, not built.** To be built after the RFMLS
Training tab work (Setup → RFMLS Training; `library`, ARCHITECTURE §4.6)
settles. Nothing in this section exists in code yet.

**What it adds — and what it does not.** The audio is a demodulated,
narrowed view of the I/Q the classifier already has, so it carries no new
information. What it adds is a different *representation*: an ear-like,
multi-resolution view of timing and tone — symbol-rate buzz, burst rhythm,
FSK warble, sync and tone squelch, the texture of analog voice — which a
single-resolution tile blurs. Part of that (symbol rate, frame rate) the
cyclic profile (§4.1) already measures; the tonal and voice texture
nothing else in the chain looks at. Its second gift is access to
audio-domain models and to audio-only recordings (open decision below).

**Where it sits.** Not a proposer: it never finds a signal, it only
re-identifies a box something else drew. The chain: energy/CFAR proposes →
the learned detector boxes → the classifier labels (§4) → *if the
classifier is unsure*, audio ID (this section) and cyclostationary
escalation (§4.3) run, each on its own trigger → a decoder confirms (§5).

**Triggers — all three, or it does not run.**

1. The classifier is unsure: its confidence is in the ambiguous band, or
   the nearest prototype is near its threshold (§4).
2. A candidate class is **audio-capable** (below).
3. The box's SNR is above the measured demod floor for the mode (below).

The analyst can also ask for it from the box's right-click menu. Off by
default; a box in *AI Detect* turns it on.

**Audio-capable is a per-type flag the grades must earn.** `type.json`
gains `audio` (default false), set by the operator for types with a real
sound. A flagged type is trained with and without the audio stage on the
same real examples — the same mechanism as `compare_generated` — and if the
stage does not raise the grade on the kept-aside real originals, it stays
off for that type and the tab says so with the two numbers.
**Voice-width types only** (`canonical == "voice"`). Wideband and very
wide types play back as hiss, and their burst cadence is already the
cyclic detector's job.

**The demod recipe is locked, like the sample rate.** Several demods run
side by side on the cut — AM envelope, FM discriminator, USB — each with a
fixed filter, AGC and audio rate. The audio rate is an integer decimation
of the voice-class canonical rate (48 kHz / 3 = 16 kHz on the RTL; 50 kHz
/ 3 ≈ 16.7 kHz on the bladeRF and HackRF); a model that needs exactly
16 kHz gets one fractional resample, written into the recipe. The whole
recipe goes into the model card, and a recipe mismatch is a refusal, the
same as a profile mismatch.

**Training — from the same library, with the same honesty.** Every example
in the RFMLS library yields its audio on the fly; the I/Q stays the one
record. Roles carry over: audio from a kept-aside original is kept aside;
audio from an AI-generated copy trains and never grades; the tier follows
the parent. Any cache lives under the receiver's `rf_data` folder, never
C:\ or AppData. First model: a small log-mel CNN per profile, ONNX on the
CPU (D4). A pretrained audio encoder is a candidate to try against it with
the same with/without comparison — weights fetched by the installer so it
runs offline, and its license checked against §8 first.

**Its floor.** It beats noise by hearing a pattern repeat over seconds, so
it runs on the §4.3 ring buffer (3–5 s to start; the right length is
measured), not on a tile. Below the demod threshold it fails — FM hardest,
because the discriminator turns to hiss — so each mode's floor is measured
on the library's *Harsh* dirty copies (accuracy against SNR) and the stage
declines to run below it. Below that floor is §4.3's and Sift's (§4.4)
territory.

**Its weight.** An opinion, never a confirmation. On the box: a `♪` badge
with the mode (e.g. `♪ FM: DMR`) in a fourth source colour, chosen by the
same search as §12.2 (not picked yet). Agreement with the classifier raises
the box's confidence; disagreement is shown as disagreement and logged, and
neither wins — a decoder does (§5).

*Where:* AI Detect → **Audio ID**.

> **Hover:** "Identifies a signal by how it sounds once demodulated — the
> way an operator does by ear. It only runs when the AI is unsure and the
> signal is strong enough to demodulate. **Use it when** a type with a
> clear sound (voice, FSK tones, digital-voice buzz) keeps coming up as
> unsure. **How:** tick the box; it listens to a few seconds of the buffer
> and adds a ♪ badge saying what it heard. It is an opinion — a decoder
> still has the final word. **Not for** wide signals (LTE, Wi-Fi), which
> only sound like hiss."

**Logging.** Every trigger, recipe, verdict and timing at INFO on its own
logger, and into the cut folder when it ran on a cut.

**Open decision — Bill's call.** Audio-only recordings from other receivers
(signal-ID reference libraries, other operators' recordings) can train only
this stage. After demodulation the receiver matters less, but not zero, and
using them bends the per-receiver rule (plan §3). Options: (a) own receivers
only; (b) allowed as a separate, labelled source (`atk:source =
outside_audio`) that trains and never grades. Not decided.

## 4.6 Weak-signal search methods from radio astronomy (planned)

Bill, 2026-10-28: *"What other radio astronomy techniques can we
incorporate? … it seems like the ideal place to fish for methods of finding
or recording / processing low powered hard to see signals."* And on the
shortlist: *"I think the ones that you recommend, we should add. Even if
they are only for specific scenarios, just make sure that when they hover
over that option, that text appears explaining when and how it should be
used."*

**Status: planned 2026-10-28, not built.** The kernels are atkdsp's Part C
(`atkdsp/docs/DSP_EXPANSION_PLAN.md`, C1–C6). The Kraken-side items —
null-and-DF-again, self-calibration, aircraft as calibrators — are
`atkdf/docs/ATKDF_PLAN.md` §13; passive radar's use is
`atkpr/docs/ATKPR_PLAN.md` §6A.18. This section is how the detector and
Sift use them and what the operator sees.

**What transfers and what does not.** Radio astronomy's *search over
unknown parameters* (drift, period, width) and its *noise statistics*
transfer; imaging does not — it assumes a sky that holds still for hours
and many antennas. Our targets are brief, intermittent and moving, on
8-bit radios, and every method below is judged against that.

**Every option has hover text.** Each method is a box, off by default, on
the sub-tab named with it. Hovering over the box shows the text drafted
here — what it finds, when to use it, how, and what it cannot do — in the
18-year-old-operator voice of the RFMLS Training tab. The drafts are part
of the design: change the method, change its text.

**1. Spectral kurtosis (SK)** — `atkdsp_sk` (C1). Gaussian noise scores 1,
impulsive or bursty energy above 1, constant-envelope signals below 1.
Three uses: (a) the noise floor (§2) leaves out bins SK flags, so bursty
signals cannot pull it up — `floor.estimate` keeps its quantile method and
gains the mask; (b) an `SK↑` badge on any box whose energy is impulsive —
pulses, sparks, switching supplies, short bursts that averaging dilutes;
(c) excision of impulsive cells before every long integration — Sift's
dwells and the §4.3 buffer here, atkdf's covariance and atkpr's CPI there.
Blind spot: OFDM, LTE and anything noise-like look Gaussian to it.
*Where:* AI Detect → **Spectral kurtosis**.

> **Hover:** "Finds energy that comes and goes — pulses, clicks, sparks,
> short bursts — that averaging hides. **Use it when** you are after
> something that keys up briefly or a noisy electrical device (a power
> supply, a cheap bug), or when the noise floor keeps jumping. **How:** tick
> the box. Bursty boxes get an SK↑ badge, and bursty moments are kept out of
> the noise floor and out of long listens. **It cannot see** signals that
> sound like steady hiss (LTE, Wi-Fi, spread spectrum)."

**2. Drift-rate search** (SETI's Taylor de-Doppler tree) —
`atkdsp_drift_tree` (C2). Sums power along slanted lines in the waterfall,
so a narrow carrier that drifts keeps its full integration gain instead of
smearing across bins. Cheap oscillators warming up drift; so do moving
transmitters, drones and low-orbit satellites. In Sift the same tree runs
across the per-segment statistics, summing along drifting α tracks instead
of one α bin — the answer to the drift loss §4.4's review measured
(13.9 → 4.8 for 4 Hz over 4 s); what it recovers is measured on that case,
not claimed. A large drift rate is a linear sweep, so sweep jammers and
LFM chirps fall out of the same search. The rate goes on the box and into
the fingerprint features (atkdsp B8).
*Where:* Sift → **Drift search**; AI Detect → **Drift-rate search** (long
buffers only).

> **Hover:** "Finds a weak, narrow signal whose frequency slowly slides — a
> cheap transmitter warming up, something moving, a satellite. Normal
> averaging smears it out; this follows the slide. **Use it when** you are
> listening a long time for faint carriers, bugs or beacons, or a signal
> you can hear will not stay in one place. **How:** tick the box and choose
> how fast a slide to look for (faster costs more). Each find shows its
> drift in Hz per second. **Not for** quick looks — it needs several
> seconds of listening to pay off."

**3. ON/OFF cadence** (SETI's interference test). A real target appears
when the antenna or beam points at it and disappears when it points away;
local interference stays either way. Alternate ON and OFF dwells — a
directional antenna turned, a Kraken beam steered, or in a bug sweep
inside the room and then outside — in Breakthrough Listen's ABACAD order
(three ONs between three different OFFs), and keep only what follows the
cadence every time, so a coincidence has to happen three times. A
procedure, not a kernel: it drives Sift's dwells and compares their
findings.
*Where:* Sift → **ON/OFF cadence**; the Bug Sniffer (ATK) when it is built.

> **Hover:** "Tells a signal coming FROM one place apart from everything
> else around you. **Use it when** you suspect a particular room, building
> or direction — a bug sweep, or checking whether a weak signal really is
> over there. **How:** tick the box; ATK tells you when to point ON (at the
> place, or inside the room) and when to point OFF (away, or outside),
> three times each. Only signals that appear every time you are ON and
> vanish when you are OFF are kept. Six listens; hold still during each."

**4. Periodic search** (pulsar methods) — `atkdsp_periodic` (C3).
Harmonic summing finds narrow repeating pulses; folding at a candidate
period stacks pulses too weak to see one at a time; the fast folding
algorithm covers long periods (about half a second and up); a small
acceleration search lets the period drift for a moving emitter. It works
*below* single-pulse detection, so it complements `pulse_runs` and the
bench's PRI tools rather than repeating them. The bench's trap applies:
score the comb, report the fundamental.
*Where:* Sift → **Periodic search**; Cuts → **Find a repeat**.

> **Hover:** "Finds something that repeats on a schedule — a beacon, a
> radar, a timed burst — even when each pulse is too weak to see, by
> stacking the pulses on top of each other. **Use it when** you suspect a
> regular transmitter (a beacon every few seconds, a radar sweep, a device
> that pings) or a cut looks like noise but might be pulsing. **How:** tick
> the box and listen for at least ten repeats. Results show the period and
> how strong the stacked pulse is. **Not for** signals that key up at
> random."

**5. Single-pulse search** (fast-radio-burst method) —
`atkdsp_boxcar_ladder` (C4). Matched filters at widths a factor of 2 apart,
from one frame to seconds; the best width wins, and a burst of any length
lands within ~1.5 dB of a matched filter where a fixed tile dilutes a short
burst and splits a long one. A proposer beside CFAR; its boxes carry the
width.
*Where:* AI Detect → **Burst-width search**.

> **Hover:** "Finds single faint bursts of any length — a key-up, a data
> blip, a one-off transmission — that the normal detector misses. **Use it
> when** you are waiting for something that transmits once, briefly.
> **How:** tick the box; it runs on the live waterfall, and each box shows
> how long the burst lasted. **Costs** a little CPU — turn it off if the
> waterfall lags."

**6. Faint-streak search** (AOFlagger's SumThreshold, turned around) —
`atkdsp_sumthreshold` (C5). Flags runs of 1, 2, 4, 8 … cells along time or
frequency whose average clears a threshold that falls as the run grows.
Astronomers use it to find faint streaks so they can delete them; here the
streaks are the find — long weak carriers and short broadband clicks that
cell-by-cell CFAR misses. Thresholds are measured on terminated and
quiet-band captures, not borrowed.
*Where:* AI Detect → **Faint-streak search**.

> **Hover:** "Finds faint lines in the waterfall — a weak signal that stays
> on for a long time, or a short click across many frequencies — too dim,
> square by square, for the normal detector. **Use it when** the waterfall
> looks empty but you suspect something faint and steady is there. **How:**
> tick the box; it marks the streaks it finds on the live waterfall."

**7. Correlation radiometer** (radio interferometry) —
`atkdsp_xcorr_radiometer` (C6), KrakenSDR only. Cross-correlate the
coherent channels: each channel's own receiver noise does not correlate
and averages away, so only energy reaching both antennas survives. That
sidesteps the noise-level uncertainty that caps every single-receiver
energy detector, and fills the gap `sift.LIMITS` states — a waveform with
no cyclic features "can hide from it, and against that only a carefully
calibrated radiometer helps." Limits, said on the page: noise that reaches
every antenna (city noise, nearby electronics) still correlates; the
Kraken's shared-clock spurs and crosstalk correlate too and are mapped
first with terminated inputs, as Sift maps the radio's own features;
sensitivity still grows only as √(B·T). The pair phases are a rough
bearing, passed to atkdf as a wide, low-weight line of bearing.
*Where:* Sift → **Correlation radiometer (Kraken)**.

> **Hover:** "For a transmitter built to look like plain noise. Two
> antennas hear the same signal, but each radio's own hiss is different —
> so comparing them cancels the hiss and leaves the signal. **Use it when**
> Sift's normal search finds nothing but you still suspect something
> (spread spectrum, noise-like, low probability of intercept). **How:**
> KrakenSDR only. Run **Calibrate** once per gain setting with terminators
> on the inputs, put the antennas back, tick the box. Expect minutes, not
> seconds. **It cannot** separate a signal from noise that comes from
> everywhere (city noise, nearby electronics)."

**Shared rules.** Twin-first in atkdsp; every threshold derived or
measured, never tuned; every search with a negative control in its test;
every trigger, setting, finding and timing logged at INFO; every finding
*Proposed* until a cut, the classifier or a decoder says more.

**On the waterfall.** These are long-listen searches, so their boxes share
one new source colour for the family, chosen by the §12.2 palette search
(not picked yet), with a badge naming the method and its number: `SK↑`,
`drift 1.2 Hz/s`, `↻ 2.0 s`, `▮ 12 ms`, `streak`, `×corr`.

**Not added, and why.** CLEAN for the detector — it belongs to passive
radar (`ATKPR_PLAN.md` §6A.4); dedispersion — nothing disperses
terrestrial RF, and the drift tree covers sweeps; imaging — five elements
give too few antenna pairs; VLBI — needs atomic clocks.

## 4.7 Below the floor — the detection_design folder

**Status: design, 2026-10-30, not built.** The work on pushing detection
below the noise floor — Bill's OmniSIG question — lives in its own folder,
`D:\Analyst_Toolkit\detection_design`, and extends §4.3, §4.4 and §4.6.
Read its `README.md` first. The documents: `BELOW_THE_FLOOR.md` (the
physics, the SNR wall, seven directions incl. the sync-word bank, track-
before-detect with hop-period folding, deep tiles, the learned noise
model and its classical comparator, the Chinese-methods bench on demand);
`UNKNOWN_FIRST.md` (every detection is a U-ID; cuts in training-library
folders under the ID; identification after the fact as a registry
binding); `HINDSIGHT.md` (the slow detectors — Sift, the sync bank, the
decoders — label the tiles the fast one could not see); `GAP_READING.md`
(reconstructing what a thin paper leaves out, as a prediction);
`RADIO_ASTRONOMY_2021_2026.md` (twelve methods from the field's last five
years beyond §4.6, and the Ma et al. 2025 deployed FRB detector);
`LITERATURE_NOTES.md`; `BUILD_ORDER.md` (phases, repos, gates).

**One correction to this document now, so nobody builds against it:**
§2 says the tile is "downsampled in time to a fixed 640-ish rows." That
line is **withdrawn** (`RADIO_ASTRONOMY_2021_2026.md` §16, decision D10):
time downsampling washes out short bursts — a 1 ms burst at 2.4 MS/s
becomes less than one row — so the tile is **cropped into time slices and
stacked along the channel dimension** at native resolution, with three
dynamic-range views as input channels and a learned mask layer in front
of the backbone (D11), the whole front end exported inside the ONNX graph
(D12). The rest of §2 stands. When the folder's pieces are built, their
sections move here and the copies there are marked *moved*.

## 5. Stage 3 — tracks and confirmation

**Tracker.** Boxes from successive tiles are associated (Hungarian
assignment on time-frequency overlap and family) into **tracks**: one
object per signal over time, with its first-seen, last-seen, duty cycle,
PRI for bursts, and its frequency drift. A hop pattern is a track whose
boxes move. The PTT scanner's existing notion of a key-up becomes a track
of family *burst* with the voice decoder as confirmer.

**Confirmer.** A track whose class has a decoder is handed to it — DSD for
P25/DMR/NXDN, the pager decoder for POCSAG/FLEX, multimon for the rest it
knows, the ADS-B path — and a decode upgrades the track to **Confirmed**,
with the decoded content attached. Disagreement (classifier says DMR, DSD
decodes P25) is shown as disagreement, and the decoder wins the label.

**Downstream.** Confirmed and Proposed tracks feed the watchlist, the RF
social graph (plan C3, once fingerprints exist), point-and-ask (plan B5,
which gets the track's class, measurements and confirmation state), and
the hunter (plan B6, whose eyes these are).

## 6. Training — pretrain on real, supervise on synthetic, fine-tune on cabled

1. **Self-supervised pretraining on real, unlabeled captures — per
   profile.** Unlabeled IQ is the one thing there is an unlimited supply
   of: record. Masked-spectrogram modeling for the 2D backbone (mask
   patches of the tile, reconstruct), and denoising / contrastive
   pretraining for the 1D backbone. This teaches the networks what *this*
   receiver in *this* environment looks like before a single label
   exists, and it is the second-largest domain-gap reducer after
   noise-floor normalization. The diffusion denoiser (plan B3) is itself a
   self-supervised model of the same data and can share the backbone.
2. **Supervised training on synthetic scenes** generated by TorchSig (and
   CSRD2025 for comparison) at the profile's exact rate, through the
   profile's impairment model and, for a target region, the environment
   profile's scene composer (plan §3.6). Labels are free and exact.
3. **Fine-tuning on the cabled set** (plan §3.5) — real receiver, perfect
   labels — and, when a region is reached, on the first minutes on site
   (plan §3.6).
4. **Calibration.** Temperature scaling of the detector's and classifier's
   confidences on the cabled set, so that "0.8" means right four times in
   five. The open-set thresholds are set here too.
5. **The card.** Every model ships with: profile, STFT geometry, FAM
   geometry, canonical rates and decimation factors, class list (trained /
   taught), dataset hashes, the domain gap,
   detection-versus-SNR curves per class, false alarms per hour on empty
   captures, unknown-rejection rate, CPU latency per tile. A model without
   a card does not load.

## 7. Deployment — ONNX on the CPU

The cognitive core owns the GPU under ATK's lease and the one AI queue; a
detector that watches a live waterfall cannot stand in that queue. So
inference runs on the **CPU** through ONNX Runtime, which also keeps the
deployed model free of a PyTorch dependency inside ATK. Budget: one tile
per second of capture, under 200 ms on the 14-core CPU for the detector
and under 20 ms per cutout for the classifier — set by the choice of
backbone and tile size, and measured as part of the card. DirectML is the
optional GPU path on Windows when the core is not loaded. Training is
PyTorch on the GPU, in this repo's venv, never inside ATK.

## 8. Components and licenses

All rights reserved means no copyleft may be linked in. Checked before
use:

| Component | License | Use |
|---|---|---|
| TorchSig | MIT | Synthetic data, impairments, the narrowband classifier references |
| torchvision detection models (FCOS, RetinaNet, Faster R-CNN) | BSD-3 | The 2D proposer, first choice |
| RT-DETR (Hugging Face Transformers implementation) | Apache 2.0 | The 2D proposer if small thin boxes need it |
| **Ultralytics YOLOv5/v8/v11** | **AGPL-3** | **Excluded.** The common default for spectrogram detection — and TorchSig's wideband examples lean on it — but AGPL would reach ATK. Not used, not vendored, not called as a process for training either, since the trained weights would carry the obligation in spirit if not in law. |
| YOLOX (Megvii) | Apache 2.0 | Acceptable alternative to the above |
| ONNX Runtime | MIT | Inference on the CPU inside ATK |
| PyTorch | BSD | Training |
| scipy / numpy | BSD | CFAR, cutout DSP, cyclostationary measurement |
| CSRD2025 | per its repo — check | Second scene generator; data only if the license allows |
| RF-Diffusion | GPL-3 | Augmentation only, as a separate process, or reimplemented (plan §8) |

## 9. Classes, v1

What Bill can actually capture around Nokesville and make with his own
hardware, plus what TorchSig can synthesize. The field will add the rest
through *teach*.

- Narrowband FM voice (amateur, business, public safety analog) ·
  P25 · DMR · NXDN (DSD confirms) · POCSAG · FLEX (the pager decoder
  confirms) · ADS-B 1090 (the air picture confirms) · FM broadcast ·
  NOAA weather · LoRa (915 MHz ISM chirps) · LTE / 5G NR downlink
  (structure, not content) · Wi-Fi / Bluetooth LE at 2.4 GHz (bladeRF,
  HackRF) · ATSC · a bladeRF-generated reference set of TorchSig families
  (FSK, PSK, QAM, OFDM, ASK, analog) · **drone links** — analog FPV video
  at 5.8 GHz and the digital control/video links at 2.4 and 5.8 GHz
  (HackRF profile; the Ludovika compact-CNN paper) · **GNSS jamming** at
  L1 (RTL, bladeRF; classified per the Sorbonne multitask paper — and the
  trigger for where-am-I, plan E3) · noise, spurs and the DC spike as
  explicit negatives.
- Not in v1: anything Bill cannot capture or make (no CSAR radios — plan
  change log), HF modes except through a Kiwi (a different profile).

## 10. Labels — SigMF annotations

Labels are **SigMF annotations**: each labeled signal is an entry in the
capture's `.sigmf-meta` `annotations` array with `core:sample_start`,
`core:sample_count`, `core:freq_lower_edge`, `core:freq_upper_edge`,
`core:label`, plus `atk:family`, `atk:snr_db`, `atk:source`
(`synthetic` · `cabled` · `taught` · `confirmed`) and, for confirmed
tracks, `atk:decoder`. The standard format, readable by anything, and the
same file that holds the capture — a label never drifts from its data.
TorchSig's dataset metadata is converted to and from this on the way in
and out.

## 11. Evaluation

- **Detection versus SNR, per class** — the curve that is the product.
- **mAP** on the cabled set and on held-out synthetic, reported separately
  (their difference is the domain gap).
- **False alarms per hour** on empty captures: terminated input, and real
  quiet bands.
- **Unknown rejection rate**: held-out classes never trained on; how often
  they are called UNKNOWN rather than forced.
- **Confirmation agreement**: how often the classifier's class matches the
  decoder's.
- **Latency** per tile and per cutout on the CPU.
- **Teach**: accuracy on a taught class from 5, 10 and 25 examples, on the
  next day's captures.
- **Baselines**: CFAR alone; the energy detector ATK already has.

## 12. Build steps

1. Front end: per-profile STFT geometry, floor measurement and tracking,
   dB-above-floor tiles, tile overlap. Unit-tested against synthetic tones
   at known SNR.
2. CFAR proposer; wire it to the existing waterfall as *Proposed* boxes.
   (Useful on its own; nothing learned yet.)
2a. Cyclostationary processing: FAM/SSCA in numpy, the cyclic profile,
   the cyclic-feature proposer on the class table's α; the cut viewer's
   SCF plot. Tested on synthetic signals of known symbol rate at low SNR.
2b. **The signal cut** — right-click, cut, analyze, clean, route, save
   (§4.2) — wired to the tools ATK already has (demodulators, DSD, pager,
   multimon, ADS-B, atkdf, the bench). Useful before any model exists.
2c. Low-SNR escalation: the IQ ring buffer and its triggers; cyclic
   detection over the buffer; matched-filter parameters to the
   demodulator; a FRESH filter (clean and separate); SCORE on Kraken cuts
   (§4.3). Each with measured before/after SNR on synthetic cuts at known
   SNR, then on cabled captures.
2d. Sifting — the deep scan (§4.4): blind segmented search at several
   widths, the ledger, record-first dwells with confirmation, the plan in
   words; RF / Radio → Sift in ATK. Built 2026-10-12; the null, κ and the
   sensitivity model are held to the engine by `tests/test_cyclo_sift.py`.
3. Labels: SigMF annotation read/write; TorchSig ↔ SigMF conversion.
4. Datasets: TorchSig synthetic at the profile's rate with its impairments;
   one cabled set (plan §3.5).
5. The 2D proposer (torchvision FCOS, ResNet-18): train on synthetic,
   fine-tune on cabled, export to ONNX, measure CPU latency. Wire beside
   CFAR; show disagreement.
6. Cutout DSP and canonical rates; the 1D classifier; prototypes and the
   open-set threshold; *teach*.
7. Classical measurement (bandwidth, symbol rate); the tracker; the
   confirmer hooks into DSD, the pager decoder, multimon, ADS-B.
8. Self-supervised pretraining on real unlabeled captures; re-measure the
   domain gap with and without.
9. Calibration; the model card; the refusal on profile mismatch.
10. Low-SNR mode with the diffusion denoiser (plan B3), flagged.

## 12.1 In ATK — on screen

Bill, 2026-10-08: *"in a way that doesn't crowd the RF screen, maybe their
own tabs and just a box to turn it on."* So the waterfall stays a
waterfall; everything new is a sub-tab of the RF workspace with one box
at the top of it, off by default, and the only things that touch the
spectrum view are what it already draws plus one menu item:

- **On the waterfall itself:** *Proposed* and *Confirmed* boxes (it draws
  boxes with a per-technology colour and caption today — these are more
  of the same, with a dashed edge for *Proposed*), and **right-click →
  Cut signal…** (§4.2). Nothing else.
- **AI Detect** sub-tab — one box: *AI detection on*. Under it, a box per
  proposer (Energy · Cyclic · Learned), the escalation buffer's length and
  triggers (§4.3), the loaded model's card (profile, classes, domain gap),
  and the refusal line when the capture's profile does not match. Latency
  and the proposer disagreements are shown here, not on the waterfall.
- **Cuts** sub-tab — the cut folders for the current profile: the viewer
  (SCF image, cyclic profile with labeled peaks, measurements, class or
  UNKNOWN, fingerprint), an *original / cleaned* switch, the **Clean**
  actions (matched-filter parameters → demodulator; FRESH clean and
  separate; SCORE when the cut has five channels) with measured
  before/after SNR beside each, and the **Route** buttons for the tools
  that take this class. Point-and-ask and Teach are buttons here.
- **Hunt** sub-tab (plan B6, later) — one box: *hunting*; the goal in
  words; the log of every retune and why.
- **Setup & Config** — the toolkit as an optional subsystem through
  ATK's airlock, with `rf_data` location, the receiver profiles and the
  cabled-loop tool (plan §3.5, with its transmit-safety arithmetic).

The same three filters are also offered from the Signals bench on any cut
it already has, so nothing is reachable from only one place.

**Hover text on the new options** (Bill, 2026-10-28). Every option added
by §4.5 and §4.6 carries hover text saying what it finds, when to use it,
how, and what it cannot do, in plain words; the drafts are in those
sections and change with the method.

## 12.2 The boxes — source is the edge, class is the caption, confirmation is the weight

Bill, 2026-10-08: *"specific color coded detector boxes on the waterfall in
the event the signal detection is an AI signal detection, and another if
it's a cyclostationary assisted processing result … the bright flashy
pretty things are all they will see … the better it looks, the more it
will be liked."* The lesson is honest here, because the colour carries
real information — where a detection came from. One rule protects what
the waterfall already means: today a box's colour is its **technology**
(P25, DMR, pager …, with the caption). Source must not fight that on the
same box. So:

| Source of the box | Edge | Badge | First appearance |
|---|---|---|---|
| Energy (CFAR) | thin, grey-white, dashed — the quiet baseline | none | no pulse (it would be constant) |
| **Cyclostationary** — found or assisted by the cyclic detector (§3, §4.3) | **vermilion** `#ff5400`, dashed (was cyan — see below) | `α`, with the integration time when it came from the buffer | one 600 ms edge pulse |
| **AI** — the learned detector (§3) | **magenta**, dashed | `AI`, with a small confidence bar | one 600 ms edge pulse |
| **Audio ID** (§4.5) — *planned* | a fourth source colour, dashed (not picked yet) | `♪` with the demod mode | one 600 ms edge pulse |
| **Radio-astronomy searches** (§4.6) — *planned* | one family colour, dashed (not picked yet) | the method and its number: `SK↑` · `drift 1.2 Hz/s` · `↻ 2.0 s` · `▮ 12 ms` · `streak` · `×corr` | one 600 ms edge pulse |
| Two or three proposers agree | alternating vermilion/magenta dash | badges stack | one pulse |
| **Confirmed** by a decoder (§5) | **solid**, heavier, in the technology's own colour (today's scheme) | ✓ — the source badges **stay**, so the box reads *"AI found it, DSD confirmed it"* | one brighter pulse |
| Appeared only after a clean (§4.3) | hatched inner edge | `post-clean` | — |
| Profile mismatch / model refused | no boxes; the refusal line in *AI Detect* | — | — |

- **Vermilion and magenta.** The rule written here first — *if the
  technology palette ever gains a cyan or magenta, the source palette
  moves, not the technology one* — applied on the day it was built: ATK
  already draws an LTE cell in cyan (`#3fd0e0`, CIEDE2000 5.5 from the
  plan's cyan — the same colour to any eye), so a cyan dashed box would have
  read as a tower. Vermilion `#ff5400` was chosen by search against the
  technology palette, the waterfall ramp, the processing green and the two
  common colour-vision deficiencies (`detect/boxes.py`, its tests); magenta
  `#ff00dd` was already the best AI colour available.
- **The pulse is one pulse**, on first appearance, never continuous; a
  box that is re-detected frame after frame does not keep flashing. It
  can be turned off in *AI Detect* for the people who hate it.
- **A legend** — a chip row in the waterfall's corner (edge styles and
  badges) — toggled from *AI Detect*; on by default for the first week,
  then it remembers.
- **Hover** on a box shows the full story: proposers, confidence,
  integration time, class, measurements, confirmation, and *cut* as the
  first action.
- The same styling is used in the Cuts viewer and in any exported image
  (reports, the capabilities deck), so a screenshot explains itself.

## 13. Decisions

- **D1 — Two proposers, a classifier, a confirmer; classical DSP on both
  sides.** ✅ Claude, 2026-10-08 (*"what methods speak for you"*); Bill
  may veto.
- **D2 — Canonical narrowband rates.** ✅ Per profile, by **integer
  decimation of the profile's own rate**, three per profile by bandwidth
  class; never fractional; never shared across profiles by assumption
  (Bill, 2026-10-08: they vary by hardware and should; the integer rule,
  Claude).
- **D3 — The 2D proposer family**: ✅ torchvision **FCOS** first; RT-DETR
  (Apache) if thin boxes need it; torchvision Faster R-CNN is the 1D
  region-based alternative the UMass paper validates; **Ultralytics
  excluded** (§8). Confirmed by Bill, 2026-10-08: *"whichever permissive
  model you recommend in place of YOLO."*
- **D4 — Inference on the CPU via ONNX Runtime; GPU for training only.**
  ✅ Claude, from ATK's rules.
- **D5 — Inputs in dB above the measured floor; STFT geometry in the
  profile.** ✅ Claude.
- **D6 — Open set by prototypes; teach = add a prototype.** ✅ Claude.
- **D7 — Tile size and duration per profile.** Pending measurement in
  build step 1.
- **D8 — Class list v1.** §9, pending Bill's additions; drone links and
  GNSS jamming added 2026-10-08.
- **D9 — Cyclostationary processing** as a third proposer, the
  classifier's second input, and the cut's measurement; synthetic labels
  carry symbol rate and carrier offset. ✅ Bill's idea, 2026-10-08;
  placed by Claude.
- **D10 — The signal cut** (§4.2): right-click → cut, analyze, clean,
  route, save; `original` and `cleaned` always together; every route
  logged into the cut folder. ✅ Bill, 2026-10-08.
- **D11 — Low-SNR escalation and cyclostationary filtering** (§4.3): the
  IQ ring buffer with its triggers; cyclic detection over the buffer;
  three classical filters — matched (parameters to the demodulator),
  FRESH (clean and separate), SCORE on the Kraken — each with measured
  gain; the cycle-conditioned diffusion denoiser on the research track.
  ✅ Bill's question, 2026-10-08; the mechanisms, Claude.
- **D12 — Build all three filters; on screen as sub-tabs with a box to
  turn each on; the waterfall gains only boxes and a right-click**
  (§12.1). ✅ Bill, 2026-10-08.
- **D13 — Box styling** (§12.2): source is the edge (grey energy,
  vermilion cyclostationary — cyan until 2026-10-09, moved by this rule's
  own clause because LTE is cyan — magenta AI), class is the caption in today's technology
  colours, confirmation is a solid heavier edge with the source badges
  kept; one pulse on first appearance; a legend. ✅ Bill's requirement,
  2026-10-08; the scheme, Claude.
- **D14 — Sifting** (§4.4): an operator's frequency list and bandwidth,
  long listens, a blind cyclostationary search with a derived threshold,
  every dwell recorded and kept only if something NEW is in it and is
  seen again; the environment classified, not thresholded. ✅ Bill's idea,
  2026-10-12 (*"an LPI/LPD detector … as our deep scan"*); the design,
  Claude.
- **D15 — Audio ID** (§4.5): optional and last to run; an escalation on
  three triggers, never a proposer; voice-width types the operator flags,
  kept only where the with/without grade on kept-aside real originals shows
  it helps; the demod recipe locked in the model card; trained from the
  RFMLS library with its roles; an opinion on the box, below a decoder.
  ✅ Bill's idea and placement, 2026-10-28; the mechanisms, Claude.
  **Planned, not built.** Outside audio-only recordings: pending Bill.
- **D16 — Radio-astronomy search methods** (§4.6): spectral kurtosis,
  drift-rate search, ON/OFF cadence, periodic search, single-pulse search,
  faint-streak search and the Kraken correlation radiometer — each an
  option off by default with hover text saying when and how to use it;
  kernels in atkdsp Part C; one family colour on the waterfall. ✅ Bill,
  2026-10-28 (*"the ones that you recommend, we should add … make sure
  that when they hover over that option, that text appears"*); the
  shortlist and the mechanisms, Claude. **Planned, not built.**
