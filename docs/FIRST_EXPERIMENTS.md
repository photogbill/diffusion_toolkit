# First experiments — the runbook for Bill's machine

Every track in `ATK_DIFFUSION_PLAN.md` §4 names a first experiment with a
number. Each one is a function in `atk_diffusion/experiments/` and a command
on `atkdiff.bat`. **Every one of them already runs here on synthetic
stand-ins** (that is how the code was proved); this file says what to point
each one at on your machine so the number becomes a real one.

Order follows the plan's build order (§6). Commands are typed in
`D:\Analyst_Toolkit\ATK_Diffusion_Toolkit` (open a Command Prompt there).
Everything a command writes goes under `D:\Analyst_Toolkit\rf_data\`; every
command says what it did, in words, and ends with the path of its report
(`report.md` for you, `result.json` for the next program).

`atkdiff.bat experiment --list` prints this list with what each needs;
`atkdiff.bat <command> --help` prints every option.

---

## 0. Once — install and check

1. In `D:\Analyst_Toolkit\ATK`, run **`get_diffusion.bat`**. It installs the
   toolkit's classical half into ATK (`envs\atk_core`: ONNX Runtime and
   tifffile) and builds the training environment `envs\atk_diffusion` from
   ATK's own Python 3.11: PyTorch with CUDA 12.6, TorchSig 2.2.0, ONNX.
   About 3 GB the first time; cached in `bin\pipcache` after that.
   `get_diffusion.bat /core` does only the ATK half (minutes, not gigabytes).
2. `atkdiff.bat status` — every line should be `[OK]`, and the GPU line
   should name the 3080 Ti. If it says *CPU-only build*, run
   `get_diffusion.bat /rebuild`.
3. Start ATK. **RF** now has *AI Detect*, *Cuts*, *Hunt* and *CSI Sensing*
   tabs; **Setup & Config** has *Diffusion Toolkit*; **Analysis Suite** has
   *Hour Log*; the **Geospatial** map has a *Layers* tab; **Network Link**
   has *Import RF social graph…*. Each new RF feature has its box, OFF.
4. `install.bat` rebuilds `envs\` from scratch: run `get_diffusion.bat`
   again after it (from the cache it downloads nothing).

## Phase 0 — the RF data foundation (track A)

**Profiles.** One per receiver at each rate you use:

    atkdiff.bat profile new rtlsdr_2400000_cu8
    atkdiff.bat profile new bladerf1_4000000_ci16
    atkdiff.bat profile new hackrf_8000000_ci8
    atkdiff.bat profile new krakensdr_2400000_cu8_ch0      (…ch1 to ch4)

ATK's **● Record I/Q** now files every capture under
`rf_data\<profile>\captures\` and writes the profile into the file
(`atk:receiver_profile`); the status line says where it went.

**Impairments (plan §3.4).** Put a 50 Ω terminator on the antenna port,
record 30 seconds in ATK, then:

    atkdiff.bat impair measure "D:\Analyst_Toolkit\rf_data\rtlsdr_2400000_cu8\captures\<file>.sigmf-data" --serial <serial>

Noise floor shape, DC spike, IQ imbalance, effective bits and spurs go into
`profiles\rtlsdr_2400000_cu8.json` and from then on into every synthetic
dataset of that profile. Re-measure when the device, firmware or gain
preset changes.

**Safe input (needed before the cabled loop).** From each receiver's data
sheet — never from memory:

    atkdiff.bat profile set-safe-input rtlsdr_2400000_cu8 --max-dbm <value> --source "<sheet name, revision, page>" --date 2026-10-10

**Datasets at the profile's exact rate.**

    atkdiff.bat synth throughput --profile rtlsdr_2400000_cu8
    atkdiff.bat synth narrowband --profile rtlsdr_2400000_cu8 --name nb1 --classes nfm,dmr,p25,pocsag,lora --n-per-class 2000 --generator torchsig
    atkdiff.bat synth wideband   --profile rtlsdr_2400000_cu8 --name wb1 --scenes 500 --env us-va-nokesville

`--generator native` uses the built-in numpy generator instead of TorchSig
(same labels, no TorchSig needed). `--env us-va-nokesville` composes scenes
from the US allocation tables and the Virginia cellular band plan — *a
prior, not the place*.

**The refusal.** Run the RTL detector on a bladeRF capture
(`atkdiff.bat detect <bladeRF capture> --learned`, after Phase 2): it must
refuse with *"…was trained for the RTL-SDR at 2.4 MS/s; this capture is the
bladeRF 1.0 (x40/x115) at 4 MS/s"*.

**The cabled loop (plan §3.5, D10) — the one transmit exception.** Cable,
DC block and fixed attenuators between the bladeRF (or HackRF) and the
receiver; never an antenna. Nothing transmits until the last step, and only
with both confirmations:

    atkdiff.bat cabled check  --tx bladerf1 --rx-profile rtlsdr_2400000_cu8 --freq 433920000 --tx-gain <g> --tx-power-dbm <dBm at that gain> --tx-power-source "<where that number came from>" --attenuation-db 50 --cabled --dc-block
    atkdiff.bat cabled txfile --tx bladerf1 --rx-profile rtlsdr_2400000_cu8 --preset
    atkdiff.bat cabled plan   ... (same numbers) ...

`check` and `plan` print the arithmetic (TX power − attenuation − splitter −
cable = expected input; target about −40 dBm; refused above the data-sheet
maximum minus 20 dB) and the exact command. The first run is always at
minimum TX gain; record the receiver in ATK while it plays, then

    atkdiff.bat cabled measure <the recording> ...   (the level, before any gain goes up)
    atkdiff.bat cabled align   <the recording> ...   (labels from the transmit manifest)
    atkdiff.bat synth cabled --profile rtlsdr_2400000_cu8 --name cab1 <aligned recordings>

The same steps are on **Setup & Config → Diffusion Toolkit** with the
arithmetic shown as you type. For the Kraken, feed it through a splitter
and put the splitter loss in (`--splitter-loss-db`, `--splitter-ways 5`).

**Exit number:** the domain gap, after Phase 2 —

    atkdiff.bat experiment domain-gap --profile rtlsdr_2400000_cu8 --model <trained model> --synthetic-dataset wb1 --cabled-dataset cab1

**Environment version (§3.6):** train on `--env us-va-nokesville` scenes,
record a few minutes from the yard as an on-site dataset, then

    atkdiff.bat experiment minutes --profile rtlsdr_2400000_cu8 --model <model> --onsite-dataset <yard dataset>

— *minutes on site to acceptable*, the number that says whether "do not
loiter" is true.

## Phase 1 — first wins (D2, B3)

**D2 speech before Whisper.** Put 20–50 DSD/PTT audio clips in a folder,
and type what was said in each into a text file, one line per clip:
`<clip file name><TAB><what was said>` (or a JSON `{clip: text}`):

    atkdiff.bat experiment wer --clips <folder> --references <refs.txt> --transcriber "<a command that prints the transcript of {wav}>" --noise-clips <a folder of channel noise with no speech>

Word error rate with and without each enhancer (spectral subtraction,
MMSE-LSA, the learned enhancer if installed), and a **hallucination check**:
does the enhancer make Whisper hear words in noise-only clips?

**B3 the weak-burst test.** Record 30 s terminated (or an empty band) per
profile, then

    atkdiff.bat experiment weak-burst --profile rtlsdr_2400000_cu8 --noise-capture <that capture>
    atkdiff.bat train denoiser --profile rtlsdr_2400000_cu8 --noise-capture <captures…> --captures <ordinary captures…>
    atkdiff.bat experiment weak-burst --profile rtlsdr_2400000_cu8 --noise-capture <capture> --denoiser <the model>

Detection versus SNR for a burst the denoiser never saw, against Wiener,
median and wavelet, plus the false-alarm and hallucination rates on noise
alone. The learned row only appears with `--denoiser`; a learned tool that
does not beat the classical ones is not shipped (plan §7).

## Phase 2 — the detector (B1, B2, B4)

    atkdiff.bat train ssl        --profile rtlsdr_2400000_cu8 ...        (optional: pretrain on your own unlabeled captures)
    atkdiff.bat train proposer   --profile rtlsdr_2400000_cu8 --dataset wb1
    atkdiff.bat train classifier --profile rtlsdr_2400000_cu8 --dataset nb1 --held-out lora
    atkdiff.bat train calibrate  --profile rtlsdr_2400000_cu8 --model <classifier> --dataset cab1
    atkdiff.bat export onnx <model folder>
    atkdiff.bat experiment detector --profile rtlsdr_2400000_cu8 --proposer <p> --classifier <c> --synthetic-dataset wb1 --cabled-dataset cab1

Each model is written with its card; `export onnx` is what ATK loads (CPU,
ONNX Runtime). In ATK: **RF → AI Detect**, tick *AI detection on*, and the
*Learned* proposer box; the card's numbers are on the page. **Exit:**
precision/recall on cabled captures, false alarms per hour on empty
captures, unknown rejection on the held-out class, CPU latency per tile.

**B4 augmentation** is kept only if it closes the gap:
`atkdiff.bat experiment augment --profile rtlsdr_2400000_cu8` (stand-ins
here; `train augmenter` on cabled recordings for the real run).

## Phase 3 — repair (D1, D3, D4, D5)

    atkdiff.bat experiment inpaint     (D1: DSD sync across injected USB dropouts; add a trained inpainter for the learned row)
    atkdiff.bat experiment pulse       (D3: deinterleaving and completion, inferred pulses flagged)
    atkdiff.bat repair audio <a WAV with dropouts>          (D5 audio; the output is INFERRED tier)

Document restoration before OCR (D5) and track inpainting (D4) are library
functions (`repair.document`, `repair.tracks`) with their own tests; ATK's
OCR and air-picture paths are where they will be called from.

## Phase 4 — emitter identification (C)

`atkdiff.bat experiment fingerprint` runs on simulated same-model radios.
The real one: two of your handhelds of the same model, cabled, 50+ key-ups
each, recorded per profile. **Read the result honestly:** on the simulated
radios the classical features told the pair apart at every SNR tested but
rejected a stranger 0 % of the time — the open-set threshold is the hard
part, and it is what the real run must measure.

## Phase 5 — posteriors and maps (E)

    atkdiff.bat experiment e1-coverage      (do the 90 % regions hold the truth 90 % of the time?)
    atkdiff.bat experiment e3-whereami --drive <drive CSV> --calib <route> --test <held-out route>
    atkdiff.bat experiment e4-aperture --freq-hz <a broadcaster's frequency>      (Kraken, driven loop, tower position public)
    atkdiff.bat experiment e5-reach         (synthetic terrain: the three layers, physics / measured / learned)

**E5 on your terrain — no training needed.** Put DTED level 1 tiles (the
usual `w078\n38.dt1` tree) in `rf_data\shared\dted`, then for a radio:

    atkdiff.bat products reach --lat 38.69 --lon -77.55 --freq 151.82e6 --power-w 5 --height 2 --radius-km 25 --label "handheld at the house"

`--model itm` (Longley-Rice, the default) or `deygout` / `bullington` /
`two_ray` / `fspl`; `--antenna directional --gain-dbi 9 --beamwidth 60
--azimuth 270` for a beam; `--antenna-file` for a pattern from ATK's antenna
designer. The map appears on the Geospatial **Layers** tab, INFERRED tier: a
prediction for planning, never a promise of contact. The real *first
experiment* is the same command for a broadcaster from its published
parameters, then a drive measuring it.

Products land in `rf_data\products\` (GeoTIFF + GeoJSON) and appear on the
Geospatial map's **Layers** tab, coloured by how they were made.

## Phase 6 — text (F)

    atkdiff.bat experiment novelty --data <your project as JSON> --llama-model <a GGUF in D:\Analyst_Toolkit\Models>

The autoregressive backend runs today with your own GGUF models; the
diffusion backend (`text.loop`, shared with Palimpsest) waits for llama.cpp
PR #24427 or runs through Transformers. (ATK itself can now run DiffusionGemma
whole-reply through the PR's server — `get_diffusion_gemma.bat` in the ATK
folder — but the host-side loop needs per-step logits, which that server does
not give.)

## Interleaved tracks

- **B7 the signal cut** — on the waterfall, Shift+drag a box (or tick
  ▭ Box) → right-click → **Cut signal…**. The cut opens on **RF → Cuts**:
  the SCF, the cyclic profile with its peaks, the measurements, *Clean*
  (matched / FRESH / FRESH separate / SCORE on Kraken cuts / Wiener / RFI
  mask / diffusion when a denoiser is loaded) with the measured SNR before
  and after, and *Route* to every tool that takes the class.
- **B5 point-and-ask and teach** — *Ask* and *Teach* on the Cuts tab.
  Teach is a prototype, not a retrain: seconds, no GPU.
- **B6 the hunter** — `atkdiff.bat experiment hunter` (simulated band; here
  it found 25 % of bursts against 4 % for a fixed scan). Live: **RF → Hunt**,
  type the goal in words, *Dry run* first, then the box — it needs AI
  detection on and a local radio streaming, and it never transmits or
  changes your gain.
- **C3 the RF social graph** — after Phase 4; Network Link → *Import RF
  social graph…*.
- **I1/I2 vital signs and presence** — when the ESP32 pair arrives:
  **RF → CSI Sensing**. Practise now with
  `atkdiff.bat experiment vitals` and `atkdiff.bat vitals replay <log>`.
  A research-grade measurement, never a diagnosis.
- **J HF propagation now** — WSPR spot files from your chosen Kiwis:
  `atkdiff.bat experiment hf --spots <files> --here FM18 --kiwi NAME:GRID …`
  (voacapl optional for the predicted side).
- **E5 predicted reach** — needs no training: `products reach` above.
