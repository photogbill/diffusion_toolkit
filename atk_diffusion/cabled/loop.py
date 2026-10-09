# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The cabled calibration loop: plan a run, run it (only on the cable), and
turn the recording into perfectly labelled data (plan §3.5, §4.A6; D5, D10).

    TorchSig/native signals at the transmitter's rate (cabled.txfiles)
      -> the bladeRF or HackRF plays them through a cable, attenuators and a
         DC block into the receiver under test (this module, under
         cabled.safety)
      -> ATK's recorder records at the receiver's profile rate
      -> align_labels: the marker is found by cross-correlation and every
         signal in the manifest becomes a SigMF annotation (atk:source
         "cabled") in the recording, with every loop number in its metadata,
         under rf.cabled(profile).

ATK's RF work is passive and receive-only; this loop is the ONE approved
exception (Bill: *"we'd have to use either the bladeRF or HackRF as the
transmitter … careful about transmit power to avoid blowing a receiver"*),
and it must be impossible to run over the air by mistake. So:

* `plan_run` builds the EXACT argument list (hackrf_transfer or bladeRF-cli)
  and carries the safety verdict; a plan whose verdict refuses is not
  runnable, and says why in sentences.
* `execute` REFUSES unless `confirm_cabled=True` is passed at the moment of
  running, re-checks the verdict against the ramp at that moment (a plan
  goes stale), DEFAULTS TO A DRY RUN, and needs the host to pass the
  `runner` explicitly — the toolkit never starts a transmitter on its own.
  The HackRF's 14 dB TX amplifier is always off (`-a 0`); the transmit file
  is played once (no repeat).
* `measure_rx` reads the receiver's own level (dBFS, over the marker) and
  clipping (`dsp.iq.clipped_fraction`) from the recording — the numbers the
  ramp needs before the gain may go up.

THE ALIGNMENT. The chirp marker is an up-chirp then a down-chirp: a
frequency offset δ between the radios moves the up-chirp's correlation peak
by −δ·T/B and the down-chirp's by +δ·T/B, so the mean of the two is the true
start and their difference measures δ. The end marker measures the clock
drift. A marker that is not found (correlation score under the threshold)
writes NO labels — a label in the wrong place is worse than none.

COMMANDS, as their tools document them: `hackrf_transfer -t <file> -f <Hz>
-s <S/s> -x <TX VGA dB> -a 0 [-d <serial>]` (cs8 file); `bladeRF-cli
[-d *:serial=<s>] -e "set frequency tx <Hz>; set samplerate tx <S/s>; set
bandwidth tx <Hz>; set gain tx <dB>; tx config file=<file> format=bin
repeat=1 delay=0; tx start; tx wait"` (SC16 Q11 file).

LIMITS. The measured level is relative (dBFS) unless the receiver's
full-scale input at its gain is known; the ramp says which check it could
make. Per-signal SNR is measured classically against the pre-roll noise, so
the recording should start before the transmitter does.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion import sigmf as _sigmf
from atk_diffusion.cabled import safety as _safety
from atk_diffusion.cabled import txfiles as _tx
from atk_diffusion.dsp import iq as _iq

#: A correlation peak must stand this far above the RMS of the correlation
#: (≈18 dB) to count as the marker.
MIN_SCORE = 8.0
#: The recording should start this long before the transmitter (noise floor).
PREROLL_S = 1.0
#: The start marker is searched for in this many seconds from the start.
SEARCH_S = 30.0

TOOLS = {"hackrf": "hackrf_transfer", "bladerf1": "bladeRF-cli",
         "bladerf2": "bladeRF-cli"}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def hackrf_args(tx_file, frequency_hz: float, rate: float, gain: float,
                serial: str = "", binary: str = "hackrf_transfer") -> list[str]:
    """hackrf_transfer, transmit once, TX amplifier OFF."""
    args = [binary, "-t", str(tx_file), "-f", str(int(round(frequency_hz))),
            "-s", str(int(round(rate))), "-x", str(int(round(gain))), "-a", "0"]
    if serial:
        args += ["-d", str(serial)]
    return args


def bladerf_args(tx_file, frequency_hz: float, rate: float, gain: float,
                 serial: str = "", bandwidth_hz: float | None = None,
                 binary: str = "bladeRF-cli") -> list[str]:
    """bladeRF-cli, one -e script (as ATK's own RX command is built)."""
    bw = int(round(bandwidth_hz if bandwidth_hz else 0.8 * rate))
    parts = [f"set frequency tx {int(round(frequency_hz))}",
             f"set samplerate tx {int(round(rate))}",
             f"set bandwidth tx {bw}",
             f"set gain tx {int(round(gain))}",
             f'tx config file="{tx_file}" format=bin repeat=1 delay=0',
             "tx start", "tx wait"]
    dev = ["-d", f"*:serial={serial}"] if serial else []
    return [binary] + dev + ["-e", "; ".join(parts)]


@dataclass
class RunPlan:
    setup: _safety.LoopSetup
    receiver: _profiles.ReceiverProfile
    verdict: _safety.Verdict
    manifest: dict
    tool: str
    args: list
    rx_seconds: float
    refusals: list = field(default_factory=list)
    created: str = ""

    @property
    def runnable(self) -> bool:
        return self.verdict.ok and not self.refusals

    def steps(self) -> list[str]:
        s = self.setup
        return [f"1. Connect the {s.transmitter} by cable, through "
                f"{s.attenuation_db} dB of attenuation and a DC block, to "
                f"{_profiles.describe(s.receiver_profile)} — no antenna.",
                f"2. Start ATK's recorder on {s.receiver_profile}, tuned to "
                f"{s.frequency_hz:g} Hz, for {self.rx_seconds:.1f} s.",
                f"3. After {PREROLL_S:g} s, transmit: {' '.join(self.args)}",
                "4. When the recorder stops: loop.measure_rx, then "
                "loop.align_labels on the recording."]

    def lines(self) -> list[str]:
        out = list(self.verdict.lines())
        out += [f"REFUSED: {r}" for r in self.refusals]
        out += self.steps()
        out.append("Runnable on the cable." if self.runnable else
                   "Not runnable until every refusal is cleared.")
        return out

    def to_json(self) -> dict:
        return {"setup": self.setup.to_json(), "receiver": self.receiver.id,
                "verdict": self.verdict.to_json(), "tool": self.tool,
                "args": list(self.args), "rx_seconds": self.rx_seconds,
                "refusals": list(self.refusals), "runnable": self.runnable,
                "steps": self.steps(), "created": self.created,
                "tx_file": self.manifest.get("file"),
                "tx_sha256": self.manifest.get("sha256")}


def plan_run(setup: _safety.LoopSetup, receiver: _profiles.ReceiverProfile,
             tx_manifest, *, ramp: _safety.Ramp | None = None,
             binary: str | None = None, bandwidth_hz: float | None = None
             ) -> RunPlan:
    """The verdict and the exact command for one run. Nothing is run."""
    m = _tx.load_manifest(tx_manifest)
    verdict = _safety.check(setup, receiver, ramp)
    refusals = []
    fam = str(setup.transmitter).lower()
    if m.get("transmitter") != fam:
        refusals.append(f"the transmit file was made for the "
                        f"{m.get('transmitter')}, not the {fam}; its format "
                        "would be wrong.")
    if "_dir" in m:
        ok, why = _tx.verify_file(m)
        if not ok:
            refusals.append(why + ".")
    tool = TOOLS.get(fam, "")
    path = str(Path(m.get("_dir", ".")) / m.get("file", ""))
    rate = float(m.get("tx_rate", 0.0))
    if fam == "hackrf":
        args = hackrf_args(path, setup.frequency_hz, rate, setup.tx_gain,
                           setup.tx_serial, binary or tool)
    elif fam in ("bladerf1", "bladerf2"):
        args = bladerf_args(path, setup.frequency_hz, rate, setup.tx_gain,
                            setup.tx_serial, bandwidth_hz, binary or tool)
    else:
        args = []
    rx_s = PREROLL_S + float(m.get("duration_s", 0.0)) + 1.0
    return RunPlan(setup, receiver, verdict, m, tool, args, rx_s, refusals,
                   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def subprocess_runner(args: list, timeout_s: float | None = None
                      ) -> tuple[int, str]:
    """A runner ATK may pass to `execute`: the tool as a child process, its
    console text kept (the tail)."""
    exe = shutil.which(args[0]) or args[0]
    p = subprocess.run([exe] + list(args[1:]), capture_output=True, text=True,
                       timeout=timeout_s)
    return p.returncode, ((p.stdout or "") + (p.stderr or ""))[-4000:]


def execute(plan: RunPlan, *, confirm_cabled: bool = False,
            runner: Callable[[list], tuple] | None = None, dry_run: bool = True,
            ramp: _safety.Ramp | None = None) -> dict:
    """Run a plan — only on the cable, only when asked, never by default.
    -> {ran, dry_run, args, why, returncode, output, verdict}."""
    out = {"ran": False, "dry_run": bool(dry_run), "args": list(plan.args),
           "why": "", "returncode": None, "output": ""}
    if confirm_cabled is not True:
        out["why"] = ("Refused: confirm, now, that the transmitter is connected "
                      "by cable through the attenuators and a DC block to the "
                      "receiver — never to an antenna (confirm_cabled=True). "
                      "Nothing was transmitted.")
        return out
    verdict = _safety.check(plan.setup, plan.receiver, ramp) if ramp is not None \
        else plan.verdict
    out["verdict"] = verdict.to_json()
    if not verdict.ok or plan.refusals:
        out["why"] = "Refused: " + " ".join(list(verdict.refusals)
                                            + list(plan.refusals))
        return out
    if not plan.args:
        out["why"] = "Refused: there is no command for this transmitter."
        return out
    if dry_run:
        out["why"] = ("Dry run — nothing was transmitted. This is the command "
                      "that would run: " + " ".join(plan.args))
        return out
    if runner is None:
        out["why"] = ("Refused: no runner was given. The toolkit never starts a "
                      "transmitter on its own; ATK passes the runner.")
        return out
    if ramp is not None:
        ramp.begin(plan.setup, verdict.expected_input_dbm)
    try:
        rc, text = runner(list(plan.args))
    except Exception as exc:                               # noqa: BLE001
        out["why"] = f"the transmitter command failed to start: {exc}"
        return out
    out.update(ran=True, returncode=rc, output=str(text or "")[-4000:],
               why=("transmitted on the cable" if rc == 0 else
                    f"the transmitter command ended with code {rc}"))
    return out


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def _xcorr(x: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """c[l] = sum_t x[l+t] conj(tmpl[t]) / ||tmpl||, l in valid range."""
    from scipy.signal import fftconvolve
    if x.size < tmpl.size:
        return np.zeros(0, dtype=np.complex64)
    c = fftconvolve(x, np.conj(tmpl[::-1]), mode="valid")
    return (c / max(float(np.linalg.norm(tmpl)), 1e-12)).astype(np.complex64)


def _first_peak(mag: np.ndarray, width: int, lo: int = 0,
                hi: int | None = None, stat: str = "ratio"
                ) -> tuple[float, float]:
    """The EARLIEST strong peak of |c| in [lo, hi): the first sample at half
    the window's maximum or more, then the local maximum within `width` of it
    (the start marker comes before the identical end marker). -> (sub-sample
    index, score).

    The score is measured against ROBUST noise statistics, so the peaks
    themselves do not move the yardstick: `ratio` = peak / noise RMS, where
    the RMS of a coherent correlation's Rayleigh-distributed noise is 1.2011
    x its median; `z` = (peak − median) / (1.4826 x MAD), for a non-coherent
    sum, whose noise has a large mean. Noise alone scores about 5 on either
    over a million lags; a marker scores in the tens."""
    n = mag.size
    hi = n if hi is None else min(n, int(hi))
    lo = max(0, int(lo))
    if hi - lo <= 0:
        return 0.0, 0.0
    win = mag[lo:hi].astype(np.float64)
    top = float(win.max())
    if top <= 0:
        return float(lo), 0.0
    i0 = int(np.flatnonzero(win >= 0.5 * top)[0])
    j1 = min(win.size, i0 + width + 1)
    i = i0 + int(np.argmax(win[i0:j1]))
    pk = float(win[i])
    frac = 0.0
    if 0 < i < win.size - 1:
        a, b, c = float(win[i - 1]), pk, float(win[i + 1])
        den = a - 2 * b + c
        if den != 0:
            frac = 0.5 * (a - c) / den
    ref = mag if win.size < 64 else win
    med = float(np.median(ref))
    if stat == "z":
        mad = float(np.median(np.abs(ref - med))) * 1.4826
        score = (pk - med) / mad if mad > 0 else 0.0
    else:
        rms = 1.2011 * med
        score = pk / rms if rms > 0 else 0.0
    return lo + i + max(-0.5, min(0.5, frac)), score


def _residual_cfo(seg: np.ndarray, tmpl: np.ndarray, fs: float) -> float | None:
    """The frequency of x·conj(template) over the marker: a tone at the
    remaining offset, unambiguous to ±fs/2 (an FFT peak, interpolated)."""
    z = seg[:tmpl.size] * np.conj(tmpl[:seg.size])
    if z.size < 16:
        return None
    nfft = 1 << int(math.ceil(math.log2(8 * z.size)))
    Z = np.abs(np.fft.fft(z, nfft))
    k = int(np.argmax(Z))
    a, b, c = Z[(k - 1) % nfft], Z[k], Z[(k + 1) % nfft]
    den = a - 2 * b + c
    frac = 0.5 * (a - c) / den if den != 0 else 0.0
    f = (k + frac) * fs / nfft
    return float(f - fs if f > fs / 2 else f)


def find_marker(x: np.ndarray, fs: float, marker: dict, f_shift_hz: float,
                lo: int = 0, hi: int | None = None) -> dict:
    """Find the (first) marker in x[lo:hi]. -> {found, start (float sample),
    score, cfo_hz}."""
    kind = marker["kind"]
    params = marker["params"]
    hi = x.size if hi is None else min(x.size, int(hi))
    lo = max(0, int(lo))
    seg = x[lo:hi]
    none = {"found": False, "start": None, "score": 0.0, "cfo_hz": None}
    if kind == "chirp":
        up, dn = _tx.chirp_parts(fs, params, f_shift_hz)
        T = float(params["duration_s"])
        B = float(params["bandwidth_hz"])
        n = up.size
        cu = np.abs(_xcorr(seg, up))
        cd = np.abs(_xcorr(seg, dn))
        if cu.size == 0:
            return none
        w = max(4, n // 4)
        pu, su = _first_peak(cu, w)
        # the down-chirp follows the up-chirp by one chirp length (± offset)
        pd, sd = _first_peak(cd, w, int(pu) + n - w, int(pu) + n + w + 1)
        start = 0.5 * (pu + pd - n)
        cfo = (pd - pu - n) * B / (2.0 * T * fs)
        score = min(su, sd)
        found = score >= MIN_SCORE
        return {"found": bool(found), "start": lo + start, "score": float(score),
                "cfo_hz": float(cfo) if found else None}
    if kind == "pn":
        tmpl = _tx.pn_waveform(fs, params, f_shift_hz)
        k = max(1, int(params.get("segments", 8)))
        L = tmpl.size
        bounds = np.linspace(0, L, k + 1).astype(int)
        nvalid = seg.size - L + 1
        if nvalid <= 0:
            return none
        acc = np.zeros(nvalid, dtype=np.float64)
        parts = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            c = _xcorr(seg, tmpl[a:b])[a:a + nvalid]
            parts.append((0.5 * (a + b) / fs, c))
            acc += np.abs(c)
        p, score = _first_peak(acc, max(4, L // (4 * k)), stat="z")
        i = int(round(p))
        found = score >= MIN_SCORE
        cfo = _residual_cfo(seg[i:i + L], tmpl, fs) if found else None
        return {"found": bool(found), "start": lo + p, "score": float(score),
                "cfo_hz": cfo if found else None}
    raise ValueError(f"unknown marker kind {kind!r}")


def _band_power(x: np.ndarray, fs: float, f_lo: float, f_hi: float,
                nfft: int = 1024) -> float | None:
    """Mean power in [f_lo, f_hi] (baseband Hz) by averaged periodograms."""
    if x.size < nfft:
        nfft = 1 << max(4, int(math.log2(max(16, x.size))))
        if x.size < nfft:
            return None
    nseg = x.size // nfft
    segs = x[:nseg * nfft].reshape(nseg, nfft) * np.hanning(nfft)
    P = np.mean(np.abs(np.fft.fft(segs, axis=1)) ** 2, axis=0)
    f = np.fft.fftfreq(nfft, 1.0 / fs)
    sel = (f >= f_lo) & (f <= f_hi)
    if not np.any(sel):
        return None
    return float(np.sum(P[sel]))


@dataclass
class AlignResult:
    found: bool
    why: str = ""
    score: float = 0.0
    start_sample: int | None = None
    cfo_hz: float | None = None
    drift_ppm: float | None = None
    end_found: bool = False
    labels: int = 0
    capture: str = ""
    annotations: list = field(default_factory=list)
    level: dict = field(default_factory=dict)

    def lines(self) -> list[str]:
        if not self.found:
            return [f"No labels were written: {self.why}"]
        out = [f"Marker found at sample {self.start_sample} (score "
               f"{self.score:.1f}); frequency offset "
               f"{0.0 if self.cfo_hz is None else self.cfo_hz:+.0f} Hz"
               + (f"; clock drift {self.drift_ppm:+.1f} ppm"
                  if self.drift_ppm is not None else "; no end marker found")
               + ".",
               f"{self.labels} cabled labels written to {self.capture}."]
        if self.why:
            out.append(self.why)
        return out


def _locate(rx_capture, manifest: dict, f_tx_hz: float | None,
            search_s: float | None):
    meta = _sigmf.read_meta(rx_capture)
    fs = _sigmf.sample_rate_of(meta)
    if _sigmf.channels_of(meta) > 1:
        raise ValueError("align one channel at a time (a Kraken capture: one "
                         "file per channel)")
    x = _sigmf.load(rx_capture, meta=meta)
    f_rx = _sigmf.center_of(meta)
    f_tx = f_rx if f_tx_hz is None else float(f_tx_hz)
    shift = f_tx - f_rx
    mk = manifest["marker"]
    bw = float(mk["params"].get("bandwidth_hz",
                                mk["params"].get("chip_rate_hz", 0.0)))
    if abs(shift) + 0.5 * bw > 0.5 * fs:
        raise ValueError(f"the transmitter's centre is {shift:+.0f} Hz from the "
                         f"receiver's and the marker is {bw:g} Hz wide — it "
                         f"does not fit in the recording's {fs:g} S/s")
    # the start marker follows the pre-roll: look in the first SEARCH_S
    # seconds unless told otherwise (a long recording is not correlated end
    # to end for a marker that is near its start)
    s_s = SEARCH_S if search_s is None else float(search_s)
    n_search = min(x.size, int(s_s * fs))
    st = find_marker(x, fs, mk, shift, 0, n_search)
    return meta, fs, x, f_rx, f_tx, shift, st


def measure_rx(rx_capture, tx_manifest, *, f_tx_hz: float | None = None,
               search_s: float | None = None) -> dict:
    """The receiver's own reading for the ramp: level over the marker (dBFS),
    the TX-off noise (the pre-roll), and the clipped fraction of the whole
    recording at the converter's rails."""
    m = _tx.load_manifest(tx_manifest)
    meta, fs, x, _f_rx, _f_tx, _shift, st = _locate(rx_capture, m, f_tx_hz,
                                                     search_s)
    clipped = _iq.clipped_fraction(x, _sigmf.datatype_of(meta))
    out = {"clipped_fraction": float(clipped), "level_dbfs": None,
           "noise_dbfs": None, "above_noise_db": None, "found": st["found"],
           "words": ""}
    if not st["found"]:
        out["words"] = (f"the marker was not found (score {st['score']:.1f}, "
                        f"needs {MIN_SCORE:g}); only the clipping could be read")
        return out
    s0 = int(round(st["start"]))
    n_mk = int(round(m["marker"]["length"] * fs / float(m["tx_rate"])))
    out["level_dbfs"] = float(_iq.power_dbfs(x[s0:s0 + n_mk]))
    guard = int(0.01 * fs)
    if s0 - guard >= 1000:
        out["noise_dbfs"] = float(_iq.power_dbfs(x[:s0 - guard]))
        out["above_noise_db"] = out["level_dbfs"] - out["noise_dbfs"]
        out["words"] = (f"marker level {out['level_dbfs']:.1f} dBFS, "
                        f"{out['above_noise_db']:.1f} dB above the TX-off noise; "
                        f"{100 * clipped:.3g}% of samples at the rails")
    else:
        out["words"] = ("marker level read, but the recording has no pre-roll "
                        "before the transmitter, so there is no noise reading")
    return out


def align_labels(rx_capture, tx_manifest, *, rf=None,
                 setup: _safety.LoopSetup | None = None,
                 verdict: _safety.Verdict | None = None,
                 f_tx_hz: float | None = None, search_s: float | None = None,
                 write: bool = True) -> AlignResult:
    """Find the marker, place every manifest signal in the recording, and
    write them as SigMF annotations (atk:source "cabled") with the loop's
    numbers in the global metadata — in a capture under rf.cabled(profile)
    (copied there when it is elsewhere; the original is not touched)."""
    m = _tx.load_manifest(tx_manifest)
    meta0 = _sigmf.read_meta(rx_capture)
    rx_profile = _profiles.profile_from_meta(meta0)
    if setup is not None:
        _profiles.check_match(setup.receiver_profile, rx_profile,
                              what="this cabled run")
        if f_tx_hz is None:
            f_tx_hz = setup.frequency_hz
    if f_tx_hz is None and m.get("tx_center_hz") is not None:
        f_tx_hz = float(m["tx_center_hz"])
    meta, fs, x, f_rx, f_tx, shift, st = _locate(rx_capture, m, f_tx_hz, search_s)
    res = AlignResult(found=False, score=float(st["score"]))
    if not st["found"]:
        res.why = (f"the alignment marker was not found (best score "
                   f"{st['score']:.1f} against {MIN_SCORE:g}). A label in the "
                   "wrong place is worse than none.")
        return res
    s0 = float(st["start"])
    cfo = float(st["cfo_hz"] or 0.0)
    ratio = fs / float(m["tx_rate"])
    drift = None
    scale = 1.0
    em = m.get("end_marker")
    if em:
        exp = s0 + float(em["start_s"]) * fs
        win = int(0.002 * float(em["start_s"]) * fs) + 4 * int(m["marker"]["length"] * ratio)
        se = find_marker(x, fs, m["marker"], shift, int(exp) - win,
                         int(exp) + win + int(m["marker"]["length"] * ratio) + 1)
        if se["found"]:
            span = float(se["start"]) - s0
            scale = span / (float(em["start_s"]) * fs)
            drift = (scale - 1.0) * 1e6
            res.end_found = True
    res.found = True
    res.start_sample = int(round(s0))
    res.cfo_hz = cfo
    res.drift_ppm = drift
    # noise for per-signal SNR: the pre-roll before the transmitter started
    guard = int(0.01 * fs)
    noise = x[:max(0, int(s0) - guard)]
    from atk_diffusion.detect import classes as _classes
    anns = []
    spans = []
    for sig in m["signals"]:
        on = sig.get("bursts_s") or [[sig["start_s"], sig["duration_s"]]]
        spans += [(sig, float(b0), float(bd)) for b0, bd in on]
    for sig, b0, bd in spans:
        a0 = int(round(s0 + b0 * fs * scale))
        cnt = int(round(bd * fs * scale))
        if a0 >= x.size or cnt <= 0:
            continue
        cnt = min(cnt, x.size - a0)
        f_c_rel = shift + float(sig["f_offset_hz"]) + cfo     # in the recording
        bw = float(sig["bandwidth_hz"] or 0.0)
        extra = {"atk:source": "cabled",
                 "atk:family": sig.get("family") or "unknown",
                 "atk:carrier_offset_hz": f_c_rel,
                 "atk:generator": sig.get("generator", "")}
        c = _classes.get(sig.get("class", ""))
        if c is not None:
            extra["atk:family"] = c.family
        if sig.get("symbol_rate_hz"):
            extra["atk:symbol_rate"] = float(sig["symbol_rate_hz"])
        half = max(0.5 * bw, 0.5 * fs / 1024)
        if noise.size >= 4096 and bw > 0:
            ps = _band_power(x[a0:a0 + cnt], fs, f_c_rel - half, f_c_rel + half)
            pn = _band_power(noise, fs, f_c_rel - half, f_c_rel + half)
            if ps is not None and pn and pn > 0:
                extra["atk:snr_db"] = float(10 * math.log10(max(ps - pn, 1e-12 * pn) / pn))
        anns.append(_sigmf.Annotation(
            a0, cnt, f_rx + f_c_rel - 0.5 * bw, f_rx + f_c_rel + 0.5 * bw,
            label=str(sig.get("class") or sig.get("label")),
            comment=f"cabled loop: {sig.get('label', '')}", extra=extra))
    res.annotations = [a.to_sigmf() for a in anns]
    res.labels = len(anns)
    if not write:
        res.capture = str(_sigmf.base_of(rx_capture))
        return res
    # -- write into a capture under rf.cabled(profile) -------------------------
    base = _sigmf.base_of(rx_capture)
    if rf is None:
        res.why = ("no rf_data root was given, so the labels were written into "
                   "the recording where it is, not under rf.cabled(profile)")
    else:
        dest_dir = Path(rf.cabled(rx_profile))
        try:
            inside = Path(base).resolve().parent == dest_dir.resolve()
        except OSError:
            inside = False
        if not inside:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(base).name
            k = 1
            while _sigmf.meta_path(dest).exists():
                dest = dest_dir / f"{Path(base).name}_{k}"
                k += 1
            shutil.copyfile(_sigmf.data_path(base), _sigmf.data_path(dest))
            shutil.copyfile(_sigmf.meta_path(base), _sigmf.meta_path(dest))
            base = dest
    _sigmf.add_annotations(base, anns, replace_source="cabled")
    meta = _sigmf.read_meta(base)
    g = meta.setdefault("global", {})
    g.setdefault("atk:tier", "record")
    g.setdefault("atk:receiver_profile", rx_profile)
    if verdict is None and setup is not None:
        receiver = None
        if rf is not None:
            receiver = _profiles.load_profile(rf, setup.receiver_profile)
        if receiver is not None:
            verdict = _safety.check(setup, receiver)
    if verdict is not None:
        for k, v in verdict.metadata().items():
            g[k] = v
    elif setup is not None:
        tp = setup.tx_power.at(setup.tx_gain) if setup.tx_power else None
        g.update({"atk:tx_power_dbm": tp,
                  "atk:attenuation_db": setup.attenuation_db,
                  "atk:splitter_loss_db": setup.splitter_loss_db or 0.0,
                  "atk:cable_loss_db": setup.cable_loss_db,
                  "atk:expected_input_dbm":
                      None if tp is None or setup.attenuation_db is None else
                      _safety.expected_input_dbm(tp, setup.attenuation_db,
                                                 setup.splitter_loss_db or 0.0,
                                                 setup.cable_loss_db)})
    g["atk:tx_file"] = m.get("file")
    g["atk:tx_file_sha256"] = m.get("sha256")
    g["atk:alignment"] = {"marker": m["marker"]["kind"], "score": res.score,
                          "start_sample": res.start_sample, "cfo_hz": res.cfo_hz,
                          "drift_ppm": res.drift_ppm, "end_marker_found": res.end_found,
                          "tier": "measured"}
    _sigmf.write_meta(base, meta)
    if rf is not None:
        try:
            if not Path(_sigmf.data_path(rx_capture)).resolve() == \
                    Path(_sigmf.data_path(base)).resolve():
                rf.record(_sigmf.data_path(base), "cabled",
                          f"copied from {Path(_sigmf.base_of(rx_capture)).name}")
            rf.record(_sigmf.meta_path(base), "cabled-meta",
                      f"{res.labels} cabled labels")
        except Exception:                                  # noqa: BLE001
            pass
    res.capture = str(base)
    return res
