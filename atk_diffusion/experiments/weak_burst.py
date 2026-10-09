# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""B3's first experiment — the weak-burst test (plan §4.B3, §6 Phase 1, §7).

Plan B3, word for word: *"inject a weak burst into real noise from each
profile — bursts Bill can actually make or capture: a bladeRF-generated
burst of a waveform the denoiser was not trained on, a handheld's PTT
key-up, a pager burst, an ADS-B squitter, a LoRa chirp — and ask at what SNR
the diffusion denoiser recovers it where a Wiener filter, a median filter
and a wavelet denoiser miss it, with the waveform unknown to all four. The
matched filter is optimal when the waveform is known; the analyst's case is
that it is not."* Phase 1's exit: *"SNR numbers against three classical
denoisers."* Nothing here needs a CSAR survival radio (plan change log,
2026-10-08: Bill has not had access to one in twenty years).

WHAT `run` DOES, per burst kind:

1. NOISE — a SigMF capture of THIS profile (terminated, or a quiet band;
   checked against the profile, refused in words otherwise), or synthetic
   noise through the profile's measured impairments (dsp.impair; a textbook
   receiver when unmeasured — the result says which). The floor per bin is
   measured from reference windows of the same noise.
2. BURSTS — `synth.native` when installed (real sync words and framing),
   else the local minimal generators (`learn.augment`). Each is placed at a
   random time and frequency inside the analysis tile and scaled to an
   IN-BAND SNR against the measured floor in its own bins (what "dB above
   the floor" means on a waterfall). The full-band SNR and E/N0 are reported
   beside it.
3. THE SAME FRONT END AND THE SAME DETECTOR for every method: the profile's
   STFT and max-pooling (the pipeline's tiles), dB above the floor; then
   raw (no denoiser), Wiener, median, wavelet, and the diffusion denoiser
   (the card's model, at its SNR-matched step); then `dsp.cfar.
   energy_proposer` (another engineer's module, the pipeline's own) when it
   is installed, else the internal CA-CFAR plus floor test.
4. A FIXED FALSE-ALARM RATE — the fair comparison. Every method changes the
   noise statistics, so a CFAR's nominal threshold means something
   different after each. The detector is swept over a grid of cell
   false-alarm rates; for each method the operating point is the loosest
   one whose TILE false-alarm rate on noise-only trials is at most `pfa`
   (the realised rate is reported). Pd at that point, per SNR.
5. THE NUMBERS — Pd vs SNR per method, the SNR where each reaches Pd 0.5
   and 0.9, and the HALLUCINATION RATE: of the noise-only tiles in which the
   raw tile had no detection at the profile's nominal false-alarm rate, the
   fraction in which a method's output has one. Classical methods should
   read 0; the diffusion denoiser's number is the one plan §2.1 is about.
6. THE MATCHED FILTER — the known-waveform reference line: the exact burst
   as template, correlated at every lag of the IQ window, its threshold
   calibrated on noise the same way. It knows the waveform; the four
   denoisers do not.

Results go to `write_report` (experiments.report, another engineer's
module; a minimal local writer when it is absent) under the profile's
`runs\\` folder or `out_dir`.

LIMITS. On synthetic noise this measures the method, not Bill's receiver;
the real number comes from his captures. The analysis tile is a crop of the
pipeline's tile (`rows` x `bins`, widened for bursts wider than it), so the
tile false-alarm rate is per crop, not per second. A burst longer than the
tile is clipped to it (reported). A denoiser trained on another geometry or
profile is refused, not adapted.
"""

from __future__ import annotations

import json
import math
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atk_diffusion import profiles, provenance
from atk_diffusion.dsp import denoise_classical as _cl

#: Bill's list, in the class names synth.native knows where it has them.
#: `mfsk8` is the waveform the B3 denoiser is NOT trained on.
DEFAULT_BURSTS = ("mfsk8", "nfm_keyup", "pocsag", "adsb", "lora")
BURST_WORDS = {
    "mfsk8": "a bladeRF-made waveform the denoiser was not trained on (8-FSK)",
    "nfm_keyup": "a handheld's PTT key-up (NFM, CTCSS)",
    "pocsag": "a pager burst (POCSAG)",
    "adsb": "an ADS-B squitter",
    "lora": "a LoRa chirp packet",
}
CLASSICAL = ("wiener", "median", "wavelet")
SNR_DEFINITION = ("in-band SNR: the burst's power while on over the measured "
                  "floor in the bins it occupies (its 99 % bandwidth) — the "
                  "'dB above the floor' an analyst reads on the waterfall")


# ---------------------------------------------------------------------------
# Detectors with a swept false-alarm rate
# ---------------------------------------------------------------------------
PFA_GRID = tuple(float(v) for v in np.geomspace(0.3, 1e-8, 31))


class _Internal:
    """CA-CFAR along frequency OR the floor-referenced test (the pipeline's
    pair, internal fallback) — monotone in the cell false-alarm rate."""
    name = "internal CA-CFAR + floor test (dsp.cfar not installed)"

    def __init__(self, geom: dict, guard: int = 2, train: int = 16,
                 min_cells: int = 3, margin_db: float = 1.0):
        self.g, self.guard, self.train = geom, guard, train
        self.min_cells, self.margin = min_cells, 10 ** (margin_db / 10.0)
        self._cache: dict = {}

    def _thr(self, pfa):
        if pfa not in self._cache:
            k, mode = self.g["pool"], self.g["pool_mode"]
            a = _cl.cfar_alpha(pfa, 2 * self.train, k, mode)
            if mode == "max":
                tau = -math.log1p(-(1.0 - pfa) ** (1.0 / k))
            else:
                from scipy.stats import gamma
                tau = float(gamma.isf(pfa, k, scale=1.0 / k))
            self._cache[pfa] = (a, tau * self.margin)
        return self._cache[pfa]

    def prepare(self, spec_db):
        P = np.power(10.0, np.asarray(spec_db, dtype=np.float64) / 10.0)
        return (P, _cl.cfar_ratio(P, self.guard, self.train))

    def fires(self, prep, pfa) -> bool:
        P, ratio = prep
        a, tau = self._thr(pfa)
        mask = (ratio > a) | (P > tau)
        if not mask.any():
            return False
        return any(b.cells >= self.min_cells for b in _cl.cfar_boxes(mask))


class _Proposer:
    """dsp.cfar.energy_proposer on a pipeline Tile — the detector the
    pipeline itself runs, with its per-row layers (CA-CFAR and the floor
    test) and WITHOUT its integrated layer.

    WHY NOT THE INTEGRATED LAYER (corrected 2026-10-09). That layer averages
    MEAN-pooled power over blocks of rows (`Tile.mean_above`). Every method
    here — raw and the four denoisers — works on the MAX-pooled dB tile, and
    no denoiser produces a mean-pooled version of its output. The first
    version filled `mean_above` from the max-pooled tile: max-pooled noise
    sits about 3.6 dB above the floor, so every block of pure noise looked
    2.3x the floor, the raw detector fired on 100 % of noise-only tiles
    (realised Pfa 1.000), its Pd column was meaningless and no tile was
    left to judge hallucination on. Same detector for every method, honestly
    described, is the fair comparison."""
    name = ("dsp.cfar.energy_proposer, per-row layers (CA-CFAR + floor test; "
            "the integrated layer needs mean-pooled power the denoisers do "
            "not produce)")

    def __init__(self, geom: dict, fs: float, min_cells: int = 3):
        from atk_diffusion.dsp import cfar as _cfar
        from atk_diffusion.dsp import stft as _stft
        self._cfar, self._stft = _cfar, _stft
        self.g, self.fs, self.min_cells = geom, float(fs), min_cells

    def tile(self, spec_db):
        s = np.asarray(spec_db, dtype=np.float32)
        rows, bins = s.shape
        g = self.g
        rp = g["pool"] * g["hop"] / self.fs
        lay = self._stft.TileLayout(
            fs=self.fs, fft_size=g["fft_size"], hop=g["hop"], window=g["window"],
            rows=rows, pool=g["pool"], frames=rows * g["pool"], overlap_rows=0,
            step_rows=rows, bin_hz=self.fs / g["fft_size"], row_period=rp,
            seconds=rows * rp, requested_seconds=rows * rp)
        bin_hz = self.fs / g["fft_size"]
        return self._stft.Tile(spec=s, abs_db=s.copy(),
                               mean_above=np.power(10.0, s / 10.0).astype(np.float32),
                               t0=0.0, t1=rows * rp, f0=0.0, f1=bins * bin_hz,
                               bin_hz=bin_hz, row_period=rp, center_hz=0.0,
                               fs=self.fs, layout=lay, rows_valid=rows)

    def prepare(self, spec_db):
        return self.tile(spec_db)

    def fires(self, tile, pfa) -> bool:
        return bool(self._cfar.energy_proposer(tile, pfa=pfa, min_cells=self.min_cells,
                                               integrate_rows=0))


def _detector(which: str, geom: dict, fs: float, min_cells: int):
    if which in ("auto", "energy_proposer"):
        try:
            d = _Proposer(geom, fs, min_cells)
            d.fires(d.prepare(np.zeros((8, 32), np.float32)), 1e-3)   # smoke
            return d
        except Exception as e:                             # noqa: BLE001
            if which == "energy_proposer":
                raise RuntimeError(f"dsp.cfar.energy_proposer could not run here: {e}") from None
    return _Internal(geom, min_cells=min_cells)


def _pmin(det, prep, grid) -> float:
    """The smallest cell false-alarm rate on the grid at which the detector
    fires on this tile (inf: not even at the loosest). Bisection: firing is
    monotone in the false-alarm rate."""
    lo, hi = 0, len(grid) - 1           # grid descends: index up = stricter
    if not det.fires(prep, grid[0]):
        return float("inf")
    if det.fires(prep, grid[hi]):
        return grid[hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if det.fires(prep, grid[mid]):
            lo = mid
        else:
            hi = mid
    return grid[lo]


def operating_point(noise_pmin, target: float, grid) -> tuple[float, float]:
    """The loosest cell false-alarm rate whose tile false-alarm rate on the
    noise trials is at most `target`. Returns (cell rate, realised rate)."""
    p = np.asarray(noise_pmin, dtype=np.float64)
    for g in grid:                       # loosest first
        rate = float(np.mean(p <= g)) if p.size else 0.0
        if rate <= target:
            return float(g), rate
    return float(grid[-1]), float(np.mean(p <= grid[-1]))


def snr_at_pd(snrs, pd, level: float):
    """Linear interpolation of the first crossing of `level` (None: never;
    the lowest SNR, flagged, when already above it there)."""
    s = np.asarray(snrs, dtype=np.float64)
    p = np.asarray(pd, dtype=np.float64)
    if p.size == 0:
        return None
    if p[0] >= level:
        return {"snr_db": float(s[0]), "at_or_below": True}
    for i in range(1, p.size):
        if p[i] >= level > p[i - 1]:
            f = (level - p[i - 1]) / (p[i] - p[i - 1])
            return {"snr_db": float(s[i - 1] + f * (s[i] - s[i - 1])), "at_or_below": False}
    return None


# ---------------------------------------------------------------------------
# The matched filter — the known-waveform line
# ---------------------------------------------------------------------------
def matched_statistic(y, template, noise_power: float) -> float:
    """max over lags of |⟨y, template⟩|² / (‖template‖²·P_n): the genie
    detector that knows the exact waveform but not when it arrives."""
    from scipy.signal import fftconvolve
    t = np.asarray(template, dtype=np.complex128)
    e = float(np.sum(np.abs(t) ** 2))
    if e <= 0 or t.size > np.asarray(y).size:
        return 0.0
    c = fftconvolve(np.asarray(y, dtype=np.complex128), np.conj(t[::-1]), mode="valid")
    return float(np.max(np.abs(c) ** 2) / (e * float(noise_power)))


# ---------------------------------------------------------------------------
def _load_denoiser(denoiser, pid: str):
    if denoiser is None:
        return None
    if isinstance(denoiser, (str, Path)):
        from atk_diffusion.dsp import denoise_runtime as _rt
        try:
            return _rt.DenoiserRuntime(denoiser, pid)
        except RuntimeError:
            from atk_diffusion.learn import denoiser as _dn
            return _dn.Denoiser.load(denoiser, for_profile=pid)
    card = getattr(denoiser, "card", None)
    if card is not None and card.profile:
        profiles.check_match(card.profile, pid, what=f"this denoiser ({card.name})")
    return denoiser


def report_dir(rf, profile: str, name: str) -> Path:
    """The run folder under the profile's runs\\ (experiments.report's
    `experiment_dir` when installed: a fresh UTC-stamped folder, never
    reused)."""
    try:
        from atk_diffusion.experiments import report as _report
        return _report.experiment_dir(rf, profile, name)
    except ImportError:
        base = Path(rf.runs(profile))
        d = base / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{name}"
        k = 2
        while d.exists():
            d = base / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{name}_{k}"
            k += 1
        d.mkdir(parents=True)
        return d


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if o is None or isinstance(o, str):
        return o
    return str(o)


def write_report(out_dir, name: str, result: dict, summary: list[str],
                 detail: list[str] | None = None, rf=None):
    """The package's one way to write a result (experiments.report, another
    engineer's module): `<name>.md` with the summary sentences and the
    scalar numbers, `<name>.json` with the whole result under "result",
    tier MEASURED and a provenance stamp — a local writer with the same
    layout when that module is absent. This experiment's own tables go
    beside it as `<name>_detail.md`. Returns (md, json, detail md | None)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from atk_diffusion.experiments import report as _report
        md, js = _report.write_report(out, name, result, summary, rf=rf)
    except ImportError:
        payload = {"name": name, "tier": "measured",
                   "tier_words": provenance.TIER_WORDS["measured"],
                   "provenance": provenance.stamp(f"experiments.{name}"),
                   "summary": [str(s) for s in summary], "result": _clean(result)}
        js = out / f"{name}.json"
        js.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        md = out / f"{name}.md"
        md.write_text("\n".join([f"# {name.replace('_', ' ')}", "",
                                 f"*{provenance.TIER_WORDS['measured']}*", ""]
                                + [f"- {s}" for s in summary]
                                + ["", f"The full result is in `{js.name}`.", ""]),
                      encoding="utf-8")
    det = None
    if detail:
        det = out / f"{name}_detail.md"
        det.write_text("\n".join(detail) + "\n", encoding="utf-8")
        if rf is not None:
            try:
                rf.record(det, "report", name)
            except Exception:                              # noqa: BLE001
                pass
    return md, js, det


@dataclass
class _Trial:
    spec: np.ndarray
    iq: np.ndarray
    template: np.ndarray | None


def run(rf=None, profile: str = "rtlsdr_2400000_cu8", *, noise_capture=None,
        bursts=DEFAULT_BURSTS, snrs_db=(-15.0, -12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0),
        trials: int = 50, noise_trials: int = 200, pfa: float = 0.01,
        cell_pfa: float | None = None, denoiser=None, denoise_steps: int = 1,
        geometry=None, pool_mode: str = "max", rows: int = 64, bins: int = 128,
        detector: str = "auto", min_cells: int = 3, seed: int = 0,
        out_dir=None, name: str = "weak_burst", progress=None) -> dict:
    """The weak-burst test. Returns the result dict (and writes the report
    when `rf` or `out_dir` is given). See the module docstring."""
    from atk_diffusion.learn import augment as _aug
    from atk_diffusion.learn import denoiser as _dn
    say = progress or (lambda s: None)
    t_start = time.time()
    pid = str(profile).lower()
    prof = profiles.load_profile(rf, pid) if rf is not None else profiles.new_profile(pid)
    fs = float(prof.sample_rate)
    rng = np.random.default_rng(seed)
    geom = _dn.tile_geometry(prof, geometry, pool_mode)
    fft, hop, pool = geom["fft_size"], geom["hop"], geom["pool"]
    cell_pfa = float(cell_pfa if cell_pfa is not None else prof.cfar_pfa)
    grid = tuple(sorted(set(PFA_GRID) | {cell_pfa}, reverse=True))
    den = _load_denoiser(denoiser, pid)
    seen_kinds: list[str] = []
    if den is not None:
        inp = den.card.input
        if inp.get("domain") != "spectrogram":
            raise ValueError("the weak-burst test runs on spectrogram tiles; this "
                             f"denoiser ({den.card.name}) works on IQ windows")
        from atk_diffusion.dsp import denoise_runtime as _rt
        _rt.check_geometry(inp["stft"], fft, hop, geom["window"], pool, pool_mode,
                           what="this experiment's tile")
        seen_kinds = list(inp.get("kinds", []))
    methods = ["raw", *CLASSICAL] + (["diffusion"] if den is not None else [])
    noise = _dn.NoiseSource(prof, fs, [noise_capture] if noise_capture else (), rng)
    say(f"noise: {noise.words}")
    F = _dn.floor_from_noise(noise, geom, frames=1024)
    n_win = _cl.samples_for_rows(rows, fft, hop, pool)
    p_n = float(np.mean(_dn.floor_from_noise(noise, {**geom, "pool": 1}, frames=256)))
    ref = _dn.noise_tiles(noise, geom, F, 16, rows, min(bins, fft))
    mu_n, sd_n = float(np.mean(ref)), float(np.std(ref))
    noise_lin = float(np.mean(np.power(10.0, ref / 10.0)))
    det = _detector(detector, geom, fs, min_cells)
    say(f"detector: {det.name}; tile {rows} rows; floor noise {mu_n:.2f} ± {sd_n:.2f} dB")

    def outputs(spec):
        out = {"raw": spec}
        for m in CLASSICAL:
            kw = {"noise_lin": noise_lin} if m == "wiener" else (
                {"sigma_db": sd_n} if m == "wavelet" else {})
            out[m] = _cl.clean_tile(spec, m, **kw).out
        if den is not None:
            out["diffusion"] = den.denoise(spec, noise_std=sd_n, noise_mean_db=mu_n,
                                           steps=denoise_steps)
        return out

    result = {"name": name, "profile": pid, "profile_words": profiles.describe(pid),
              "fs": fs, "geometry": geom, "noise": noise.words, "detector": det.name,
              "pfa_tile": float(pfa), "cell_pfa_nominal": cell_pfa,
              "snrs_db": [float(s) for s in snrs_db], "trials": int(trials),
              "noise_trials": int(noise_trials), "snr_definition": SNR_DEFINITION,
              "methods": methods,
              "method_tiers": {m: ("record" if m == "raw" else provenance.tier_for(
                  "diffusion_denoise" if m == "diffusion" else m)) for m in methods},
              "floor_noise_db": {"mean": mu_n, "std": sd_n},
              "denoiser": None, "bursts": {}, "skipped": {}}
    if den is not None:
        result["denoiser"] = {"name": den.card.name, "sha256": den.card.weights.get("sha256"),
                              "t_star": den.card.input.get("t_star"),
                              "card_hallucination_rate": (den.card.metrics or {}).get("hallucination_rate"),
                              "trained_kinds": seen_kinds}
    for kind in bursts:
        try:
            probe, _t = _aug.burst_samples(kind, fs, 64, np.random.default_rng(0))
        except ValueError as e:
            result["skipped"][kind] = str(e)
            say(f"{kind}: skipped — {e}")
            continue
        # a stable per-kind stream (str hash() is salted per process)
        krng = np.random.default_rng([int(seed), zlib.crc32(kind.encode("utf-8"))])
        want = int(round(_aug.natural_duration(kind) * fs))
        nb = int(np.clip(want, 16, n_win))
        bw0 = _aug.occupied_bandwidth(_aug.burst_samples(kind, fs, nb, krng)[0], fs)
        mult = int((den.card.input.get("multiple", 1)) if den is not None else 1)
        W = int(min(fft, max(bins, math.ceil(1.25 * bw0 / (fs / fft) / mult) * mult)))
        W -= W % mult
        say(f"{kind}: tile {rows} x {W} bins, burst {nb / fs * 1e3:.2f} ms "
            f"(natural {want / fs * 1e3:.2f} ms), ~{bw0 / 1e3:.1f} kHz")

        def make(snr_db):
            x = noise.draw(n_win)
            b0 = int(krng.integers(0, fft - W + 1))
            s, tr = _aug.burst_samples(kind, fs, nb, krng)
            bw = _aug.occupied_bandwidth(s, fs)
            bin_hz = fs / fft
            f_lo, f_hi = (b0 - fft // 2) * bin_hz, (b0 + W - fft // 2) * bin_hz
            lo, hi = f_lo + bw / 2, f_hi - bw / 2
            f0 = float(krng.uniform(lo, hi)) if hi > lo else 0.5 * (f_lo + f_hi)
            if snr_db is None:
                # noise only; the matched filter still gets a template of the
                # kind at a random frequency, as it would in a signal trial
                spec = _cl.db_above(_cl.power_tile(x, fft, hop, pool, pool_mode,
                                                   geom["window"]), F)[:rows, b0:b0 + W]
                return _Trial(spec, x, _aug.shift(s, f0, fs)), None
            kb = np.clip(((np.array([f0 - bw / 2, f0 + bw / 2]) / bin_hz)
                          + fft // 2).astype(int), 0, fft - 1)
            fbar = float(np.mean(F[kb[0]:kb[1] + 1]))
            a = math.sqrt(10 ** (snr_db / 10.0) * fbar * max(bw, bin_hz) / fs)
            st = int(krng.integers(0, n_win - s.size + 1))
            tmpl = np.zeros(n_win, dtype=np.complex64)
            tmpl[st:st + s.size] = a * _aug.shift(s, f0, fs)
            y = x + tmpl
            spec = _cl.db_above(_cl.power_tile(y, fft, hop, pool, pool_mode,
                                               geom["window"]), F)[:rows, b0:b0 + W]
            info = {"bw_hz": bw, "fullband_snr_db": 10 * math.log10(a * a / p_n),
                    "ebn0_db": 10 * math.log10(a * a * s.size / p_n),
                    "generator": tr.get("generator", "local")}
            return _Trial(spec, y, tmpl[st:st + s.size]), info

        # noise-only trials: calibration and hallucination
        n_pmin = {m: [] for m in methods}
        n_mf = []
        halluc = {m: {"eligible": 0, "hallucinated": 0} for m in methods if m != "raw"}
        for _ in range(int(noise_trials)):
            tr, _ = make(None)
            outs = outputs(tr.spec)
            pm = {m: _pmin(det, det.prepare(o), grid) for m, o in outs.items()}
            for m in methods:
                n_pmin[m].append(pm[m])
            n_mf.append(matched_statistic(tr.iq, tr.template, p_n))
            if pm["raw"] > cell_pfa:                    # raw: nothing at nominal
                for m in halluc:
                    halluc[m]["eligible"] += 1
                    if pm[m] <= cell_pfa:
                        halluc[m]["hallucinated"] += 1
        op = {m: operating_point(n_pmin[m], pfa, grid) for m in methods}
        mf_thr = float(np.quantile(n_mf, 1.0 - pfa)) if n_mf else float("inf")
        # signal trials
        pd = {m: [] for m in methods}
        pd_mf, fb, eb, gens, bws = [], [], [], set(), []
        for snr in snrs_db:
            hits = {m: 0 for m in methods}
            mf_hits = 0
            fbs, ebs = [], []
            for _ in range(int(trials)):
                tr, info = make(float(snr))
                outs = outputs(tr.spec)
                for m, o in outs.items():
                    if _pmin(det, det.prepare(o), grid) <= op[m][0]:
                        hits[m] += 1
                if matched_statistic(tr.iq, tr.template, p_n) > mf_thr:
                    mf_hits += 1
                fbs.append(info["fullband_snr_db"])
                ebs.append(info["ebn0_db"])
                gens.add(info["generator"])
                bws.append(info["bw_hz"])
            for m in methods:
                pd[m].append(hits[m] / max(1, int(trials)))
            pd_mf.append(mf_hits / max(1, int(trials)))
            fb.append(float(np.mean(fbs)))
            eb.append(float(np.mean(ebs)))
            say(f"{kind} @ {snr:+.1f} dB: " + ", ".join(f"{m} {pd[m][-1]:.2f}" for m in methods)
                + f", matched filter {pd_mf[-1]:.2f}")
        result["bursts"][kind] = {
            "words": BURST_WORDS.get(kind, kind),
            "seen_by_denoiser": (kind in seen_kinds) if den is not None else None,
            "generator": sorted(gens), "tile": [rows, W],
            "duration_s": nb / fs, "natural_duration_s": want / fs, "clipped": nb < want,
            "bw_hz": float(np.median(bws)) if bws else None,
            "pd": pd, "pd_matched_filter": pd_mf,
            "operating_point": {m: {"cell_pfa": op[m][0], "tile_pfa_realised": op[m][1]}
                                for m in methods},
            "matched_filter_tile_pfa": float(np.mean(np.asarray(n_mf) > mf_thr)) if n_mf else None,
            "snr_at_pd": {**{m: {"0.5": snr_at_pd(snrs_db, pd[m], 0.5),
                                 "0.9": snr_at_pd(snrs_db, pd[m], 0.9)} for m in methods},
                          "matched_filter": {"0.5": snr_at_pd(snrs_db, pd_mf, 0.5),
                                             "0.9": snr_at_pd(snrs_db, pd_mf, 0.9)}},
            "hallucination": {m: {**h, "rate": (h["hallucinated"] / h["eligible"])
                                  if h["eligible"] else None} for m, h in halluc.items()},
            "fullband_snr_db": fb, "ebn0_db": eb}
    result["seconds"] = round(time.time() - t_start, 1)
    result["verdict"] = _verdict(result)
    target = out_dir
    if target is None and rf is not None:
        target = report_dir(rf, pid, name)
    if target is not None:
        summary = [f"The weak-burst test on {result['profile_words']}: noise from "
                   f"{result['noise']}.",
                   f"Detector: {result['detector']}; every method held at a tile "
                   f"false-alarm rate of {result['pfa_tile']:g}; {SNR_DEFINITION}.",
                   *result["verdict"],
                   "Diffusion outputs are INVENTED tier; the matched filter knows the "
                   "waveform and the others do not. Tables: the _detail.md beside this."]
        md, js, det = write_report(target, name, result, summary,
                                   detail=report_lines(result), rf=rf)
        result["report_md"], result["report_json"] = str(md), str(js)
        result["report_detail_md"] = str(det)
    return result


def _fmt_snr(v) -> str:
    if v is None:
        return "not reached"
    return ("≤ " if v["at_or_below"] else "") + f"{v['snr_db']:+.1f} dB"


def _verdict(result: dict) -> list[str]:
    out = []
    if result["denoiser"] is None:
        out.append("No diffusion denoiser was given: this run is the classical "
                   "baseline only. Train one with learn.denoiser.train_denoiser "
                   "and pass its folder as denoiser=.")
    for kind, b in result["bursts"].items():
        s = b["snr_at_pd"]
        cls = [s[m]["0.9"]["snr_db"] for m in ("raw", *CLASSICAL) if s[m]["0.9"]]
        best = min(cls) if cls else None
        if "diffusion" in s:
            d = s["diffusion"]["0.9"]
            h = b["hallucination"].get("diffusion", {}).get("rate")
            if d is None:
                out.append(f"{kind}: the diffusion denoiser never reached Pd 0.9 "
                           "in this SNR range.")
            elif best is None or d["snr_db"] < best:
                gain = "" if best is None else f" ({best - d['snr_db']:.1f} dB better than the best classical)"
                out.append(f"{kind}: diffusion reaches Pd 0.9 at {_fmt_snr(d)}{gain}"
                           + (f" — with a hallucination rate of {h:.3f} on noise-only tiles"
                              if h is not None else ""))
            else:
                out.append(f"{kind}: diffusion does NOT beat the classical denoisers "
                           f"(Pd 0.9 at {_fmt_snr(d)} vs {best:+.1f} dB). Plan §7: not shipped.")
    return out


def report_lines(result: dict) -> list[str]:
    """The markdown report, written from the facts in `result`."""
    L = [f"# The weak-burst test — {result['profile_words']} ({result['profile']})", "",
         f"- noise: {result['noise']}",
         f"- detector: {result['detector']}; tile false-alarm rate held at "
         f"{result['pfa_tile']:g} (calibrated per method on {result['noise_trials']} "
         "noise-only tiles); hallucination judged at the profile's nominal cell "
         f"false-alarm rate {result['cell_pfa_nominal']:g}",
         f"- {result['snr_definition']}",
         f"- {result['trials']} trials per SNR per burst; geometry "
         f"{result['geometry']['fft_size']}-point {result['geometry']['window']}, "
         f"{result['geometry']['pool']} frames {result['geometry']['pool_mode']}-pooled per row",
         f"- tiers: " + ", ".join(f"{m} {t}" for m, t in result["method_tiers"].items())]
    if result["denoiser"]:
        d = result["denoiser"]
        L.append(f"- diffusion denoiser: {d['name']} (sha256 {str(d['sha256'])[:12]}…), "
                 f"t* = {d['t_star']}; its card's hallucination rate "
                 f"{d['card_hallucination_rate']}")
    L += ["", "**Every diffusion output is INVENTED tier** — "
          + provenance.TIER_WORDS["invented"], ""]
    for kind, b in result["bursts"].items():
        meths = list(b["pd"])
        L += [f"## {kind} — {b['words']}", "",
              f"tile {b['tile'][0]} x {b['tile'][1]} bins; burst "
              f"{b['duration_s'] * 1e3:.2f} ms"
              + (f" (clipped from {b['natural_duration_s'] * 1e3:.2f} ms)" if b["clipped"] else "")
              + f"; ~{(b['bw_hz'] or 0) / 1e3:.1f} kHz; generator {', '.join(b['generator'])}"
              + ("" if b["seen_by_denoiser"] is None else
                 f"; {'SEEN' if b['seen_by_denoiser'] else 'NOT seen'} by the denoiser in training"),
              "", "| SNR in-band (dB) | full-band (dB) | E/N0 (dB) | "
              + " | ".join(meths) + " | matched filter |",
              "|" + "---|" * (len(meths) + 4)]
        for i, s in enumerate(result["snrs_db"]):
            L.append(f"| {s:+.1f} | {b['fullband_snr_db'][i]:+.1f} | {b['ebn0_db'][i]:+.1f} | "
                     + " | ".join(f"{b['pd'][m][i]:.2f}" for m in meths)
                     + f" | {b['pd_matched_filter'][i]:.2f} |")
        L += ["", "| method | SNR at Pd 0.5 | SNR at Pd 0.9 | tile Pfa realised | hallucination |",
              "|---|---|---|---|---|"]
        for m in meths + ["matched_filter"]:
            op = b["operating_point"].get(m, {})
            h = b["hallucination"].get(m)
            hs = ("—" if h is None else
                  (f"{h['rate']:.3f} ({h['hallucinated']}/{h['eligible']})" if h["rate"] is not None
                   else "no eligible tiles"))
            pr = (f"{op['tile_pfa_realised']:.3f}" if op else
                  (f"{b['matched_filter_tile_pfa']:.3f}" if b['matched_filter_tile_pfa'] is not None else "—"))
            L.append(f"| {m} | {_fmt_snr(b['snr_at_pd'][m]['0.5'])} | "
                     f"{_fmt_snr(b['snr_at_pd'][m]['0.9'])} | {pr} | {hs} |")
        L.append("")
    if result["skipped"]:
        L += ["## Not run", ""] + [f"- {k}: {v}" for k, v in result["skipped"].items()] + [""]
    L += ["## Verdict", ""] + [f"- {v}" for v in result["verdict"]]
    L += ["", "The matched filter knows the exact waveform; the other columns do "
          "not. On synthetic noise this measures the method; the number that "
          "counts comes from Bill's own captures (plan §7)."]
    return L
