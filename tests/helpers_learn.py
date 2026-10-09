# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Tiny datasets in the exact ARCHITECTURE §5 layout, for the learn/ and
experiments/ tests.

The real builder (`atk_diffusion.synth.datasets`) is written concurrently by
another engineer; these helpers let the training and evaluation code be
tested without it, against the layout the contract fixes:

    <rf_data>/<profile>/datasets/<name>/manifest.json
    train/ val/ test/
      narrowband: shard_NNN.npz  iq (N, L) complex64, label (N,) int32,
                  family (N,) int32, snr_db, symbol_rate_hz,
                  carrier_offset_hz, bandwidth_hz (N,) float32,
                  [scf (N, H, W) float16]
      wideband:   tiles_NNN.npz  spec (N, rows, bins) float16 dB above floor,
                  boxes (M, 6) float32 [tile, row0, bin0, row1, bin1, family],
                  box_class (M,) int32

Two optional fields the learn/ code reads when present are also written
where a test needs them: `t_s` (seconds since an on-site session began) and
`box_snr_db` (per box). The signals are simple and deliberately easy: the
tests check that the code paths work and learn, not that a model is good.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from atk_diffusion import profiles as P
from atk_diffusion import sigmf
from atk_diffusion.detect import classes as C
from atk_diffusion.provenance import sha256_path

PROFILE = "rtlsdr_2400000_cu8"

#: family -> (a class of that family from the class table v1)
FAMILY_CLASS = {"fm": "nfm_voice", "burst": "adsb", "ofdm": "ref_ofdm",
                "fsk": "ref_2fsk", "psk_qam": "ref_qpsk", "am": "ref_am",
                "spread": "lora", "unknown": "spur"}


def _write_manifest(d: Path, manifest: dict) -> Path:
    files = {}
    for split in ("train", "val", "test"):
        sd = d / split
        if sd.is_dir():
            for f in sorted(sd.iterdir()):
                files[f"{split}/{f.name}"] = sha256_path(f)
    manifest = dict(manifest)
    manifest["files"] = files
    manifest.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime()))
    p = d / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Wideband
# ---------------------------------------------------------------------------
def _paint(spec_lin, fam, rng, rows, bins):
    """Draw one signal of family `fam` into a linear-power tile; returns its
    box (row0, bin0, row1, bin1)."""
    if fam == "fm":
        w = int(rng.integers(6, 12))
        h = int(rng.integers(rows // 2, rows + 1))
    elif fam == "burst":
        w = int(rng.integers(bins // 6, bins // 3))
        h = int(rng.integers(4, 9))
    elif fam == "ofdm":
        w = int(rng.integers(bins // 4, bins // 2))
        h = int(rng.integers(rows // 3, rows // 2 + 1))
    else:                                   # fsk: two lines
        w = int(rng.integers(10, 16))
        h = int(rng.integers(rows // 3, rows // 2 + 1))
    r0 = int(rng.integers(0, rows - h + 1))
    b0 = int(rng.integers(0, bins - w + 1))
    return r0, b0, r0 + h, b0 + w


def make_wideband(rf, profile: str = PROFILE, name: str = "wb_test",
                  splits=None, rows: int = 64, bins: int = 128,
                  families=("fm", "burst", "ofdm"), per_tile=(1, 3),
                  snr_range=(10.0, 20.0), seed: int = 0,
                  noise_only: bool = False, with_negative: bool = False,
                  with_t_s: bool = False, tiles_per_file: int = 8,
                  generator: str = "test-helper",
                  label_sources=("synthetic",)) -> Path:
    """A wideband dataset of rows × bins tiles, dB above a unit floor."""
    splits = dict(splits or {"train": 16, "val": 6, "test": 6})
    rng = np.random.default_rng(seed)
    d = Path(rf.datasets(profile, name))
    d.mkdir(parents=True, exist_ok=True)
    fam_names = list(C.FAMILIES)
    cls_names = sorted({FAMILY_CLASS[f] for f in families}
                       | ({"spur"} if with_negative else set()))
    fs = float(P.parse_profile_id(profile).sample_rate)
    geom = P.default_stft(fs)
    t_clock = 0.0
    for split, n in splits.items():
        sd = d / split
        sd.mkdir(exist_ok=True)
        for f0 in range(0, n, tiles_per_file):
            k = min(tiles_per_file, n - f0)
            spec = np.zeros((k, rows, bins), np.float16)
            boxes, bcls, bsnr, tts = [], [], [], []
            for t in range(k):
                lin = rng.exponential(1.0, size=(rows, bins))
                if not noise_only:
                    nsig = int(rng.integers(per_tile[0], per_tile[1] + 1))
                    for _ in range(nsig):
                        fam = families[int(rng.integers(0, len(families)))]
                        r0, b0, r1, b1 = _paint(lin, fam, rng, rows, bins)
                        snr = float(rng.uniform(*snr_range))
                        amp = 10 ** (snr / 10)
                        if fam == "fsk":
                            mid = (b0 + b1) // 2
                            for r in range(r0, r1):
                                c = b0 + 2 if (r // 3) % 2 else mid + 2
                                lin[r, c - 2:c + 2] += amp
                        elif fam == "fm":
                            lin[r0:r1, b0:b1] += amp * rng.uniform(
                                0.6, 1.0, size=(r1 - r0, b1 - b0))
                        else:
                            lin[r0:r1, b0:b1] += amp
                        boxes.append([t, r0, b0, r1, b1, fam_names.index(fam)])
                        bcls.append(cls_names.index(FAMILY_CLASS[fam]))
                        bsnr.append(snr)
                    if with_negative:
                        c = int(rng.integers(2, bins - 2))
                        lin[:, c:c + 1] += 10 ** 1.5
                        boxes.append([t, 0, c, rows, c + 1,
                                      fam_names.index("unknown")])
                        bcls.append(cls_names.index("spur"))
                        bsnr.append(15.0)
                spec[t] = (10 * np.log10(lin)).astype(np.float16)
                tts.append(t_clock)
                t_clock += geom.tile_seconds * (1 - geom.tile_overlap)
            arrs = {"spec": spec,
                    "boxes": np.asarray(boxes, np.float32).reshape(-1, 6),
                    "box_class": np.asarray(bcls, np.int32),
                    "box_snr_db": np.asarray(bsnr, np.float32)}
            if with_t_s:
                arrs["t_s"] = np.asarray(tts, np.float64)
            np.savez_compressed(sd / f"tiles_{f0 // tiles_per_file:03d}.npz",
                                **arrs)
    stft = {"fft_size": bins, "hop": bins, "window": "hann",
            "tile_seconds": geom.tile_seconds, "tile_rows": rows,
            "tile_overlap": geom.tile_overlap}
    _write_manifest(d, {"name": name, "profile": profile, "sample_rate": fs,
                        "kind": "wideband", "generator": generator,
                        "stft": stft, "fam": {}, "classes": cls_names,
                        "families": fam_names, "splits": splits,
                        "label_sources": list(label_sources), "environment": "",
                        "resampled": False, "params": {"seed": seed}})
    return d


# ---------------------------------------------------------------------------
# Narrowband
# ---------------------------------------------------------------------------
def _symbols(cls: str, n: int, rng):
    if cls == "ref_bpsk":
        return rng.choice([-1.0, 1.0], n).astype(np.complex128)
    if cls == "ref_qpsk":
        return ((rng.choice([-1, 1], n) + 1j * rng.choice([-1, 1], n))
                / np.sqrt(2))
    if cls == "ref_8psk":
        return np.exp(1j * (np.pi / 4) * rng.integers(0, 8, n))
    return None


def synth_signal(cls: str, L: int, fs: float, rng, sps: int | None = None,
                 offset_hz: float | None = None):
    """(x complex64 unit power, symbol_rate_hz, carrier_offset_hz, bw_hz)."""
    sps = int(sps or rng.choice([4, 8]))
    rs = fs / sps
    f0 = float(rng.uniform(-0.04, 0.04) * fs) if offset_hz is None else offset_hz
    n = np.arange(L)
    nsym = L // sps + 2
    if cls in ("ref_bpsk", "ref_qpsk", "ref_8psk"):
        sym = _symbols(cls, nsym, rng)
        base = np.repeat(sym, sps)[:L]
        k = np.ones(max(1, sps // 2)) / max(1, sps // 2)
        base = np.convolve(base, k, mode="same")
        bw = rs * 1.5
    elif cls == "ref_2fsk":
        bits = rng.choice([-1.0, 1.0], nsym)
        freq = np.repeat(bits, sps)[:L] * (rs / 2)
        base = np.exp(1j * 2 * np.pi * np.cumsum(freq) / fs)
        bw = 2 * rs
    elif cls == "ref_ofdm":
        k = 16
        base = np.zeros(L, np.complex128)
        for sc in range(-k // 2, k // 2):
            base += (rng.choice([-1, 1]) + 1j * rng.choice([-1, 1])) * \
                np.exp(1j * 2 * np.pi * sc * (fs / 64) * n / fs)
        rs = 0.0
        bw = k * fs / 64
    elif cls == "ref_am":
        tone = float(rng.uniform(0.01, 0.05) * fs)
        base = 1.0 + 0.7 * np.cos(2 * np.pi * tone * n / fs)
        rs = 0.0
        bw = 2 * tone
    else:
        raise ValueError(cls)
    x = base * np.exp(1j * (2 * np.pi * f0 * n / fs + rng.uniform(0, 2 * np.pi)))
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    return x.astype(np.complex64), float(rs), float(f0), float(bw)


def caf_image(x, H: int, W: int) -> np.ndarray:
    """A cheap cyclic-autocorrelation magnitude image |R_x^α(τ)|, lags 0..H-1
    by the first W cycle-frequency bins — the SCF's Fourier partner, a
    stand-in for the real SCF the cyclo engineer computes."""
    x = np.asarray(x, np.complex128)
    L = x.size
    nfft = 2 * W
    img = np.zeros((H, W))
    p = np.mean(np.abs(x) ** 2) + 1e-12
    for tau in range(H):
        r = x[: L - tau] * np.conj(x[tau:])
        img[tau] = np.abs(np.fft.fft(r, nfft))[:W] / (L * p)
    return img.astype(np.float16)


def make_narrowband(rf, profile: str = PROFILE, name: str = "nb_test",
                    classes=("ref_bpsk", "ref_qpsk", "ref_2fsk"),
                    per_class=None, L: int = 256, with_scf: bool = True,
                    scf_shape=(8, 32), snr_range=(0.0, 20.0), seed: int = 0,
                    canonical: dict | None = None, with_t_s: bool = False,
                    per_shard: int = 64, generator: str = "test-helper",
                    manifest_classes=None, label_sources=("synthetic",)) -> Path:
    """A narrowband dataset at the profile's voice-class canonical rate."""
    per_class = dict(per_class or {"train": 24, "val": 10, "test": 10})
    rng = np.random.default_rng(seed)
    fs_prof = float(P.parse_profile_id(profile).sample_rate)
    if canonical is None:
        can = P.canonical_rates(fs_prof)[0]
        canonical = {"class": can.cls, "rate": can.rate,
                     "decimation": can.decimation}
    fs = float(canonical["rate"])
    names = list(manifest_classes or classes)
    fam_names = list(C.FAMILIES)
    d = Path(rf.datasets(profile, name))
    d.mkdir(parents=True, exist_ok=True)
    counts = {}
    t_clock = 0.0
    for split, n_each in per_class.items():
        sd = d / split
        sd.mkdir(exist_ok=True)
        rows = []
        for cls in classes:
            for _ in range(int(n_each)):
                rows.append(cls)
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        counts[split] = len(rows)
        for s0 in range(0, len(rows), per_shard):
            chunk = rows[s0:s0 + per_shard]
            k = len(chunk)
            iq = np.zeros((k, L), np.complex64)
            lab = np.zeros(k, np.int32)
            fam = np.zeros(k, np.int32)
            snr = np.zeros(k, np.float32)
            srate = np.zeros(k, np.float32)
            coff = np.zeros(k, np.float32)
            bw = np.zeros(k, np.float32)
            scf = np.zeros((k,) + tuple(scf_shape), np.float16)
            tts = np.zeros(k, np.float64)
            for i, cls in enumerate(chunk):
                x, rs, f0, b = synth_signal(cls, L, fs, rng)
                sn = float(rng.uniform(*snr_range))
                noise = (rng.normal(size=L) + 1j * rng.normal(size=L)) / np.sqrt(2)
                y = (x + noise * 10 ** (-sn / 20)).astype(np.complex64)
                iq[i] = y
                lab[i] = names.index(cls)
                fam[i] = fam_names.index(C.get(cls).family)
                snr[i], srate[i], coff[i], bw[i] = sn, rs, f0, b
                if with_scf:
                    scf[i] = caf_image(y, *scf_shape)
                tts[i] = t_clock
                t_clock += L / fs
            arrs = {"iq": iq, "label": lab, "family": fam, "snr_db": snr,
                    "symbol_rate_hz": srate, "carrier_offset_hz": coff,
                    "bandwidth_hz": bw}
            if with_scf:
                arrs["scf"] = scf
            if with_t_s:
                arrs["t_s"] = tts
            np.savez(sd / f"shard_{s0 // per_shard:03d}.npz", **arrs)
    _write_manifest(d, {"name": name, "profile": profile,
                        "sample_rate": fs_prof, "kind": "narrowband",
                        "generator": generator, "canonical": canonical,
                        "stft": {}, "fam": {"channel_fft": 64, "hop": 16,
                                            "window": "hamming"},
                        "classes": names, "families": fam_names,
                        "splits": counts, "label_sources": list(label_sources),
                        "environment": "", "resampled": False,
                        "params": {"seed": seed, "L": L}})
    return d


# ---------------------------------------------------------------------------
# A recorded capture (for self-supervised pretraining from captures)
# ---------------------------------------------------------------------------
def make_capture(rf, profile: str = PROFILE, seconds: float = 0.05,
                 seed: int = 0, name: str = "cap_test") -> Path:
    """A short cu8 SigMF capture at the profile's rate: a few QPSK and FSK
    signals at offsets across the span, in noise."""
    rng = np.random.default_rng(seed)
    fs = float(P.parse_profile_id(profile).sample_rate)
    n = int(seconds * fs)
    t = np.arange(n)
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.02
    for off in (-600e3, -150e3, 300e3, 700e3):
        sps = int(rng.choice([50, 100, 200]))
        sym = (rng.choice([-1, 1], n // sps + 1)
               + 1j * rng.choice([-1, 1], n // sps + 1)) / np.sqrt(2)
        base = np.repeat(sym, sps)[:n]
        x = x + 0.1 * base * np.exp(2j * np.pi * off * t / fs)
    base = Path(rf.captures(profile)) / name
    sigmf.write_pair(base, x.astype(np.complex64), fs, 915e6, datatype="cu8",
                     hw="RTL-SDR v3",
                     extra_global={"atk:receiver_profile": profile})
    return base
