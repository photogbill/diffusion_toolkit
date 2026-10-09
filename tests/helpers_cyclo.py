# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Test signals with known truth for the cyclostationary tests (not a test
module). Every generator returns unit-power complex samples, so an SNR is
set by one scale factor, and states its truth (rate, carrier, timing).

The OFDM signal is "LTE-like": random QPSK on 72 used subcarriers of a
15 kHz grid, IFFT, the normal cyclic prefix (slots of 7 symbols with the
first prefix longer), so a cyclic-prefix probe meets the same structure a
cell's downlink has — without PSS/SSS/CRS, which only make a real cell
easier to see.
"""

from __future__ import annotations

import numpy as np


def rrc_taps(beta: float, sps: int, span: int = 10) -> np.ndarray:
    """Root-raised-cosine, unit energy, odd length, centred."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1)
    t = n / float(sps)
    h = np.empty(t.size)
    for i, tt in enumerate(t):
        if abs(tt) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(4 * beta * tt) - 1) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = ((np.sin(np.pi * tt * (1 - beta))
                     + 4 * beta * tt * np.cos(np.pi * tt * (1 + beta)))
                    / (np.pi * tt * (1 - (4 * beta * tt) ** 2)))
    return h / np.sqrt(np.sum(h ** 2))


def linmod(kind: str, fs: float, rate: float, n: int, rng, beta: float = 0.35,
           fc: float = 0.0, rect: bool = False, delay: int = 0) -> np.ndarray:
    """BPSK or QPSK at `rate` (fs/rate must be an integer), RRC or
    rectangular pulses, carrier offset `fc`. Symbol centres fall at sample
    `delay` + k·sps (the timing truth)."""
    sps = int(round(fs / rate))
    assert abs(sps - fs / rate) < 1e-9, "fs/rate must be an integer"
    nsym = n // sps + 24
    if kind == "bpsk":
        a = rng.choice([-1.0, 1.0], nsym).astype(complex)
    else:
        a = (rng.choice([-1, 1], nsym) + 1j * rng.choice([-1, 1], nsym)) / np.sqrt(2)
    if rect:
        x = np.repeat(a, sps)
        x = np.roll(x, sps // 2)            # pulse centres at k·sps
    else:
        up = np.zeros(nsym * sps, complex)
        up[::sps] = a
        x = np.convolve(up, rrc_taps(beta, sps), mode="same")
    x = np.roll(x, int(delay))[:n]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    return x * np.exp(2j * np.pi * fc * np.arange(n) / fs)


def _rc(beta: float, sps: int, span: int = 8) -> np.ndarray:
    t = np.arange(-span * sps, span * sps + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
    sing = np.isclose(np.abs(2 * beta * t), 1.0)
    h[sing] = np.pi / 4 * np.sinc(1 / (2 * beta))
    return h / h.sum()


def fsk4(fs: float, rate: float, n: int, rng, dev: float = 600.0,
         shape: str = "rc", beta: float = 0.2) -> np.ndarray:
    """C4FM-like 4FSK: symbols ±1, ±3 × `dev` Hz, frequency pulses shaped by
    a raised cosine (P25 / DMR style) or rectangular (`shape="rect"`),
    continuous phase, unit power."""
    sps = int(round(fs / rate))
    nsym = n // sps + 20
    sym = rng.choice([-3.0, -1.0, 1.0, 3.0], nsym)
    if shape == "rect":
        f = np.repeat(sym, sps)
    else:
        up = np.zeros(nsym * sps)
        up[::sps] = sym
        f = np.convolve(up, _rc(beta, sps) * sps, mode="same")
    f = f[:n] * dev
    return np.exp(1j * 2 * np.pi * np.cumsum(f) / fs)


def lte_like(fs: float, n: int, rng, nfft: int = 128, used: int = 72,
             cps: tuple = (10, 9, 9, 9, 9, 9, 9), cfo: float = 0.0
             ) -> np.ndarray:
    """LTE-like downlink: 15 kHz subcarriers at fs = 15 kHz·nfft, `used`
    QPSK subcarriers around (not on) DC, slots of len(cps) symbols with the
    given prefix lengths (LTE normal CP at 1.92 MS/s: 10, then six of 9 —
    0.5 ms slots, 7 symbols, average symbol rate 14 kHz)."""
    qpsk = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    k = np.concatenate([np.arange(1, used // 2 + 1),
                        np.arange(nfft - used // 2, nfft)])
    out, total = [], 0
    while total < n:
        for cp in cps:
            F = np.zeros(nfft, complex)
            F[k] = qpsk[rng.integers(0, 4, k.size)]
            t = np.fft.ifft(F)
            out.append(np.concatenate([t[-cp:], t]))
            total += cp + nfft
    x = np.concatenate(out)[:n]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    return x * np.exp(2j * np.pi * cfo * np.arange(n) / fs)


def noise(n: int, rng, power: float = 1.0) -> np.ndarray:
    return np.sqrt(power / 2) * (rng.standard_normal(n)
                                 + 1j * rng.standard_normal(n))


def inband_amplitude(snr_db: float, occupied_hz: float, fs: float,
                     noise_power: float = 1.0) -> float:
    """Amplitude for a unit-power signal so its in-band per-Hz SNR (its
    spectrum level against the white floor) is `snr_db`. 0 dB = the signal
    level WITH the floor (3 dB above it in total); −10 dB = a tenth of it."""
    floor_per_hz = noise_power / fs
    return float(np.sqrt(10 ** (snr_db / 10) * floor_per_hz * occupied_hz))


def mix(sig: np.ndarray, offset_hz: float, fs: float) -> np.ndarray:
    return sig * np.exp(2j * np.pi * offset_hz * np.arange(sig.size) / fs)


def uca(M: int, theta: float, r_over_lambda: float = 0.5) -> np.ndarray:
    """Steering vector of an M-element uniform circular array (the Kraken's
    five antennas on a circle), radius in wavelengths."""
    ang = 2 * np.pi * np.arange(M) / M
    return np.exp(2j * np.pi * r_over_lambda * np.cos(theta - ang))


def sinad(y, s, trim: int = 3000) -> float:
    """Signal-to-everything-else after the best scalar fit of the truth s
    (noise, interference and distortion all count against it)."""
    y = np.asarray(y)[trim:-trim]
    s = np.asarray(s)[trim:-trim]
    g = np.vdot(s, y) / np.vdot(s, s)
    e = y - g * s
    return float(10 * np.log10(abs(g) ** 2 * np.vdot(s, s).real
                               / np.vdot(e, e).real))


def per_frame_cfar(x, fs: float, nfft: int = 1024, pfa: float = 1e-4,
                   guard: int = 2, train: int = 16):
    """The energy proposer as DETECTION_DESIGN §3 defines it — cell-
    averaging CFAR on the PER-FRAME periodogram — reduced to what the tests
    need: a boolean [frames, bins] mask of cells over threshold and the
    frequencies of the bins (fftshifted). The threshold factor is the
    exact CA-CFAR one for exponential cells, N·(pfa^(−1/N) − 1)."""
    x = np.asarray(x)
    nfr = x.size // nfft
    X = np.fft.fftshift(np.fft.fft(x[: nfr * nfft].reshape(nfr, nfft)
                                   * np.hanning(nfft), axis=1), axes=1)
    P = np.abs(X) ** 2
    N = 2 * train
    alpha = N * (pfa ** (-1.0 / N) - 1.0)
    k = np.concatenate([np.arange(-guard - train, -guard),
                        np.arange(guard + 1, guard + train + 1)])
    ref = np.zeros_like(P)
    for d in k:
        ref += np.roll(P, d, axis=1)
    mask = P > alpha * ref / N
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
    return mask, freqs
