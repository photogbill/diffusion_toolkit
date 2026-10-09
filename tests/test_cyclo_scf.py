# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The SCF: FAM, SSCA, the cyclic profile, the fixed-geometry image
(DETECTION_DESIGN §4.1). The key checks of ATK's tests/test_siga_csp.py are
ported here — the fast surfaces against the scanned reference, the
conjugate discriminator, the derived threshold, the local floor — plus the
conjugate-mapping correction this port made."""

from __future__ import annotations

import numpy as np
import pytest

import helpers_cyclo as H
from atk_diffusion import profiles
from atk_diffusion.cyclo import scf

FS = 48_000.0


def _bpsk(rng, n=24_000, rate=2400.0, fc=600.0, snr_db=10.0):
    s = H.linmod("bpsk", FS, rate, n, rng, beta=0.35, fc=fc)
    return s + H.noise(n, rng, 10 ** (-snr_db / 10))


def _peak(alphas, prof, lo=100.0, hi=None):
    m = np.abs(alphas) > lo
    if hi:
        m &= np.abs(alphas) < hi
    return float(abs(alphas[m][np.argmax(prof[m])]))


def test_fam_finds_the_symbol_rate_and_the_conjugate_carrier(rng):
    """BPSK at 2400 Bd, carrier +600 Hz: the non-conjugate surface peaks at
    the symbol rate, the conjugate one at TWICE the carrier."""
    x = _bpsk(rng)
    a, p = scf.alpha_profile(scf.fam(x, FS, alpha_max=6000.0))
    assert _peak(a, p, lo=500) == pytest.approx(2400.0, abs=10.0)
    a, p = scf.alpha_profile(scf.fam(x, FS, alpha_max=6000.0, conj=True))
    assert float(a[np.argmax(p)]) == pytest.approx(1200.0, abs=10.0)


def test_the_conjugate_mapping_is_the_corrected_one(rng):
    """The bench's conjugate coherence multiplied X(f+α/2)·X(f−α/2) and
    showed nothing at the true α = 2·f_c (0.45 vs 0.45 elsewhere). The
    conjugate SCF needs X(α/2 − f): with it the carrier stands out."""
    x = _bpsk(rng, snr_db=20.0)
    at = scf.spectral_coherence(x, FS, 1200.0, df=FS / 64, conj=True).max()
    off = scf.spectral_coherence(x, FS, 3333.0, df=FS / 64, conj=True).max()
    assert at > 0.9 and off < 0.6
    # FAM and SSCA agree with it
    for method in ("fam", "ssca"):
        a, p = scf.cyclic_profile(x, FS, method=method, conj=True)
        assert float(a[np.argmax(p)]) == pytest.approx(1200.0, abs=15.0)


def test_qpsk_has_no_conjugate_feature(rng):
    x = H.linmod("qpsk", FS, 2400.0, 24_000, rng, fc=600.0) + H.noise(
        24_000, rng, 0.1)
    a, p = scf.cyclic_profile(x, FS, conj=True, normalized=True)
    b, q = scf.cyclic_profile(_bpsk(rng), FS, conj=True, normalized=True)
    assert p.max() < 0.5 * q.max()


def test_ssca_resolves_alpha_more_finely_than_a_budgeted_fam(rng):
    """Why both exist: FAM's Δα = fs/(hop·P) is set by how many blocks it
    can afford; SSCA's Δα = fs/N comes from the whole record."""
    x = _bpsk(rng)
    f = scf.fam(x, FS, alpha_max=6000.0, max_blocks=512)
    s = scf.ssca(x, FS, max_samples=1 << 14, alpha_max=6000.0)
    assert s.dalpha_hz < f.dalpha_hz
    a, p = scf.alpha_profile(s)
    assert _peak(a, p, lo=500) == pytest.approx(2400.0, abs=10.0)


def test_a_surface_agrees_with_the_scanned_reference(rng):
    """The fast methods against the definition (the bench's rule)."""
    x = _bpsk(rng, n=4800, snr_db=40.0)
    ref = scf.scf_scan(x, FS, [0.0, 1200.0, 2400.0, 5000.0], df=FS / 32,
                       nfft=512)
    prof = ref.magnitude.max(axis=1)
    assert prof[2] > prof[3]
    a, p = scf.alpha_profile(scf.fam(x, FS, alpha_max=6000.0,
                                     normalized=True))
    v_rate = p[np.argmin(np.abs(a - 2400.0))]
    v_off = p[np.argmin(np.abs(a - 5000.0))]
    assert v_rate > 1.5 * v_off


def test_the_scf_at_alpha_zero_is_the_power_spectrum(rng):
    x = H.linmod("qpsk", FS, 2400.0, 4096, rng)
    s0 = np.abs(scf.scf_at(x, FS, 0.0, df=FS / 64, nfft=1024))
    X = np.fft.fft(x[:1024] * np.hanning(1024))
    k = int(round((FS / 64) / (FS / 1024)))
    lin = np.convolve(np.abs(X) ** 2, np.ones(k) / k, mode="same")
    lin = np.fft.fftshift(lin)
    assert float(np.corrcoef(s0 / s0.max(), lin / lin.max())[0, 1]) > 0.95


def test_scf_image_has_the_profiles_fixed_geometry(rng):
    """The classifier's second input: the same shape and the same α and f
    axes (in units of fs) whatever the cut's length, float32 in 0..1."""
    geom = profiles.FamGeometry()
    x = _bpsk(rng, n=48_000, fc=0.0)
    img, f_ax, a_ax = scf.scf_image(x, FS, geom)
    assert img.shape == (64, 128) and img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() == pytest.approx(1.0)
    assert f_ax.size == 64 and a_ax.size == 128
    assert a_ax[0] > 0 and a_ax[-1] < FS / 2
    # the strongest cyclic column (beyond α ≈ 0) is the symbol rate's
    col = img[:, 2:].max(axis=0)
    assert abs(a_ax[2 + int(np.argmax(col))] - 2400.0) <= FS / 256
    short, _f, _a = scf.scf_image(x[:6000], FS, geom)
    assert short.shape == (64, 128)
    conj, _f2, a2 = scf.scf_image(x, FS, geom, conj=True)
    assert a2[0] < 0 < a2[-1]


def test_lag_profile_threshold_holds_on_noise_and_finds_the_rate(rng):
    """The detector's null is derived: noise alone stays under the
    threshold for a 1e-6 false-alarm rate; a BPSK clears it at its rate."""
    for seed in range(3):
        w = H.noise(24_000, np.random.default_rng(seed))
        a, st = scf.lag_profile(w, FS)
        r = scf.local_ratio(st)
        m = (a > 100) & (a < 20_000)
        assert r[m].max() < scf.detection_threshold(1e-6, int(m.sum()))
    a, st = scf.lag_profile(_bpsk(rng, snr_db=0.0), FS)
    r = scf.local_ratio(st)
    m = (a > 500) & (a < 20_000)
    assert r[m].max() > scf.detection_threshold(1e-6, int(m.sum()))
    assert float(a[m][np.argmax(r[m])]) == pytest.approx(2400.0, abs=2.0)


def test_the_local_ratio_uses_a_local_floor():
    prof = np.full(4096, 1.0)
    prof[1800:2300] = 50.0
    prof[2000] = 120.0
    assert scf.local_ratio(prof, span=256)[2000] < 10.0
    prof2 = np.full(4096, 1.0)
    prof2[2000] = 120.0
    assert scf.local_ratio(prof2, span=256)[2000] > 50.0


def test_the_threshold_moves_the_way_the_algebra_says():
    a = scf.detection_threshold(1e-3, 10_000)
    assert scf.detection_threshold(1e-9, 10_000) > a
    assert scf.detection_threshold(1e-3, 10_000_000) > a


def test_ssca_refuses_rather_than_exhausting_memory(monkeypatch, rng):
    monkeypatch.setattr(scf, "SSCA_MAX_POINTS", 1000)
    out = scf.ssca(_bpsk(rng), FS)
    assert out.magnitude.size == 0
    assert "refused" in out.note and "GB" in out.note


def test_a_surface_says_whether_its_resolution_holds(rng):
    s = scf.fam(_bpsk(rng), FS, alpha_max=6000.0)
    assert isinstance(s.valid, bool) and "Δf" in s.resolution_note()
    assert "FAM surface" in scf.fam_cost(24_000, FS)
    assert "MB" in scf.ssca_cost(24_000, FS)


def test_harmonics_fold_onto_one_fundamental():
    fund, orders = scf.fold_harmonics([(9600.0, 5.0), (19200.0, 3.0),
                                       (28800.0, 2.0)])
    assert fund == pytest.approx(9600.0) and orders == [1, 2, 3]


def test_the_targeted_caf_is_finite():
    x = H.linmod("qpsk", FS, 2400.0, 4800, np.random.default_rng(0))
    r = scf.caf(x, FS, 2400.0, max_lag=32)
    assert r.size == 33 and np.all(np.isfinite(r))
