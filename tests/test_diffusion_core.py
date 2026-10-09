# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The diffusion core (learn.diffusion) and the U-Nets (learn.unet).

The samplers and the SNR <-> timestep map are checked against an EXACT
denoiser: for Gaussian data x0 ~ N(0, C) the posterior mean E[x0 | x_t] has a
closed form, so the theory (arXiv 2501.07030) is tested on its own terms,
not through a network's training luck.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from atk_diffusion.learn import diffusion as D


def _oracle(C: np.ndarray, ac: np.ndarray):
    """eps_fn for x0 ~ N(0, C) on x of shape [B, 1, n] (numpy)."""
    lam, U = np.linalg.eigh(C)
    lam = np.clip(lam, 0.0, None)

    def eps_fn(x, t):
        a = ac[int(np.asarray(t).reshape(-1)[0])]
        g = math.sqrt(a) * lam / (a * lam + 1.0 - a)
        xt = x[:, 0, :]
        m = ((xt @ U) * g) @ U.T
        return ((xt - math.sqrt(a) * m) / math.sqrt(1.0 - a))[:, None, :]
    return eps_fn


def _spread_cov(n=32, seed=0):
    """A full-rank 'signal' covariance with unit power per element and an
    eigen-spectrum falling 3 dB per component — so at every SNR some
    components sit near the noise level, where the choice of step matters
    (a low-rank covariance hides the effect: its noise dims are zeroed at
    any step)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = 2.0 ** (-np.arange(n) / 2.0)
    C = (Q * lam) @ Q.T
    return C * (n / np.trace(C))


def _draw(C, b, rng):
    lam, U = np.linalg.eigh(C)
    z = rng.standard_normal((b, C.shape[0]))
    return (z * np.sqrt(np.clip(lam, 0, None))) @ U.T


# -- schedules ---------------------------------------------------------------
@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedules_are_monotone_and_round_trip_exactly(kind):
    s = D.make_schedule(kind, 1000)
    ac = s.alphas_cumprod
    assert ac.shape == (1000,)
    assert np.all(np.diff(ac) < 0)
    assert ac[0] > 0.99 and ac[-1] < 0.01
    assert np.all((s.betas > 0) & (s.betas < 1))
    back = D.Schedule.from_json(s.to_json())
    assert np.array_equal(back.alphas_cumprod, ac)


def test_short_schedules_reach_the_same_noise():
    a = D.make_schedule("linear", 100).alphas_cumprod[-1]
    b = D.make_schedule("linear", 1000).alphas_cumprod[-1]
    assert a < 0.01 and b < 0.01


def test_unknown_schedule_is_refused_in_words():
    with pytest.raises(ValueError, match="unknown noise schedule"):
        D.make_schedule("sigmoid", 10)


# -- SNR <-> timestep --------------------------------------------------------
def test_snr_matched_timestep_inverts_the_forward_snr():
    ac = D.make_schedule("cosine", 1000).alphas_cumprod
    log_snr = np.log(D.forward_snr(ac))
    prev = None
    for snr_db in np.arange(-10.0, 31.0, 2.5):
        t = D.snr_matched_timestep(10 ** (snr_db / 10), ac)
        assert isinstance(t, int)
        gap = max(abs(log_snr[max(t - 1, 0)] - log_snr[t]),
                  abs(log_snr[min(t + 1, 999)] - log_snr[t]))
        assert abs(log_snr[t] - math.log(10 ** (snr_db / 10))) <= gap
        assert abs(D.timestep_snr_db(t, ac) - snr_db) < 1.0
        if prev is not None:
            assert t <= prev            # more SNR, less noise, earlier step
        prev = t
    assert D.snr_matched_timestep(0.0, ac) == 999      # noise only: generation
    assert D.snr_matched_timestep(-1.0, ac) == 999
    assert D.snr_matched_timestep(1e12, ac) == 0
    arr = D.snr_matched_timestep(np.array([1.0, 10.0, 100.0]), ac)
    assert arr.shape == (3,) and arr[0] > arr[1] > arr[2]
    with pytest.raises(ValueError, match="not a number"):
        D.snr_matched_timestep(float("nan"), ac)


def test_rescaled_observation_has_unit_power_and_one_step_is_tweedie(rng):
    """x_t* = √ᾱ*·y has unit power, and ŝ = y − σ·ε̂ exactly at t*."""
    ac = D.make_schedule("cosine", 1000).alphas_cumprod
    t = 300
    snr = float(D.forward_snr(ac)[t])            # an SNR the schedule hits exactly
    sigma = 1.0 / math.sqrt(snr)
    x0 = rng.standard_normal((4000, 1, 8))
    y = x0 + sigma * rng.standard_normal(x0.shape)
    xt = D.observation_to_xt(y, D.snr_matched_timestep(snr, ac), ac)
    assert abs(float(np.mean(xt ** 2)) - 1.0) < 0.03
    k = D.noise_matched_scale(sigma, t, ac)
    assert np.allclose(k * y, xt)

    w = rng.standard_normal((8, 8)) * 0.1          # an arbitrary 'network'

    def eps_fn(x, tt):
        return np.tanh(x @ w)

    s_hat, t_used = D.denoise(eps_fn, y * 3.0, sigma * 3.0, snr, ac, steps=1)
    assert t_used == t
    assert np.allclose(s_hat, y * 3.0 - sigma * 3.0 * eps_fn(xt, None), atol=1e-9)


@pytest.mark.parametrize("snr_db", [3.0, 10.0, 20.0])
def test_denoise_at_t_star_beats_half_and_double(snr_db):
    """The plan's claim, numerically: the SNR-matched step is the best one."""
    rng = np.random.default_rng(7)
    ac = D.make_schedule("cosine", 1000).alphas_cumprod
    C = _spread_cov()
    eps_fn = _oracle(C, ac)
    snr = 10 ** (snr_db / 10)
    sigma = 1.0 / math.sqrt(snr)
    x0 = _draw(C, 3000, rng)[:, None, :]
    y = x0 + sigma * rng.standard_normal(x0.shape)
    t_star = D.snr_matched_timestep(snr, ac)
    mse = {}
    for name, t in (("t*", t_star), ("t*/2", t_star // 2), ("2t*", min(2 * t_star, 999))):
        s_hat, used = D.denoise(eps_fn, y, sigma, snr, ac, steps=1, t=t)
        assert used == t
        mse[name] = float(np.mean((s_hat - x0) ** 2))
    raw = float(np.mean((y - x0) ** 2))
    # measured margins at these SNRs: t*/2 is 7-24 % worse, 2·t* 18-190 %
    assert mse["t*"] * 1.03 < mse["t*/2"]
    assert mse["t*"] * 1.08 < mse["2t*"]
    assert mse["t*"] < raw


def test_denoise_refuses_noise_only_input():
    ac = D.make_schedule("cosine", 100).alphas_cumprod
    with pytest.raises(ValueError, match="pure generation"):
        D.denoise(lambda x, t: x, np.zeros((1, 1, 4)), 1.0, 0.0, ac)


# -- the forward process and the samplers -------------------------------------
def test_q_sample_moments(rng):
    ac = D.make_schedule("linear", 200).alphas_cumprod
    x0 = np.full((20000, 1, 1), 2.0)
    xt = D.q_sample(x0, 50, rng.standard_normal(x0.shape), ac)
    assert abs(xt.mean() - 2.0 * math.sqrt(ac[50])) < 0.03
    assert abs(xt.var() - (1 - ac[50])) < 0.03
    tv = np.array([0, 199])
    two = D.q_sample(np.ones((2, 1, 3)), tv, np.zeros((2, 1, 3)), ac)
    assert np.allclose(two[0], math.sqrt(ac[0])) and np.allclose(two[1], math.sqrt(ac[199]))


def test_timesteps_sequence():
    assert D.timesteps(500, 1) == [500, -1]
    seq = D.timesteps(500, 4)
    assert seq[0] == 500 and seq[-1] == -1 and len(seq) == 5
    assert all(a > b for a, b in zip(seq, seq[1:]))
    assert D.timesteps(0, 10) == [0, -1]


@pytest.mark.parametrize("sampler", ["ddim", "ddpm"])
def test_samplers_draw_the_oracle_distribution(sampler):
    rng = np.random.default_rng(3)
    ac = D.make_schedule("cosine", 400).alphas_cumprod
    C = np.diag([4.0, 1.0, 0.25, 0.04])
    eps_fn = _oracle(C, ac)
    x_T = rng.standard_normal((4000, 1, 4))
    if sampler == "ddim":
        x = D.ddim_sample(eps_fn, x_T, 399, ac, steps=100, eta=0.0)
    else:
        x = D.ddpm_sample(eps_fn, x_T, 399, ac, rng=rng)
    var = x[:, 0, :].var(axis=0)
    assert np.all(np.abs(var / np.diag(C) - 1.0) < 0.2), var


def test_repaint_keeps_the_known_and_fills_toward_the_conditional():
    rng = np.random.default_rng(11)
    n, b = 16, 256
    ac = D.make_schedule("cosine", 400).alphas_cumprod
    idx = np.arange(n)
    C = 0.9 ** np.abs(idx[:, None] - idx[None, :])
    eps_fn = _oracle(C, ac)
    x0 = _draw(C, 1, rng)[0]
    mask = np.ones(n)
    unk = np.arange(6, 10)
    mask[unk] = 0
    known = np.where(mask > 0)[0]
    x_known = np.tile(x0, (b, 1))[:, None, :] * mask
    out = D.repaint(eps_fn, x_known, mask[None, None, :], ac, steps=50,
                    resample=8, rng=rng)
    assert np.allclose(out[:, 0, known], x0[known])
    Ckk = C[np.ix_(known, known)]
    Cuk = C[np.ix_(unk, known)]
    mu = Cuk @ np.linalg.solve(Ckk, x0[known])
    cov = C[np.ix_(unk, unk)] - Cuk @ np.linalg.solve(Ckk, Cuk.T)
    fill = out[:, 0, unk]
    # RePaint is an approximation to the true conditional: its bias shrinks
    # as resampling grows (measured: ~0.1-0.25 of the conditional std at
    # 40x4, about half that at 60x10) and grows with |mu|.
    cond_std = math.sqrt(np.mean(np.diag(cov)))
    rms_mu = math.sqrt(np.mean(mu ** 2))
    err_mean = math.sqrt(np.mean((fill.mean(axis=0) - mu) ** 2))
    assert err_mean < 0.3 * cond_std + 0.1 * rms_mu
    ratio = fill.var(axis=0) / np.diag(cov)
    assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio


def test_importing_the_core_does_not_import_torch():
    code = ("import sys; import atk_diffusion.learn.diffusion as d; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "False"


# -- torch paths ---------------------------------------------------------------
def test_samplers_run_on_torch_tensors_and_agree_with_numpy():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    ac = D.make_schedule("cosine", 200).alphas_cumprod
    C = np.diag([2.0, 0.5, 0.1])
    np_fn = _oracle(C, ac)

    def torch_fn(x, t):
        return torch.as_tensor(np_fn(x.numpy().astype(np.float64), t.numpy()),
                               dtype=x.dtype)
    x_T = np.random.default_rng(0).standard_normal((64, 1, 3))
    a = D.ddim_sample(np_fn, x_T, 199, ac, steps=20)
    b = D.ddim_sample(torch_fn, torch.as_tensor(x_T, dtype=torch.float64), 199, ac,
                      steps=20)
    assert isinstance(b, torch.Tensor)
    assert np.allclose(a, b.numpy(), atol=1e-8)
    noisy = D.repaint(torch_fn, torch.zeros(4, 1, 3, dtype=torch.float64),
                      torch.tensor([[[1.0, 0.0, 1.0]]], dtype=torch.float64), ac,
                      steps=5, resample=2, rng=torch.Generator().manual_seed(0))
    assert noisy.shape == (4, 1, 3) and torch.all(noisy[:, :, 0] == 0)


def test_unet_shapes_conditions_and_classes():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import unet as U
    torch.manual_seed(0)
    m = U.UNet(**U.TINY_1D)
    x = torch.randn(3, 2, 32)
    t = torch.tensor([0, 10, 99])
    out = m(x, t)
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0          # zero-initialised head
    mc = U.UNet(**{**U.TINY_1D, "cond_ch": 2})
    assert mc(x, t, cond=torch.randn(3, 2, 32)).shape == x.shape
    with pytest.raises(ValueError, match="condition channels"):
        mc(x, t)
    m2 = U.UNet(**{**U.TINY_2D, "num_classes": 3})
    xx = torch.randn(2, 1, 16, 8)
    assert m2(xx, torch.tensor([5, 6]), y=torch.tensor([0, 2])).shape == xx.shape
    assert m2(xx, torch.tensor([5, 6])).shape == xx.shape   # the "no class" token
    rebuilt = U.build_unet(m2.config)
    assert U.count_params(rebuilt) == U.count_params(m2)
    assert U.multiple(U.TINY_1D) == 2
    padded, sp = U.pad_to_multiple(torch.randn(1, 2, 31), 4)
    assert padded.shape[-1] == 32 and U.crop_to(padded, sp).shape[-1] == 31


def test_tiny_unet_learns_and_eps_loss_starts_at_one():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import unet as U
    torch.manual_seed(1)
    ac = D.make_schedule("cosine", 100).alphas_cumprod
    m = U.UNet(**U.TINY_1D)
    g = torch.Generator().manual_seed(0)
    n = torch.arange(32, dtype=torch.float32)

    def batch(step, gen):
        ph = torch.rand(16, 1, generator=gen) * 6.28
        x0 = torch.stack([torch.cos(0.4 * n + ph), torch.sin(0.4 * n + ph)], dim=1) \
            .reshape(16, 2, 32) * math.sqrt(1.0)
        return {"x0": x0}

    first = D.eps_loss(m, batch(0, g)["x0"], ac, generator=g)
    assert abs(float(first.detach()) - 1.0) < 0.2  # zero model predicts no noise
    hist = D.fit(m, batch, ac, steps=80, lr=3e-3, generator=g)
    assert np.mean(hist["loss"][-15:]) < np.mean(hist["loss"][:15])


def test_onnx_export_matches_torch_with_dynamic_batch_and_length(tmp_path):
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    from atk_diffusion.learn import unet as U
    torch.manual_seed(2)
    m = U.UNet(**{**U.TINY_1D, "cond_ch": 2}).eval()
    with torch.no_grad():                      # make the head non-trivial
        m.out.weight.normal_(0, 0.1)
    p = U.export_onnx(m, tmp_path / "m.onnx", (32,))
    assert [f.name for f in tmp_path.iterdir()] == ["m.onnx"]   # one file
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    assert [i.name for i in s.get_inputs()] == ["x", "t"]
    assert [o.name for o in s.get_outputs()] == ["eps"]
    x = np.random.default_rng(0).standard_normal((3, 4, 64)).astype(np.float32)
    t = np.array([1, 50, 99], dtype=np.int64)
    got = s.run(["eps"], {"x": x, "t": t})[0]
    with torch.no_grad():
        ref = m(torch.from_numpy(x[:, :2]), torch.from_numpy(t),
                cond=torch.from_numpy(x[:, 2:])).numpy()
    assert np.allclose(got, ref, atol=1e-4)
