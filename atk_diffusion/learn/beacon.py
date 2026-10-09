# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""A beacon co-designed with its detector — a DESIGN STUDY (plan §4.R).

    NOTHING IN THIS MODULE TRANSMITS. It learns and scores waveforms in
    simulation. Transmitting any beacon — this one or any other — would need
    proper authorisation (a licence or the spectrum authority's permission
    for the band, the power and the place). ATK's RF work is receive-only;
    the cabled loop (`cabled.*`) is the one approved exception, and it puts
    signals on a cable, never in the air.

THE IDEA (DARPA RFMLS task 2; DeepSig's channel-autoencoder work, plan §9):
learn a waveform that is easy for OUR detector to find, at low SNR, through
THIS receiver. For AURA: a relief beacon that the field laptop hears first.

HOW. A channel autoencoder (O'Shea & Hoydis): an ENCODER maps one of M = 2^k
messages to n complex samples at unit average power; a CHANNEL applies the
receiver profile's impairments — a random carrier phase, a small frequency
offset, IQ imbalance, a DC offset, AWGN at the training SNR, and the ADC's
quantisation (8 bits for an RTL-SDR, 12 for a bladeRF, from the profile;
straight-through gradient); the DECODER — the detector's side — returns the
message and a PRESENCE score (beacon or noise alone). Encoder and decoder
are trained together, so the waveform is shaped for this decoder through
this receiver.

THE BASELINE, BESIDE IT (plan §7): BPSK with a pilot symbol and the k bits
repetition-coded over the rest of the block, decoded by non-coherent maximum
likelihood over its codebook (the right detector for an unknown phase), and
detected by the same statistic against noise — through the SAME channel
model. `evaluate` reports block error rate (BLER) and detection probability
at a fixed false-alarm rate against Es/N0 for both — the learned waveform's
detection twice: by the decoder's own presence head, and by the same
classical matched statistic over its learned codebook. A learned waveform
that does not beat the baseline does not earn a place; on the first
synthetic run (k = 4, n = 8, an 8-bit receiver) it decoded better than BPSK
and was detected slightly worse — a result to read, not a verdict.

LIMITS. A simulated channel is the study's whole world: no multipath, no
interference, an impairment model of stated parameters (measured ones from
the receiver profile where present). The card is per receiver profile, like
every RF model here — the waveform is learned for one receiver's front end.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion import profiles as _profiles
from atk_diffusion import provenance

provenance.METHOD_TIERS.setdefault("beacon_study", "measured")

KIND = "beacon"
WEIGHTS = "beacon_ae.pt"
NOT_A_TRANSMITTER = ("DESIGN STUDY — nothing here transmits. Transmitting any "
                     "beacon would need proper authorisation.")


@dataclass
class Impairments:
    """The receiver's impairments as the study's channel applies them."""
    random_phase: bool = True
    cfo_max: float = 0.002          # cycles per sample, uniform ±
    iq_gain_db: float = 0.5         # I/Q amplitude imbalance
    iq_phase_deg: float = 2.0       # I/Q phase imbalance
    dc: float = 0.01                # DC offset, relative to unit power
    adc_bits: int = 8               # 0 = no quantisation
    full_scale: float = 4.0         # ADC full scale in units of the noise-free RMS

    @classmethod
    def for_profile(cls, profile) -> "Impairments":
        """From a ReceiverProfile (measured impairments where present) or a
        profile id (the family's ADC depth, stated defaults otherwise)."""
        if isinstance(profile, str):
            prof = _profiles.new_profile(profile)
        else:
            prof = profile
        imp = cls(adc_bits=int(prof.adc_bits or 0))
        m = prof.impairments or {}
        for key, attr in (("iq_gain_db", "iq_gain_db"),
                          ("iq_imbalance_db", "iq_gain_db"),
                          ("iq_phase_deg", "iq_phase_deg"),
                          ("dc_offset", "dc"), ("dc", "dc")):
            if isinstance(m.get(key), (int, float)):
                setattr(imp, attr, float(m[key]))
        return imp


# ---------------------------------------------------------------------------
# The channel, in numpy (the baseline) — the torch version mirrors it
# ---------------------------------------------------------------------------
def channel(x: np.ndarray, snr_db: float, imp: Impairments, rng,
            signal: bool = True) -> np.ndarray:
    """x (B, n) complex at unit average power -> what the receiver hands the
    detector. `signal=False` sends noise alone (for the false-alarm side)."""
    x = np.asarray(x, dtype=np.complex128)
    B, n = x.shape
    y = x.copy() if signal else np.zeros_like(x)
    if signal and imp.random_phase:
        y *= np.exp(1j * rng.uniform(0, 2 * np.pi, (B, 1)))
    if signal and imp.cfo_max:
        eps = rng.uniform(-imp.cfo_max, imp.cfo_max, (B, 1))
        y *= np.exp(2j * np.pi * eps * np.arange(n)[None, :])
    sigma = math.sqrt(10 ** (-float(snr_db) / 10.0) / 2.0)
    y = y + sigma * (rng.standard_normal((B, n)) + 1j * rng.standard_normal((B, n)))
    g = 10 ** (imp.iq_gain_db / 20.0)
    ph = math.radians(imp.iq_phase_deg)
    i, q = y.real, y.imag
    y = i + 1j * g * (q * math.cos(ph) + i * math.sin(ph))
    y = y + imp.dc * (1 + 1j)
    if imp.adc_bits:
        levels = 2 ** (imp.adc_bits - 1)
        fs = imp.full_scale
        quant = lambda v: np.clip(np.round(v / fs * levels), -levels, levels - 1) * fs / levels
        y = quant(y.real) + 1j * quant(y.imag)
    return y


def bpsk_codebook(k: int, n: int) -> np.ndarray:
    """The baseline: a +1 pilot, then the k bits as BPSK, repeated in turn
    to fill the other n − 1 samples (every sample carries a bit; unit
    power). (M, n) complex."""
    if n < k + 1:
        raise ValueError("the block needs at least k + 1 samples (a pilot)")
    M = 2 ** k
    cb = np.ones((M, n), dtype=np.complex128)
    for m in range(M):
        bits = np.array([(m >> b) & 1 for b in range(k)], dtype=float)
        cb[m, 1:] = 1 - 2 * bits[np.arange(n - 1) % k]
    return cb


def noncoherent_ml(y: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """argmax_m |<y, c_m>| — maximum likelihood for an unknown carrier phase
    over equal-energy codewords. -> (decisions, the winning statistic)."""
    corr = np.abs(y @ np.conj(codebook).T)
    return np.argmax(corr, axis=1), corr.max(axis=1)


# ---------------------------------------------------------------------------
# The autoencoder (PyTorch, inside functions only)
# ---------------------------------------------------------------------------
def _nets(M: int, n: int, hidden: int):
    import torch
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(M, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 2 * n))

        def forward(self, onehot):
            v = self.f(onehot)
            v = v / torch.sqrt(torch.mean(v * v, dim=1, keepdim=True) * 2.0 + 1e-12)
            return v                     # [re(n) | im(n)], unit average power

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(2 * n, 2 * hidden), nn.ReLU(),
                                   nn.Linear(2 * hidden, hidden), nn.ReLU())
            self.msg = nn.Linear(hidden, M)
            self.present = nn.Linear(hidden, 1)

        def forward(self, y):
            h = self.f(y)
            return self.msg(h), self.present(h).squeeze(-1)
    return Encoder(), Decoder()


def _torch_channel(v, snr_db, imp: Impairments, gen, signal: bool = True):
    """The same channel as `channel`, differentiable (quantisation passes
    the gradient straight through)."""
    import torch
    B, two_n = v.shape
    n = two_n // 2
    re, im = v[:, :n], v[:, n:]
    if not signal:
        re, im = torch.zeros_like(re), torch.zeros_like(im)
    else:
        th = torch.zeros(B, 1)
        if imp.random_phase:
            th = th + torch.rand(B, 1, generator=gen) * 2 * math.pi
        if imp.cfo_max:
            eps = (torch.rand(B, 1, generator=gen) * 2 - 1) * imp.cfo_max
            th = th + 2 * math.pi * eps * torch.arange(n).float()[None, :]
        c, s = torch.cos(th), torch.sin(th)
        re, im = re * c - im * s, re * s + im * c
    if isinstance(snr_db, (int, float)):
        snr = torch.full((B, 1), float(snr_db))
    else:
        snr = snr_db.reshape(B, 1)
    sigma = torch.sqrt(10 ** (-snr / 10.0) / 2.0)
    re = re + sigma * torch.randn(B, n, generator=gen)
    im = im + sigma * torch.randn(B, n, generator=gen)
    g = 10 ** (imp.iq_gain_db / 20.0)
    ph = math.radians(imp.iq_phase_deg)
    re, im = re, g * (im * math.cos(ph) + re * math.sin(ph))
    re, im = re + imp.dc, im + imp.dc
    if imp.adc_bits:
        levels = 2 ** (imp.adc_bits - 1)
        fs = imp.full_scale

        def q(t):
            qt = torch.clamp(torch.round(t / fs * levels), -levels, levels - 1) * fs / levels
            return t + (qt - t).detach()
        re, im = q(re), q(im)
    return torch.cat([re, im], dim=1)


@dataclass
class BeaconModel:
    card: object
    encoder: object
    decoder: object
    imp: Impairments
    k: int
    n: int

    def waveform(self, m: int) -> np.ndarray:
        """The learned n-sample waveform for message m (unit average power) —
        for study and plotting; NOT for transmission (see the module doc)."""
        import torch
        M = 2 ** self.k
        with torch.no_grad():
            v = self.encoder(torch.eye(M)[[int(m)]]).numpy()[0]
        return (v[:self.n] + 1j * v[self.n:]).astype(np.complex64)

    def codebook(self) -> np.ndarray:
        return np.stack([self.waveform(m) for m in range(2 ** self.k)])

    def decode(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(messages, presence score) for received blocks y (B, n)."""
        import torch
        Y = np.concatenate([y.real, y.imag], axis=1).astype(np.float32)
        with torch.no_grad():
            logits, pres = self.decoder(torch.from_numpy(Y))
        return logits.argmax(dim=1).numpy(), pres.numpy()


def train(profile: str, out_dir, *, k: int = 4, n: int = 8, hidden: int = 64,
          steps: int = 1500, batch: int = 256, lr: float = 3e-3,
          snr_train_db=(-2.0, 10.0), seed: int = 0,
          imp: Impairments | None = None,
          progress: Callable[[str], None] | None = None) -> BeaconModel:
    """Train encoder and decoder through the profile's channel. Saves the
    weights and a card (kind "beacon", per profile) in `out_dir`."""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    pid = _profiles.parse_profile_id(profile)
    imp = imp or Impairments.for_profile(str(profile))
    M = 2 ** int(k)
    enc, dec = _nets(M, int(n), int(hidden))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    eye = torch.eye(M)
    lo, hi = snr_train_db
    last = 0.0
    for step in range(int(steps)):
        msg = torch.randint(0, M, (batch,), generator=gen)
        snr = lo + (hi - lo) * torch.rand(batch, generator=gen)
        y1 = _torch_channel(enc(eye[msg]), snr, imp, gen, signal=True)
        y0 = _torch_channel(torch.zeros(batch, 2 * n), snr, imp, gen, signal=False)
        l1, p1 = dec(y1)
        _l0, p0 = dec(y0)
        loss = ce(l1, msg) + 0.5 * (bce(p1, torch.ones(batch))
                                    + bce(p0, torch.zeros(batch)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
        if progress and (step + 1) % 250 == 0:
            progress(f"step {step + 1}: loss {last:.3f}")
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict()},
               d / WEIGHTS)
    card = _cards.new_card(
        f"beacon-k{k}-n{n}", KIND, str(profile).lower(),
        input={"k": int(k), "n": int(n), "hidden": int(hidden),
               "impairments": asdict(imp), "snr_train_db": list(snr_train_db)},
        metrics={"final_loss": last}, license="all rights reserved",
        trained_on=f"simulated channel of {_profiles.describe(profile)}",
        notes=[NOT_A_TRANSMITTER, "research track (plan §4.R)"])
    _cards.save(d, card, WEIGHTS)
    enc.eval()
    dec.eval()
    return BeaconModel(card, enc, dec, imp, int(k), int(n))


def load(model_dir, for_profile: str | None = None) -> BeaconModel:
    import torch
    card = _cards.load(model_dir, expect_kind=KIND, for_profile=for_profile)
    k, n, hidden = (int(card.input[x]) for x in ("k", "n", "hidden"))
    enc, dec = _nets(2 ** k, n, hidden)
    state = torch.load(Path(model_dir) / card.weights["file"], map_location="cpu",
                       weights_only=True)
    enc.load_state_dict(state["encoder"])
    dec.load_state_dict(state["decoder"])
    enc.eval()
    dec.eval()
    return BeaconModel(card, enc, dec, Impairments(**card.input["impairments"]), k, n)


def evaluate(model: BeaconModel, snrs_db=(-2.0, 0.0, 2.0, 4.0, 6.0, 8.0), *,
             blocks: int = 4000, pfa: float = 0.01, seed: int = 1) -> dict:
    """BLER and detection probability (at `pfa`, the threshold set on noise
    alone) against Es/N0, learned waveform versus the BPSK baseline, through
    the same channel. -> {snr: {...}}, plus the label."""
    rng = np.random.default_rng(int(seed))
    imp, k, n = model.imp, model.k, model.n
    M = 2 ** k
    cb_bpsk = bpsk_codebook(k, n)
    cb_ae = model.codebook().astype(np.complex128)
    rows = {}
    for snr in snrs_db:
        msg = rng.integers(0, M, blocks)
        y_ae = channel(cb_ae[msg], snr, imp, rng)
        y_bp = channel(cb_bpsk[msg] / np.sqrt(np.mean(np.abs(cb_bpsk) ** 2)),
                       snr, imp, rng)
        n_ae = channel(cb_ae[msg], snr, imp, rng, signal=False)
        n_bp = channel(cb_bpsk[msg], snr, imp, rng, signal=False)
        d_ae, s_ae = model.decode(y_ae)
        _d0, s0_ae = model.decode(n_ae)
        _dm, sm_ae = noncoherent_ml(y_ae, cb_ae)        # the learned codebook,
        _dn, sm0_ae = noncoherent_ml(n_ae, cb_ae)       # matched classically
        d_bp, s_bp = noncoherent_ml(y_bp, cb_bpsk)
        _d1, s0_bp = noncoherent_ml(n_bp, cb_bpsk)
        thr_ae = float(np.quantile(s0_ae, 1 - pfa))
        thr_am = float(np.quantile(sm0_ae, 1 - pfa))
        thr_bp = float(np.quantile(s0_bp, 1 - pfa))
        rows[float(snr)] = {
            "ae_bler": float(np.mean(d_ae != msg)),
            "bpsk_bler": float(np.mean(d_bp != msg)),
            "ae_pd": float(np.mean(s_ae > thr_ae)),
            "ae_pd_matched": float(np.mean(sm_ae > thr_am)),
            "bpsk_pd": float(np.mean(s_bp > thr_bp))}
    return {"rows": rows, "pfa": pfa, "blocks": blocks, "k": k, "n": n,
            "label": NOT_A_TRANSMITTER, "tier": "measured",
            "what": ("simulated BLER and detection probability versus Es/N0 "
                     "(per complex sample) through the receiver profile's "
                     "impairment model; BPSK baseline: pilot + repetition, "
                     "non-coherent ML")}


def report_md(ev: dict) -> str:
    lines = ["# Beacon co-designed with its detector — design study (plan §4.R)",
             "", f"**{ev['label']}**", "", f"*{ev['what']}.*", "",
             f"k = {ev['k']} bits in n = {ev['n']} samples; detection at a "
             f"false-alarm rate of {ev['pfa']:g}.", "",
             "| Es/N0 dB | learned BLER | BPSK BLER | learned Pd (its head) | "
             "learned Pd (matched) | BPSK Pd (matched) |",
             "|---|---|---|---|---|---|"]
    for snr, r in sorted(ev["rows"].items()):
        lines.append(f"| {snr:g} | {r['ae_bler']:.4f} | {r['bpsk_bler']:.4f} | "
                     f"{r['ae_pd']:.3f} | {r['ae_pd_matched']:.3f} | "
                     f"{r['bpsk_pd']:.3f} |")
    return "\n".join(lines) + "\n"
