# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Tiers: what a piece of output IS, said on every output (plan §2.1).

*"Every learned output is a reconstruction or a proposal — never evidence."*
The raw capture stays the record; everything made from it carries a tier,
shown wherever it is shown, and traceable to what it was made from.

The vocabulary is ATK's own (the Forensics Workshop's tier system, built for
SAM Audio), extended by two words the detector needs:

    record     the capture as recorded — the evidence
    measured   a number computed from the record by a stated classical method
               (bandwidth, symbol rate, SNR) — checkable, not an opinion
    cleaned    a deterministic linear process applied to the record (matched
               filter, Wiener, FRESH, SCORE): nothing added that was not in it,
               but no longer the record
    inferred   a gap filled by interpolation or a statistical model — the
               samples are a best guess
    invented   made by a generative model (diffusion denoiser, inpainter,
               translator, speech enhancer): may contain structure that was
               never there. A lead, never a reading.
    proposed   a detection nobody has confirmed (a box marked Proposed)
    confirmed  a detection a decoder confirmed

DECISION, refining the plan's wording (§4.2/§4.3 said every `cleaned` file is
Invented tier). The rule underneath — the original is the record and every
cleaned file is labelled as not the record — is kept exactly. What changes is
that a Wiener filter is not called *invented*: it cannot add a signal, and an
analyst told that it might would learn to ignore the warning that matters,
the one on the diffusion denoiser. A decode from anything but the record says
so, whatever the tier.
"""

from __future__ import annotations

import getpass
import hashlib
import platform
import socket
import time
from pathlib import Path

TIERS = ("record", "measured", "cleaned", "inferred", "invented",
         "proposed", "confirmed")

TIER_WORDS = {
    "record": "RECORD — the capture as recorded.",
    "measured": "MEASURED — a classical measurement of the record.",
    "cleaned": "CLEANED — a linear process applied to the record; nothing "
               "added, but this is not the record.",
    "inferred": "INFERRED — a gap filled by interpolation or a statistical "
                "model; these samples are a best guess.",
    "invented": "INVENTED — made by a generative model. It can hold structure "
                "that was never received: a lead, never a reading.",
    "proposed": "PROPOSED — a detection no decoder has confirmed.",
    "confirmed": "CONFIRMED — a decoder decoded it.",
}

#: Tiers that are NOT the record: a decode or measurement made from one of
#: these must say so.
NOT_RECORD = ("cleaned", "inferred", "invented")

#: Method -> tier, for every reconstruction or cleaning method in the
#: toolkit. A method that is not listed has no tier and cannot write output;
#: `tier_for` raises rather than guess.
METHOD_TIERS = {
    # classical, linear: nothing added
    "matched_filter": "cleaned", "wiener": "cleaned", "fresh": "cleaned",
    "fresh_separate": "cleaned", "score": "cleaned", "bandpass": "cleaned",
    "median": "cleaned", "wavelet": "cleaned", "spectral_subtraction": "cleaned",
    "mmse_lsa": "cleaned", "deskew": "cleaned", "resample": "cleaned",
    # statistical fills
    "interpolate": "inferred", "rfi_mask_interp": "inferred",
    "ar_fill": "inferred", "kalman_fill": "inferred", "pri_fill": "inferred",
    "kriging": "inferred", "idw": "inferred", "lpc_fill": "inferred",
    "itm": "inferred", "free_space": "inferred", "two_ray": "inferred",
    # learned
    "diffusion_denoise": "invented", "diffusion_inpaint": "invented",
    "diffusion_translate": "invented", "diffusion_augment": "invented",
    "speech_enhance": "invented", "diffusion_radiomap": "invented",
    "diffusion_track": "invented", "learned_rfi_mask": "invented",
    "diffusion_posterior": "invented", "learned_position": "invented",
    "diffusion_text": "invented",
}


def tier_for(method: str) -> str:
    try:
        return METHOD_TIERS[str(method)]
    except KeyError:
        raise ValueError(f"no tier is declared for method {method!r}; "
                         "declare it in provenance.METHOD_TIERS before it "
                         "may write output") from None


def check_tier(tier: str) -> str:
    t = str(tier).strip().lower()
    if t not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return t


def is_record(tier: str) -> bool:
    return check_tier(tier) == "record"


def decoded_from_note(tier: str) -> str:
    """The line a decoder result carries when its input was not the record."""
    t = check_tier(tier)
    if t in NOT_RECORD:
        return (f"decoded from a {t.upper()} signal, not from the record — "
                "confirm against the original before relying on it")
    return ""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def who() -> str:
    try:
        return getpass.getuser()
    except Exception:                                      # noqa: BLE001
        return "unknown"


def stamp(tool: str, **fields) -> dict:
    """A provenance step: what ran, where, when, by whom, with what."""
    from atk_diffusion import __version__
    return {"tool": str(tool), "version": __version__,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "by": who(), "host": socket.gethostname(),
            "platform": platform.platform(terse=True), **fields}


def relpath(path, start) -> str:
    try:
        return Path(path).resolve().relative_to(Path(start).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()
