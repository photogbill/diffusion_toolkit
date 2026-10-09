# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The confirmer: only a decoder turns Proposed into Confirmed
(DETECTION_DESIGN §5; plan §2.1, §4.B1; ARCHITECTURE §4.1).

*"A track whose class has a decoder is handed to it — DSD for P25/DMR/NXDN,
the pager decoder for POCSAG/FLEX, multimon for the rest it knows, the ADS-B
path — and a decode upgrades the track to Confirmed, with the decoded content
attached. Disagreement (classifier says DMR, DSD decodes P25) is shown as
disagreement, and the decoder wins the label."*

THE TOOLKIT OWNS THE RULES; ATK OWNS THE DECODERS. The registry maps a
decoder key of the class table (`classes.DECODERS`) to a callable the host
supplies:

    registry["dsd"] = lambda det, iq, fs: ConfirmResult(ok=True, decoded="TG 1234",
                                                         decoder_class="p25")

and `Confirmer(registry).confirm(det, iq, fs)` applies the rules:

1. Only decoders whose `confirms` is True can confirm. A demodulator (NFM,
   WFM, AM, SSB) makes audio and confirms nothing; it is never called here
   even when registered.
2. The class's decoders are tried in `classes.tools_for(cls)` order (best
   first); the first that decodes wins and the rest are not run.
3. The decoder wins the label: `Detection.confirm(decoder, decoded,
   decoder_class)` — a classifier class it contradicts is kept as
   `measurements["classifier_said"]` and the `disagreement` flag, never
   erased.
4. A decoder that raises is caught and reported in words; the next one is
   tried. Nothing a decoder does can take the detector down.
5. A decode from an input that is not the record (a cleaned, inferred or
   invented cut) still confirms — a decoder decoded it, which is what
   Confirmed means — but says so wherever the decode is shown:
   `provenance.decoded_from_note(tier)` rides on the result, in the
   detection's decoded line and in its measurements (provenance's rule:
   *"A decode from anything but the record says so, whatever the tier."*).

Every attempt is a sentence in `ConfirmResult.attempts`, so the AI Detect tab
can show exactly what was tried and why it did or did not confirm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from atk_diffusion import provenance as _prov
from atk_diffusion.detect import classes as _classes
from atk_diffusion.detect.boxes import Detection

DecoderFn = Callable[[Detection, object, float], "ConfirmResult | None"]


@dataclass
class ConfirmResult:
    """What a decoder (or the confirmer) says.

    ok             the decoder decoded it
    decoder        its key in classes.DECODERS (the confirmer fills it in)
    decoded        a short line of content (talkgroup, pager text, ICAO …)
    decoder_class  the class the decoder says it is ('' = it does not say)
    details        anything else the host wants kept (dict)
    why            when not ok: the reason, in words
    note           the not-the-record warning, when the input was not
    attempts       (the confirmer's result) one sentence per decoder tried
    tier           the tier of the input the decode came from
    seconds        time spent decoding
    """
    ok: bool = False
    decoder: str = ""
    decoded: str = ""
    decoder_class: str = ""
    details: dict = field(default_factory=dict)
    why: str = ""
    note: str = ""
    attempts: list = field(default_factory=list)
    tier: str = "record"
    seconds: float = 0.0


class Confirmer:
    """Applies the confirmation rules over host-supplied decoders."""

    def __init__(self, registry: dict | None = None):
        reg = dict(registry or {})
        for key, fn in reg.items():
            if key not in _classes.DECODERS:
                raise ValueError(f"{key!r} is not a decoder in the class table "
                                 f"(known: {', '.join(sorted(_classes.DECODERS))})")
            if not callable(fn):
                raise ValueError(f"the decoder registered as {key!r} is not "
                                 "callable")
        self.registry = reg

    # -- what can be done ------------------------------------------------------
    @staticmethod
    def confirming_decoders(cls: str) -> list[str]:
        """Decoders that can confirm `cls`, best first (the class table)."""
        return [k for k in _classes.tools_for(cls)
                if _classes.DECODERS.get(k, {}).get("confirms")]

    def available(self, cls: str) -> list[str]:
        return [k for k in self.confirming_decoders(cls) if k in self.registry]

    def describe(self, cls: str) -> str:
        """'DMR can be confirmed by dsd-neo (P25 / DMR / NXDN) — available.'"""
        c = _classes.get(cls)
        name = c.label if c else (cls or "an unclassified signal")
        keys = self.confirming_decoders(cls)
        if not keys:
            return (f"{name} has no decoder that can confirm it; it stays "
                    "PROPOSED with its class and confidence")
        parts = []
        for k in keys:
            label = _classes.DECODERS[k]["label"]
            parts.append(f"{label} — {'available' if k in self.registry else 'not registered by ATK'}")
        return f"{name} can be confirmed by " + "; ".join(parts)

    # -- the rules ---------------------------------------------------------------
    def confirm(self, det: Detection, iq, fs: float, tier: str = "record",
                track=None) -> ConfirmResult:
        """Try the class's confirming decoders on `iq` (the detection's cut
        at `fs`); upgrade `det` (and `track`, a tracker.Track) on the first
        decode. Returns the confirmer's result with every attempt in words."""
        tier = _prov.check_tier(tier)
        note = _prov.decoded_from_note(tier)
        cls = det.cls or ""
        res = ConfirmResult(tier=tier, note=note)
        if not cls or cls == _classes.UNKNOWN:
            res.why = ("the detection has no class, so no decoder is tried; "
                       "it stays PROPOSED" if not cls else
                       "the detection is UNKNOWN, so no decoder is tried; it "
                       "stays PROPOSED")
            res.attempts.append(res.why)
            return res
        keys = self.confirming_decoders(cls)
        if not keys:
            res.why = self.describe(cls)
            res.attempts.append(res.why)
            return res
        t_start = time.perf_counter()
        for key in keys:
            label = _classes.DECODERS[key]["label"]
            fn = self.registry.get(key)
            if fn is None:
                res.attempts.append(f"{key}: {label} is not registered by ATK "
                                    "on this machine")
                continue
            try:
                out = fn(det, iq, fs)
            except Exception as e:                       # noqa: BLE001
                res.attempts.append(f"{key}: {label} failed — "
                                    f"{type(e).__name__}: {e}")
                continue
            if out is None or not getattr(out, "ok", False):
                why = (getattr(out, "why", "") or "decoded nothing") \
                    if out is not None else "decoded nothing"
                res.attempts.append(f"{key}: {label} tried — {why}")
                continue
            # a decode: the decoder wins the label
            before = det.cls
            decoded = str(out.decoded or "")
            if note:
                decoded = f"{decoded} [{note}]" if decoded else f"[{note}]"
            det.confirm(key, decoded, decoder_class=str(out.decoder_class or ""))
            det.measurements = dict(det.measurements)
            det.measurements["confirmed_from_tier"] = tier
            if note:
                det.measurements["decode_note"] = note
            res.ok = True
            res.decoder = key
            res.decoded = str(out.decoded or "")
            res.decoder_class = str(out.decoder_class or "")
            res.details = dict(getattr(out, "details", {}) or {})
            disagree = bool(res.decoder_class and before
                            and res.decoder_class != before)
            res.attempts.append(
                f"{key}: {label} decoded it"
                + (f" as {res.decoder_class}" if res.decoder_class else "")
                + (f" — the classifier said {before}; the decoder wins the "
                   "label and the disagreement is kept" if disagree else ""))
            if track is not None:
                track.state = "confirmed"
                track.confirmed_by = key
                track.decoded = det.decoded
                if det.cls:
                    track.cls = det.cls
            break
        res.seconds = time.perf_counter() - t_start
        if not res.ok:
            res.why = "no decoder confirmed it: " + "; ".join(res.attempts)
        return res
