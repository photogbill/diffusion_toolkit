# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""D1's experiment — does a learned fill keep DMR decodable across a USB
dropout better than interpolation or AR? (plan §4.D1, §6 Phase 3, §7).

Phase 3's exit for D1: *"DSD decodes recovered across injected dropouts."*
Plan §7: *"every inpainter [is scored] against interpolation"* and every
reconstruction tool reports a hallucination rate.

THE SET-UP. DMR-like 4FSK TDMA traffic with KNOWN symbols
(`learn.inpaint.tdma_stream`: 4800 sym/s, the BS voice sync word in the
middle of each burst, one slot active) at the profile's voice-class
canonical rate, in white noise at a stated in-band SNR. A USB-style dropout
— a run of zeros where the buffer was lost, the timeline already restored
(that is `repair.iq_dropout`'s job) — is injected inside a burst. Each
method fills it:

  zeros      no repair: the dropout as it reached the decoder
  linear     straight-line interpolation (`repair.iq_dropout`, or the local
             fallback when that module is absent)
  ar         forward–backward Burg AR (`repair.iq_dropout`, when installed)
  janssen    Janssen–Veldhuis–Vries (`repair.iq_dropout`, when installed)
  diffusion  the D1 inpainter (`learn.inpaint`), when a model is given

THE NUMBERS, per gap length and method:
* symbol error rate from a minimal 4FSK channel filter + discriminator +
  slicer with DSD-style level tracking (`learn.inpaint.fsk4_demod`) —
  INSIDE the gap (random payload: every method should sit near chance,
  0.75; anything better there is the fixed sync pattern or luck), AFTER
  the gap (the decoder's tracking disturbed — the plan's "breaks decoder
  sync"), and over the whole burst; the fraction of bursts whose 24-symbol
  sync word is still found (≤ 3 symbol errors), overall and among the
  trials whose gap hit the sync word;
* the waveform SNR inside the gap against the CLEAN signal;
* the HALLUCINATION RATE: gaps placed in silence (between bursts, noise
  only) whose fill carries more power than noise alone does 99 % of the
  time — filling silence with signal.

LIMITS. The modem is not DSD (no FEC, no CACH, known symbol timing); the
noise is white; the dropout is zeros with the timeline intact. A real run
points `inpaint_eval` at a trained model and, for the real proof, the
repaired IQ goes through DSD itself (ATK's side).
"""

from __future__ import annotations

import math
import time

import numpy as np

from atk_diffusion import profiles, provenance
from atk_diffusion.learn import inpaint as _ip

SYNC_TOLERANCE = 3          # symbol errors a sync search still accepts


def _classical():
    """{method: fill(x, start, count, fs) -> y} from repair.iq_dropout, or
    the local straight line when that module is absent."""
    try:
        from atk_diffusion.repair import iq_dropout as _iqd

        def make(m):
            def f(x, s, c, fs):
                y, used = _iqd.fill_spans(x, [(s, c)], method=m, fs=fs)
                return np.asarray(y, dtype=np.complex64), used[0]
            return f
        return ({"linear": make("linear"), "ar": make("ar"), "janssen": make("janssen")},
                "repair.iq_dropout")
    except ImportError:
        def lin(x, s, c, fs):
            y = np.array(x, dtype=np.complex64, copy=True)
            a, b = complex(y[s - 1]) if s > 0 else 0j, complex(y[s + c]) if s + c < y.size else 0j
            f = np.arange(1, c + 1) / (c + 1)
            y[s:s + c] = a + (b - a) * f
            return y, "interpolate"
        return {"linear": lin}, "local straight-line fallback (repair.iq_dropout not installed)"


def _load(inpainter, pid):
    if inpainter is None:
        return None
    if isinstance(inpainter, _ip.Inpainter):
        profiles.check_match(inpainter.card.profile, pid,
                             what=f"this inpainter ({inpainter.card.name})")
        return inpainter
    return _ip.Inpainter.load(inpainter, for_profile=pid)


def run(rf=None, profile: str = "rtlsdr_2400000_cu8", *, inpainter=None,
        canonical: str = "voice", trials: int = 40, gaps_ms=(0.5, 1.0, 2.0),
        snr_db: float = 20.0, silence_trials: int = 40, steps: int | None = None,
        resample: int | None = None, seed: int = 0, out_dir=None,
        name: str = "inpaint_eval", progress=None) -> dict:
    """The D1 experiment. Returns the result dict (and writes the report when
    `rf` or `out_dir` is given)."""
    from atk_diffusion.experiments.weak_burst import write_report
    say = progress or (lambda s: None)
    t_start = time.time()
    pid = str(profile).lower()
    prof = profiles.load_profile(rf, pid) if rf is not None else profiles.new_profile(pid)
    inp = _load(inpainter, pid)
    if inp is not None:
        fs = inp.rate
    else:
        fs, _dec, _cls = _ip._canonical(prof, canonical, None)
    sps = fs / _ip.BAUD
    if abs(sps - round(sps)) > 1e-9:
        raise ValueError(f"the DMR-like test traffic needs a whole number of samples "
                         f"per symbol; {fs:g} S/s gives {sps:.3f}")
    sps = int(round(sps))
    classical, classical_src = _classical()
    methods = ["zeros", *classical] + (["diffusion"] if inp is not None else [])
    rng = np.random.default_rng(seed)
    nb = _ip.BURST_SYMBOLS
    sync_lo, sync_hi = _ip.SYNC_AT, _ip.SYNC_AT + 24
    p_n = 1.0
    a = math.sqrt(10 ** (snr_db / 10.0) * p_n * _ip.NOMINAL_BW_HZ / fs)
    per_gap = {}
    reference = []
    for gap_ms in gaps_ms:
        g = max(1, int(round(gap_ms * 1e-3 * fs)))
        gsym = int(math.ceil(g / sps)) + 1
        acc = {m: {"ser_gap": [], "ser_post": [], "ser_burst": [], "sync_ok": [],
                   "sync_ok_hit": [], "gap_snr_db": [], "fill_to_context_db": [],
                   "used": set()} for m in methods}
        for _ in range(int(trials)):
            clean, bursts = _ip.tdma_stream(2, fs, rng)
            clean = a * clean
            y, _ = _ip.add_noise(clean, 0.0, rng, fs=fs, p_n=p_n)
            b = bursts[1]
            lo_sym, hi_sym = 8, nb - gsym - 24
            s_sym = int(rng.integers(lo_sym, max(lo_sym + 1, hi_sym)))
            s = b["start"] + s_sym * sps + int(rng.integers(0, sps))
            k0 = (s - b["start"]) // sps                    # first symbol touched
            k1 = min(nb, (s + g - 1 - b["start"]) // sps + 1)   # one past the last
            hit_sync = k0 < sync_hi and k1 > sync_lo
            ref_sym, _v = _ip.fsk4_demod(y, fs, b["start"], nb)
            reference.append(float(np.mean(ref_sym != b["symbols"])))
            dropped = y.copy()
            dropped[s:s + g] = 0
            mask = np.zeros(y.size, dtype=bool)
            mask[s:s + g] = True
            for m in methods:
                if m == "zeros":
                    rep, used = dropped, "none"
                elif m == "diffusion":
                    rep, info = inp.inpaint(dropped, mask, noise_power=p_n, steps=steps,
                                            resample=resample, seed=int(rng.integers(0, 1 << 30)))
                    used = info["method"]
                else:
                    rep, used = classical[m](dropped, s, g, fs)
                acc[m]["used"].add(used)
                sym, _v = _ip.fsk4_demod(rep, fs, b["start"], nb)
                err = sym != b["symbols"]
                acc[m]["ser_gap"].append(float(np.mean(err[k0:k1])))
                post = err[k1:k1 + 48]
                if post.size:
                    acc[m]["ser_post"].append(float(np.mean(post)))
                acc[m]["ser_burst"].append(float(np.mean(err)))
                ok = int(np.sum(err[sync_lo:sync_hi])) <= SYNC_TOLERANCE
                acc[m]["sync_ok"].append(ok)
                if hit_sync:
                    acc[m]["sync_ok_hit"].append(ok)
                truth = clean[s:s + g]
                e = float(np.sum(np.abs(truth) ** 2))
                acc[m]["gap_snr_db"].append(10 * math.log10(
                    max(e, 1e-30) / max(float(np.sum(np.abs(rep[s:s + g] - truth) ** 2)), 1e-30)))
                ctx = np.concatenate([rep[max(0, s - 256):s], rep[s + g:s + g + 256]])
                pf = float(np.mean(np.abs(rep[s:s + g]) ** 2))
                pc = float(np.mean(np.abs(ctx) ** 2))
                if pc > 0 and pf > 0:
                    acc[m]["fill_to_context_db"].append(10 * math.log10(pf / pc))
        # silence: the hallucination test (learn.inpaint.HALLUCINATION_RULE)
        hall = {m: {"eligible": 0, "hallucinated": 0, "excess_db": []}
                for m in methods if m != "zeros"}
        for _ in range(int(silence_trials)):
            clean, bursts = _ip.tdma_stream(2, fs, rng)
            y, _ = _ip.add_noise(a * clean, 0.0, rng, fs=fs, p_n=p_n)
            quiet0 = bursts[0]["end"] + 64                 # after burst 0 is off the air
            quiet1 = bursts[1]["start"] - _ip.GUARD_SYMBOLS * sps - 64 - g   # before burst 1
            if quiet1 <= quiet0:
                continue
            s = int(rng.integers(quiet0, quiet1))
            dropped = y.copy()
            dropped[s:s + g] = 0
            mask = np.zeros(y.size, dtype=bool)
            mask[s:s + g] = True
            for m in hall:
                if m == "diffusion":
                    rep, _info = inp.inpaint(dropped, mask, noise_power=p_n, steps=steps,
                                             resample=resample, seed=int(rng.integers(0, 1 << 30)))
                else:
                    rep, _u = classical[m](dropped, s, g, fs)
                hall[m]["eligible"] += 1
                flag, ex = _ip.hallucinated(rep[s:s + g], y[s:s + g])
                hall[m]["hallucinated"] += int(flag)
                hall[m]["excess_db"].append(ex)

        def mean(v):
            return float(np.mean(v)) if len(v) else None
        per_gap[f"{gap_ms:g}"] = {
            "gap_samples": g, "gap_symbols": round(g / sps, 2),
            "methods": {m: {"ser_gap": mean(acc[m]["ser_gap"]),
                            "ser_post": mean(acc[m]["ser_post"]),
                            "ser_burst": mean(acc[m]["ser_burst"]),
                            "sync_found": mean(acc[m]["sync_ok"]),
                            "sync_found_when_hit": mean(acc[m]["sync_ok_hit"]),
                            "sync_hit_trials": len(acc[m]["sync_ok_hit"]),
                            "gap_snr_db": mean(acc[m]["gap_snr_db"]),
                            "fill_to_context_db": (float(np.median(acc[m]["fill_to_context_db"]))
                                                   if acc[m]["fill_to_context_db"] else None),
                            "used": sorted(acc[m]["used"]),
                            "hallucination": ({"eligible": hall[m]["eligible"],
                                               "hallucinated": hall[m]["hallucinated"],
                                               "rate": (hall[m]["hallucinated"] / hall[m]["eligible"])
                                               if hall[m]["eligible"] else None,
                                               "mean_excess_db": mean(hall[m]["excess_db"])}
                                              if m in hall else None)}
                        for m in methods}}
        say(f"gap {gap_ms:g} ms: " + ", ".join(
            f"{m} post-gap SER {per_gap[f'{gap_ms:g}']['methods'][m]['ser_post']}"
            for m in methods))
    tiers = {"zeros": "record", "linear": provenance.tier_for("interpolate"),
             "ar": provenance.tier_for("ar_fill"),
             "janssen": provenance.METHOD_TIERS.get("janssen_fill", "inferred"),
             "diffusion": provenance.tier_for("diffusion_inpaint")}
    result = {"name": name, "profile": pid, "profile_words": profiles.describe(pid),
              "fs": fs, "snr_db_in_band": float(snr_db), "trials": int(trials),
              "silence_trials": int(silence_trials), "methods": methods,
              "method_tiers": {m: tiers[m] for m in methods},
              "classical_from": classical_src,
              "inpainter": (None if inp is None else
                            {"name": inp.card.name, "sha256": inp.card.weights.get("sha256"),
                             "backend": inp.backend,
                             "card_hallucination_rate": (inp.card.metrics or {}).get("hallucination_rate")}),
              "reference_ser_no_dropout": float(np.mean(reference)) if reference else None,
              "hallucination_rule": _ip.HALLUCINATION_RULE,
              "gaps": per_gap, "seconds": round(time.time() - t_start, 1),
              "modem": "learn.inpaint.fsk4_demod — channel filter, discriminator, "
                       "DSD-style level tracking, known timing; not DSD"}
    result["verdict"] = _verdict(result)
    target = out_dir
    if target is None and rf is not None:
        from atk_diffusion.experiments.weak_burst import report_dir
        target = report_dir(rf, pid, name)
    if target is not None:
        summary = [f"D1 on {result['profile_words']}: DMR-like 4FSK at {fs:g} S/s, "
                   f"{snr_db:g} dB in-band SNR, USB-style dropouts (zeros, timeline "
                   f"intact); classical fills from {classical_src}.",
                   f"Symbol errors with no dropout at all: "
                   f"{_f(result['reference_ser_no_dropout'], True)} (the floor).",
                   *result["verdict"],
                   f"Hallucination: {_ip.HALLUCINATION_RULE}.",
                   "A learned fill is INVENTED tier. Tables: the _detail.md beside this."]
        md, js, det = write_report(target, name, result, summary,
                                   detail=report_lines(result), rf=rf)
        result["report_md"], result["report_json"] = str(md), str(js)
        result["report_detail_md"] = str(det)
    return result


def _verdict(result: dict) -> list[str]:
    """One sentence per gap length: what each method left the decoder."""
    out = []
    for gk, gv in result["gaps"].items():
        ms = gv["methods"]
        parts = [f"{m} {_f(r['ser_post'], True)}" for m, r in ms.items()]
        out.append(f"{gk} ms dropout — symbol errors after the gap: " + ", ".join(parts) + ".")
        if "diffusion" in ms:
            d = ms["diffusion"]
            best = min((r["ser_post"] for m, r in ms.items()
                        if m not in ("diffusion", "zeros") and r["ser_post"] is not None),
                       default=None)
            h = (d["hallucination"] or {}).get("rate")
            if best is not None and d["ser_post"] is not None:
                word = "beats" if d["ser_post"] < best else "does NOT beat"
                out.append(f"{gk} ms: the learned fill {word} the best classical fill "
                           f"after the gap ({_f(d['ser_post'], True)} vs {_f(best, True)})"
                           + (f"; it filled {h:.0%} of silent gaps with signal" if h is not None
                              else "") + ".")
    return out


def _f(v, pct=False):
    if v is None:
        return "—"
    return f"{100 * v:.1f} %" if pct else f"{v:.1f}"


def report_lines(result: dict) -> list[str]:
    L = [f"# D1 — IQ dropout repair before the decoder — {result['profile_words']}", "",
         f"- DMR-like 4FSK at {result['fs']:g} S/s, {result['snr_db_in_band']:g} dB in-band "
         f"SNR; {result['trials']} dropouts per gap length inside bursts, "
         f"{result['silence_trials']} in silence",
         f"- classical fills from {result['classical_from']}",
         f"- symbol errors with no dropout at all: {_f(result['reference_ser_no_dropout'], True)} "
         "(the floor)",
         f"- hallucination: {result['hallucination_rule']}",
         f"- modem: {result['modem']}",
         "- tiers: " + ", ".join(f"{m} {t}" for m, t in result["method_tiers"].items())]
    if result["inpainter"]:
        i = result["inpainter"]
        L.append(f"- inpainter: {i['name']} ({i['backend']}), sha256 {str(i['sha256'])[:12]}…")
    L += ["", "**A learned fill is INVENTED tier** — " + provenance.TIER_WORDS["invented"], ""]
    for gk, gv in result["gaps"].items():
        L += [f"## {gk} ms dropout ({gv['gap_samples']} samples, {gv['gap_symbols']} symbols)", "",
              "| method | SER in gap | SER after gap | SER burst | sync found | sync found when hit | "
              "gap SNR (dB) | fill vs context (dB) | hallucination in silence |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m, r in gv["methods"].items():
            h = r["hallucination"]
            hs = "—" if h is None else (f"{h['rate']:.3f} ({h['hallucinated']}/{h['eligible']}; "
                                        f"fill {h['mean_excess_db']:+.1f} dB vs truth)"
                                        if h["rate"] is not None else "none tested")
            L.append(f"| {m} | {_f(r['ser_gap'], True)} | {_f(r['ser_post'], True)} | "
                     f"{_f(r['ser_burst'], True)} | {_f(r['sync_found'], True)} | "
                     f"{_f(r['sync_found_when_hit'], True)} (n={r['sync_hit_trials']}) | "
                     f"{_f(r['gap_snr_db'])} | {_f(r['fill_to_context_db'])} | {hs} |")
        L.append("")
    L += ["Random payload inside a gap is lost to every method (chance is 75 % "
          "symbol errors); what a fill can save is the decoder's tracking after "
          "the gap, and fixed structure such as the sync word. A decode across "
          "any fill is a decode of a guess (provenance.decoded_from_note)."]
    return L
