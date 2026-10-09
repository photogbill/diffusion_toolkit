# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Detections, merging, and the box palette (DETECTION_DESIGN §12.2, D13)."""

from __future__ import annotations

import pytest

from atk_diffusion.detect import boxes as B
from atk_diffusion.detect import classes as C

#: ATK's waterfall technology palette as of 2026-10-08
#: (atk/ui/spectrum_view.py PROTOCOL_COLOURS), plus the processing green and
#: the selection amber, which also live on the waterfall. ATK holds its own
#: copy of this check against the live table (tests/test_diffusion_boxes.py).
ATK_TECH = {"p25": "#ff5c8a", "dmr": "#ff9f40", "nxdn": "#56e0b0",
            "ysf": "#c48cff", "dstar": "#ff6b6b", "m17": "#ffffff",
            "tetra": "#b8e04a", "dpmr": "#8fa6c4", "provoice": "#e8a0d8",
            "pocsag": "#f0c674", "analog": "#9aa7b8", "lte": "#3fd0e0",
            "handset": "#ffd24a", "processing": "#5ad469",
            "selection": "#e6b34a"}
WF = [(8, 14, 40), (18, 46, 110), (30, 96, 170), (60, 150, 190),
      (140, 195, 150), (230, 220, 90), (255, 250, 190)]


def test_the_plans_cyan_collides_with_the_lte_cyan():
    """Why the cyclostationary colour moved (D13's own rule)."""
    cyan = B.rgb_lab(B.hex_rgb("#00e5ff"))
    lte = B.rgb_lab(B.hex_rgb(ATK_TECH["lte"]))
    assert B.delta_e(cyan, lte) < 7


def test_source_colours_stay_clear_of_the_technology_palette_and_the_ramp():
    rep = B.palette_report(ATK_TECH, WF)
    for src in ("cyclic", "learned"):
        assert rep[src]["de_technology"] >= 15, rep[src]
        assert rep[src]["de_waterfall"] >= 25, rep[src]
        assert rep[src]["contrast_on_floor"] >= 3.0, rep[src]


def test_source_colours_survive_colour_vision_deficiency():
    rep = B.palette_report(ATK_TECH, WF)
    for pair, d in rep["pairs"].items():
        for kind in ("normal", "protan", "deutan"):
            assert d[kind] >= 20, (pair, kind, d)


def _det(**kw):
    base = dict(t0=0.0, t1=1.0, f_lo=162.39e6, f_hi=162.41e6)
    base.update(kw)
    return B.Detection(**base)


def test_proposed_ai_box_is_magenta_dashed_with_its_confidence():
    s = B.box_style(_det(sources=("learned",), confidence=0.87, cls="dmr"),
                    technology_colour="#ff9f40")
    assert s["edge"] == [B.AI_COLOUR] and s["dash"] == "dash"
    assert s["badges"] == ["AI 0.87"]
    assert s["caption"] == "DMR (4FSK TDMA)" and s["caption_colour"] == "#ff9f40"
    assert s["pulse_ms"] == B.PULSE_MS


def test_agreement_alternates_and_badges_stack():
    s = B.box_style(_det(sources=("cyclic", "learned"), integration_s=4.2,
                         confidence=0.5))
    assert s["dash"] == "alternate"
    assert s["edge"] == [B.CYCLIC_COLOUR, B.AI_COLOUR]
    assert s["badges"] == ["α 4.2 s", "AI 0.50"]


def test_confirmed_is_solid_in_the_technology_colour_and_keeps_its_badges():
    d = _det(sources=("learned",), confidence=0.9, cls="p25")
    d.confirm("dsd", "TG 1234")
    s = B.box_style(d, technology_colour="#ff5c8a")
    assert s["dash"] == "solid" and s["edge"] == ["#ff5c8a"] and s["width"] == 3
    assert s["badges"] == ["AI 0.90", "✓ dsd"]
    assert "confirmed by dsd: TG 1234" in s["tooltip"]


def test_energy_alone_is_quiet():
    s = B.box_style(_det(sources=("energy",)))
    assert s["edge"] == [B.ENERGY_COLOUR] and s["pulse_ms"] == 0
    assert s["badges"] == []


def test_the_decoder_wins_the_label_and_the_disagreement_is_kept():
    d = _det(sources=("learned",), cls="dmr", confidence=0.7)
    d.confirm("dsd", "", decoder_class="p25")
    assert d.cls == "p25" and "disagreement" in d.flags
    assert d.measurements["classifier_said"] == "dmr"


def test_merge_unions_sources_and_keeps_disagreement():
    a = _det(sources=("energy",), snr_db=6.0)
    b = _det(t0=0.1, t1=0.9, sources=("learned",), cls="dmr", confidence=0.8,
             family="fsk")
    c = _det(t0=0.05, sources=("cyclic",), cls="p25", confidence=0.6,
             alpha_hz=4800.0, integration_s=3.0)
    far = _det(f_lo=163e6, f_hi=163.02e6, sources=("energy",))
    out = B.merge([a, b, c, far])
    assert len(out) == 2
    m = [o for o in out if o.f_lo < 163e6][0]
    assert m.sources == ("energy", "cyclic", "learned")
    assert m.cls == "dmr" and m.alpha_hz == 4800.0 and m.family == "fsk"
    assert "disagreement" in m.flags


def test_json_round_trip():
    d = _det(sources=("cyclic",), flags=("escalated",))
    assert B.Detection.from_json(d.to_json()) == d


def test_unknown_flags_are_refused():
    with pytest.raises(ValueError):
        _det().with_flag("probably-fine")


def test_class_table_is_consistent():
    names = C.names()
    assert len(names) == len(set(names))
    for c in C.CLASSES:
        assert c.family in C.FAMILIES, c.name
        for d in c.decoders:
            assert d in C.DECODERS, (c.name, d)
    assert C.get("dmr").confirmable and not C.get("lora").confirmable
    assert C.tools_for("pocsag") == ["pager", "multimon"]
    assert ("p25", 4800.0) in C.cycle_frequencies(2.4e6)
    assert all(a < 1.2e6 for _n, a in C.cycle_frequencies(2.4e6))


def test_every_technology_key_is_one_atk_draws():
    keys = {c.technology for c in C.CLASSES if c.technology}
    assert keys <= set(ATK_TECH), keys - set(ATK_TECH)
