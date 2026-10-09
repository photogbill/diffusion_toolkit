# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The RF social graph (plan C3): the first experiment in simulation —
handhelds on a scripted weekly schedule; does the graph recover the
schedule and the pairs? — and the exports, round-tripped."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

from atk_diffusion.fingerprint import social as S
from atk_diffusion.geo import products as P


@pytest.fixture(scope="module")
def week():
    sts = S.scripted_week(0)
    return sts, S.build(sts, labels={"A": "base radio", "B": "field radio"})


def test_the_graph_recovers_the_pairs(week):
    _, g = week
    ab = g.link("A", "B", "call_response")
    assert ab is not None and ab.directed and ab.count > 50
    assert ab.expected < 1.0 and ab.p_value < 1e-6
    assert all(0.5 <= e.value <= 10.0 for e in ab.evidence)    # delays, seconds
    assert g.link("B", "A", "call_response") is None            # B answers A, not back
    cd = g.link("C", "D", "co_movement")
    assert cd is not None and not cd.directed and cd.count > cd.expected * 5
    assert g.link("A", "B", "co_movement") is None              # A is parked
    assert g.link("C", "D", "schedule") and g.link("A", "B", "schedule")
    assert g.link("A", "C", "schedule") is None                 # day shift vs night
    assert not [l for l in g.links if "F" in (l.source, l.target)]
    words = " ".join(g.summary()["words"])
    assert "B answers A" in words and "C and D move together" in words


def test_the_graph_recovers_the_schedules(week):
    _, g = week
    assert g.nodes["E"].period_h == 1.0
    a = g.nodes["A"]
    assert a.period_h == 24.0
    hours = set(a.schedule)
    assert {h % 24 for h in hours} <= set(range(8, 17))
    assert {h // 24 for h in hours} <= {0, 1, 2, 3, 4}         # Mon-Fri
    c = g.nodes["C"]
    assert {h % 24 for h in c.schedule} <= {20, 21, 22, 23, 0, 1}
    assert len({h // 24 for h in c.schedule}) == 7              # every night
    assert g.nodes["F"].period_h is None
    assert "Mon, Tue, Wed, Thu, Fri" in a.schedule_words()
    assert g.nodes["A"].decoder_ids == ["DMR:3110001"]
    assert g.nodes["C"].moving and not g.nodes["A"].moving


def test_i2_export_is_one_item_per_row_and_round_trips(week, tmp_path):
    _, g = week
    paths = S.write_i2(g, tmp_path / "i2")
    for p in paths:
        raw = p.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")                 # UTF-8 with BOM
        assert b"\r\n" in raw and b"None" not in raw
        with open(p, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == [c.name for c in S.I2_FILES[p.name]]
        assert all(len(r) == len(rows[0]) for r in rows)
    with open(tmp_path / "i2" / "i2_link_evidence.csv", encoding="utf-8-sig") as f:
        n_ev = sum(1 for _ in csv.reader(f)) - 1
    assert n_ev == sum(len(l.evidence) for l in g.links)      # child table
    with open(tmp_path / "i2" / "i2_entities.csv", encoding="utf-8-sig") as f:
        ent = list(csv.DictReader(f))
    assert ent[0]["first_seen"].endswith("Z") and "T" in ent[0]["first_seen"]
    back = S.read_i2(tmp_path / "i2")
    assert S.same_graph(g, back) == []
    # the reader refuses a file whose columns are not the specification's
    (tmp_path / "i2" / "i2_links.csv").write_text("﻿link,from,to\r\n",
                                                  encoding="utf-8")
    with pytest.raises(ValueError, match="specification"):
        S.read_i2(tmp_path / "i2")


def test_network_link_geojson_and_the_product(week, rf):
    _, g = week
    nl = g.to_network_link("week test")
    assert nl["atk_project_version"] == 1
    nodes, edges = nl["graph"]["nodes"], nl["graph"]["edges"]
    assert {n["entity_type"] for n in nodes} == {"Emitter"}
    assert {n["id"] for n in nodes} == {f"emitter:{k}" for k in g.nodes}
    assert {e["relation_type"] for e in edges} == {"Communication", "Social"}
    assert all(e["meta"]["tier"] == "inferred" and e["meta"]["evidence"] for e in edges)
    json.dumps(nl)                                             # JSON-safe
    feats = g.geojson_features()
    kinds = {f["properties"]["role"] for f in feats}
    assert kinds == {"emitter", "link"}
    run = Path(g.to_products(rf, run="social_week"))
    man = json.loads((run / "manifest.json").read_text())
    assert man["kind"] == "emitters" and man["tier"] == "inferred"
    assert man["method"] == "social_graph"
    assert {"social.geojson", "network_link.atkproj", "i2_links.csv",
            "i2_link_evidence.csv"} <= set(man["files"])
    assert P.verify_run(run) == (True, [])
    gj = P.read_geojson(run / "social.geojson")
    assert all(f["properties"]["atk:tier"] == "inferred" for f in gj["features"])


def test_atks_own_project_loader_reads_the_network_link():
    cands = [os.environ.get("ATK_HOME", ""), "/home/claude/atk_snapshot"]
    home = next((c for c in cands if c and (Path(c) / "atk" / "core" /
                                            "projects.py").exists()), None)
    if home is None:
        pytest.skip("ATK's source is not reachable (set ATK_HOME)")
    sys.path.insert(0, home)
    try:
        from atk.core import projects
    finally:
        sys.path.remove(home)
    g = S.build(S.scripted_week(1))
    nl = g.to_network_link()
    graph = projects._dict_to_graph(nl["graph"])
    assert len(graph.nodes) == len(g.nodes)
    assert len(graph.edges) == len(g.links)
    assert {e.relation_type for e in graph.edges} <= {"Communication", "Social"}


def test_sightings_as_dicts_with_iso_times_and_refusals():
    rows = [{"emitter_id": "X", "t": "2026-10-08T12:00:00Z", "lat": 38.7, "lon": -77.5,
             "decoder_id": "P25:12"},
            {"emitter_id": "Y", "t": "2026-10-08T12:00:03.250000Z"}]
    g = S.build(rows)
    assert g.nodes["X"].decoder_ids == ["P25:12"]
    assert S.iso(g.nodes["Y"].first_seen) == "2026-10-08T12:00:03.250000Z"
    with pytest.raises(ValueError, match="no sightings"):
        S.build([])
    with pytest.raises(ValueError, match="needs a time"):
        S.build([{"emitter_id": "Z", "t": ""}])
