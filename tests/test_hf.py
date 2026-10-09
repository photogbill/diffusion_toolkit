# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""HF propagation now (plan §4.J): Maidenhead, WSPR spots, the decoder at
arm's length, the openness matrix and its products, the VOACAP adapter and
the measured-versus-predicted experiment."""

from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion.experiments import hf_eval as E
from atk_diffusion.hf import maidenhead as M
from atk_diffusion.hf import openness as O
from atk_diffusion.hf import wspr as W


# -- Maidenhead ------------------------------------------------------------------
@pytest.mark.parametrize("lat,lon,grid", [(51.4778, -0.0015, "IO91xl"),
                                          (-33.86, 151.21, "QF56od"),
                                          (38.70, -77.57, "FM18fq")])
def test_known_places(lat, lon, grid):
    assert M.from_latlon(lat, lon, 6) == grid
    la, lo = M.to_latlon(grid)
    assert abs(la - lat) < 1 / 24 and abs(lo - lon) < 2 / 24


def test_round_trips_at_every_precision_and_the_rectangle():
    for p in (2, 4, 6, 8, 10):
        g = M.from_latlon(38.7123, -77.5678, p)
        assert len(g) == p and M.from_latlon(*M.to_latlon(g), p) == g
    assert M.bounds("FM18") == (38.0, -78.0, 39.0, -76.0)
    assert M.to_latlon("FM18", center=False) == (38.0, -78.0)
    assert M.normalize("fm18LR") == "FM18lr" and M.is_grid("FM18lr12")
    assert 95 < M.uncertainty_km("FM18") < 110
    with pytest.raises(ValueError, match="not a Maidenhead locator"):
        M.to_latlon("ZZ99")


# -- WSPR spots --------------------------------------------------------------------------
ALL_WSPR = ["210314 1830   3 -21  0.18  14.0970957  K1ABC FN42 37   0     1    0",
            "260108 0102  -15  1.2  10.140150  <W1XYZ> FN31pr 23  1  1  0",
            "this is not a spot",
            "260108 0104  -25  -0.3  7.040112  PJ4/K1ABC 33  -1"]
WSPRD = ["1830 -21  0.2  14.097096  0  K1ABC FN42 37",
         "<DecodeFinished>"]


def test_both_formats_and_all_three_message_types():
    p = W.SpotParser("KIWI1", "IO91xl")
    spots = [s for s in (p.parse(x) for x in ALL_WSPR) if s]
    assert len(spots) == 3 and p.skipped == 1
    a, b, c = spots
    assert (a.time_utc, a.snr_db, a.call, a.grid, a.power_dbm, a.drift_hz) == \
        ("2021-03-14T18:30:00Z", -21, "K1ABC", "FN42", 37, 0)
    assert a.band == "20m" and a.freq_hz == pytest.approx(14_097_095.7)
    assert b.call == "<W1XYZ>" and b.grid == "FN31pr" and b.band == "30m"
    assert c.call == "PJ4/K1ABC" and c.grid == "" and c.distance_km is None
    assert a.distance_km == pytest.approx(5_200, rel=0.05)   # FN42 to IO91
    assert 40 < a.bearing_tx_to_rx < 60 and 280 < a.bearing_rx_to_tx < 300
    d = W.parse_lines(WSPRD, "KIWI1", "IO91xl", date="2026-10-08")[0]
    assert d.time_utc == "2026-10-08T18:30:00Z" and d.drift_hz == 0
    assert d.tier == "measured"


def test_band_of_frequency():
    assert W.band_of(7.0386e6) == "40m" and W.band_of(475_000) == "630m"
    assert W.band_of(10.0e6) == ""


def test_parse_file_reports_what_it_skipped(tmp_path):
    f = tmp_path / "ALL_WSPR.TXT"
    f.write_text("\n".join(ALL_WSPR) + "\n")
    spots, status = W.parse_file(f, rx_grid="FM18")
    assert len(spots) == 3 and "1 lines skipped" in status


def test_wsprd_runs_only_as_its_own_program(tmp_path):
    spots, why = W.run_wsprd(tmp_path / "x.wav", wsprd_path=None, dial_mhz=14.0956,
                             out_dir=tmp_path)
    assert spots == [] and "GPL" in why and "never includes it" in why
    seen = []

    def runner(args):
        seen.append(args)
        return 0, "\n".join(WSPRD)
    spots, why = W.run_wsprd(tmp_path / "x.wav", wsprd_path="/opt/wsprd",
                             dial_mhz=14.0956, out_dir=tmp_path / "w",
                             rx_grid="FM18", date="2026-10-08", runner=runner)
    assert len(spots) == 1 and "decoded 1 spot" in why
    assert seen[0][1:5] == ["-a", str(tmp_path / "w"), "-f", "14.095600"]
    assert seen[0][-1].endswith("x.wav")


def test_prepare_wav_is_12_khz_mono_16_bit(tmp_path):
    fs = 20_250.0
    t = np.arange(int(5 * fs)) / fs
    p = W.prepare_wav(np.sin(2 * np.pi * 1500 * t), fs, tmp_path / "a.wav")
    with wave.open(str(p)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (12000, 1, 2)
        assert w.getnframes() == pytest.approx(60_000, abs=2)
    with pytest.raises(ValueError, match="not IQ"):
        W.prepare_wav(np.ones(10, complex), 12000, tmp_path / "b.wav")


# -- the openness matrix and its products -----------------------------------------------------
def _spots():
    mk = []
    for h, snr, call, band, f in ((22, -20, "K4AA", "40m", 7.0401e6),
                                  (22, -10, "K4AB", "40m", 7.0402e6),
                                  (22, -14, "K4AA", "40m", 7.0401e6),
                                  (23, -25, "K4AA", "20m", 14.0971e6)):
        mk.append(W.place(W.Spot(time_utc=f"2026-10-08T{h:02d}:10:00Z",
                                 snr_db=snr, dt_s=0.1, freq_hz=f, call=call,
                                 grid="FM18", power_dbm=23, band=band,
                                 rx_call="Kiwi UK", rx_grid="IO91xl")))
    mk.append(W.place(W.Spot(time_utc="2026-10-08T22:12:00Z", snr_db=-5,
                             dt_s=0.0, freq_hz=7.0401e6, call="EA1XX", grid="IN73",
                             power_dbm=23, band="40m", rx_call="Kiwi UK",
                             rx_grid="IO91xl")))
    return mk


def test_matrix_counts_medians_and_best_distance():
    here = {"lat": 38.5, "lon": -77.0, "radius_km": 300}
    m = O.matrix(_spots(), hours=(22, 23), receivers={"grids": ["IO91"]}, here=here)
    assert m["bands"] == ["40m", "20m"] and m["spots_used"] == 4   # EA1XX is not here
    c = m["cells"]["40m"][22]
    assert (c["spots"], c["stations"], c["median_snr_db"]) == (3, 2, -14.0)
    assert c["best_km"] == pytest.approx(5_900, rel=0.05) and c["open"]
    assert m["cells"]["20m"][22]["spots"] == 0 and not m["cells"]["20m"][22]["open"]
    assert m["tier"] == "measured" and "not that the band was closed" in m["what"]
    lines = O.matrix_lines(m)
    assert lines[1].startswith("40m") and "·" in lines[2]
    assert O.matrix(_spots(), receivers={"lat": 0.0, "lon": 0.0,
                                         "radius_km": 100})["spots_used"] == 0


def test_great_circle_and_arcs_coloured_in_atks_ramp():
    pts = O.great_circle(38.5, -77.0, 51.5, 0.0, 16)
    assert pts[0] == [-77.0, 38.5] and pts[-1] == [0.0, 51.5]
    wrap = O.great_circle(10.0, 170.0, 10.0, -170.0, 8)
    assert all(abs(b[0] - a[0]) < 30 for a, b in zip(wrap, wrap[1:]))
    gj = O.arcs_geojson(_spots())
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 5
    f = gj["features"][0]
    assert f["geometry"]["type"] == "LineString"
    assert f["properties"]["tier"] == "measured"
    assert O.snr_colour(-40.0) == "#080e28" and O.snr_colour(20.0) == "#fffabe"


def test_products_land_outside_the_install_and_are_logged(rf):
    m = O.matrix(_spots(), receivers={"grids": ["IO91"]})
    files = O.write_products(rf, m, _spots(), run="t1")
    for k in ("matrix_json", "matrix_csv", "arcs"):
        p = Path(files[k])
        assert str(rf.products("hf")) in str(p) and rf.verify(p)[0]
    j = json.loads(Path(files["matrix_json"]).read_text())
    assert j["prediction"]["available"] is False and j["provenance"]["tool"]
    assert Path(files["matrix_csv"]).read_text().startswith("band,hour_utc,spots")


# -- the prediction -----------------------------------------------------------------------------
SAMPLE_OUT = """\
   22.0 18.5  7.0 14.1 FREQ
         1F2  1F2 MODE
        0.95 0.40 REL
          25    8 SNR
   23.0 16.2  7.0 14.1 FREQ
        0.98 0.10 REL
          30    2 SNR
"""


def test_voacap_unavailable_is_said_in_words():
    v = O.Voacap(exe="", itshfbc="")
    r = v.predict((38.5, -77.0), (51.5, 0.0), year=2026, month=10, ssn=100,
                  freqs_mhz=[7.0386])
    assert r["available"] is False and r["why"].startswith("prediction unavailable")


def test_the_deck_has_voacaps_card_layout():
    deck = O.voacap_deck((38.5, -77.0), (51.5, 0.0), year=2026, month=10,
                         ssn=100, freqs_mhz=[7.0386, 14.0956], hours=range(1, 25))
    lines = deck.splitlines()
    names = [ln[:10].strip() for ln in lines]
    for card in ("COEFFS", "TIME", "MONTH", "SUNSPOT", "CIRCUIT", "SYSTEM",
                 "ANTENNA", "FREQUENCY", "METHOD", "EXECUTE", "QUIT"):
        assert card in names
    fq = next(ln for ln in lines if ln.startswith("FREQUENCY"))
    assert len(fq) == 10 + 11 * 5 and fq[10:20] == " 7.0414.10"
    assert "38.50N" in deck and "77.00W" in deck and "51.50N" in deck
    assert next(ln for ln in lines if ln.startswith("METHOD")).split()[1] == "30"


def test_the_output_reader_and_the_adapter_with_a_stand_in_runner(tmp_path):
    """Self-consistency only: the reader against the documented row layout
    (labels at the right). The real program is tested below when present."""
    t = O.parse_voacap_output(SAMPLE_OUT, [7.0, 14.1])
    assert t["hours"][22][7.0] == {"rel": 0.95, "snr": 25.0}
    assert t["hours"][23][14.1]["rel"] == 0.10 and t["muf"][22] == 18.5

    def runner(args, run_dir):
        (Path(run_dir) / "voacapx.out").write_text(SAMPLE_OUT)
        return 0, ""
    v = O.Voacap(runner=runner)
    r = v.predict((38.5, -77.0), (51.5, 0.0), year=2026, month=10, ssn=100,
                  freqs_mhz=[7.0, 14.1], hours=[22, 23])
    assert r["available"] and r["tier"] == "inferred"

    def silent(args, run_dir):
        return 1, "boom"
    r = O.Voacap(runner=silent).predict((38.5, -77.0), (51.5, 0.0), year=2026,
                                        month=10, ssn=100, freqs_mhz=[7.0])
    assert not r["available"] and "could not be read" in r["why"]


@pytest.mark.skipif(shutil.which("voacapl") is None,
                    reason="voacapl is not installed here; the deck writer and "
                           "output reader are verified against it on a machine "
                           "that has it")
def test_against_the_real_voacapl(tmp_path):
    import os
    v = O.Voacap(itshfbc=os.environ.get("ITSHFBC", str(Path.home() / "itshfbc")))
    r = v.predict((38.5, -77.0), (51.5, 0.0), year=2026, month=10, ssn=100,
                  freqs_mhz=[7.0386, 14.0956], hours=range(1, 25))
    assert r["available"], r.get("why")
    assert len(r["table"]["hours"]) >= 20


# -- the experiment -------------------------------------------------------------------------------
def test_hf_eval_measured_beside_predicted(rf):
    r = E.run(rf)
    assert r["synthetic"] and "SYNTHETIC" in r["report_md"]
    assert len(r["kiwis"]) == 3 and all(k["spots_used"] > 0 for k in r["kiwis"])
    assert "prediction unavailable" in r["report_md"]
    for k in r["kiwis"]:
        for p in k["products"].values():
            assert rf.verify(p)[0]
    assert len(r["files"]) == 2 and "kiwisdr_12000_ci16" in r["files"][0]

    hours = list(E.DEFAULT_HOURS)
    body = "".join(f"   {h if h else 24:.1f} 20.0  3.6  7.0 10.1 14.1 FREQ\n"
                   f"        0.50 0.90 0.90 0.20 REL\n"
                   f"          10   20   20    5 SNR\n" for h in hours)

    def runner(args, run_dir):
        (Path(run_dir) / "voacapx.out").write_text(body)
        return 0, ""
    r2 = E.run(None, voacap=O.Voacap(runner=runner),
               kiwis=E.DEFAULT_KIWIS[:1])
    k = r2["kiwis"][0]
    assert k["prediction_available"] and k["compared_cells"] > 0
    assert 0.0 <= k["agreement"] <= 1.0
    assert "agree on open/closed" in r2["report_md"]
