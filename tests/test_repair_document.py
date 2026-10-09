# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Document restoration before OCR (plan §4.D5) — measured on synthetic
pages, without Tesseract."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PIL")      # the synthetic page is drawn with Pillow

from atk_diffusion import provenance  # noqa: E402
from atk_diffusion.repair import document as D  # noqa: E402

LINES = ["NOTICE TO ALL STATIONS 0815Z",
         "Relief convoy departs grid 18S UJ 2345 6789",
         "Water point open at the school, bring containers"]


@pytest.fixture(scope="module")
def page():
    return D.synthetic_page(LINES, size=(760, 220), font_px=24)


def test_methods_are_cleaned_tier():
    for m in ("median", "bilateral", "nl_means", "deskew", "wiener",
              "richardson_lucy", "sauvola"):
        assert provenance.tier_for(m) == "cleaned"


def test_deskew_recovers_the_angle(page, rng):
    for ang in (3.5, -2.0):
        deg = D.degrade(page, skew_deg=ang, blur_sigma=1.0, noise=0.05, rng=rng)
        est, _ = D.estimate_skew(deg)
        assert abs(est - ang) < 0.3, (ang, est)
        fixed, used = D.deskew(deg, est)
        assert used == est
        assert D.psnr(page, fixed) > D.psnr(page, deg) + 2.0


def test_blur_sigma_is_estimated(page):
    for s in (1.0, 1.5, 2.5):
        deg = D.degrade(page, skew_deg=0, blur_sigma=s, noise=0.0)
        assert abs(D.estimate_blur_sigma(deg) - s) < 0.12 * s


@pytest.mark.parametrize("method", D.DENOISERS)
def test_denoisers_with_and_without_opencv(page, method, rng):
    noisy = D.degrade(page, 0, 0.8, 0.06, rng)
    n0 = D.noise_sigma(noisy)
    # the bilateral filter is gentle by design: it keeps edges first
    need = 2.0 if method == "bilateral" else 3.0
    if D._cv2() is not None:
        assert D.noise_sigma(D.denoise(noisy, method, use_cv2=True)) < n0 / need
    crop = noisy[:100, :200]                     # the numpy path, small and quick
    out = D.denoise(crop, method, use_cv2=False)
    assert out.shape == crop.shape
    assert D.noise_sigma(out) < D.noise_sigma(crop) / 2.0
    with pytest.raises(ValueError):
        D.denoise(crop, "magic")


@pytest.mark.parametrize("method", D.DEBLURS)
def test_deblur_restores_sharpness(page, method):
    blurred = D.degrade(page, 0, 1.5, 0.0)
    out, info = D.deblur(blurred, method)
    assert abs(info["sigma_px"] - 1.5) < 0.2
    assert D.sharpness(out) > 2 * D.sharpness(blurred)
    assert D.psnr(page, out) > D.psnr(page, blurred)
    assert out.min() >= 0 and out.max() <= 1


def test_sauvola_beats_one_global_threshold_under_uneven_light(page):
    light = np.linspace(0.45, 1.0, page.shape[1])[None, :]
    lit = page * light
    ref = page < 0.5
    glob = lit < D.otsu(lit)
    assert D.ink_f1(ref, D.sauvola(lit)) > D.ink_f1(ref, glob) + 0.1


def test_full_restoration_is_measurably_better(page, rng):
    deg = D.degrade(page, skew_deg=3.0, blur_sigma=1.5, noise=0.08, rng=rng)
    ref = D.sauvola(page)
    r = D.restore(deg)
    assert r.tier == "cleaned" and abs(r.skew_deg - 3.0) < 0.3
    assert [s["step"] for s in r.steps] == ["denoise", "deskew", "deblur", "binarize"]
    assert all(s["tier"] == "cleaned" for s in r.steps)
    before = D.ink_f1(ref, D.sauvola(deg))
    after = D.ink_f1(ref, r.image < 0.5)
    assert before < 0.4 and after > 0.75
    assert D.specks(r.image < 0.5) < D.specks(D.sauvola(deg)) / 10
    grey = D.restore(deg, steps=("denoise", "deskew", "deblur"))
    assert D.psnr(page, grey.image) > D.psnr(page, deg) + 4.0
    rep = r.report()
    assert rep["tier_words"].startswith("CLEANED") and "image" not in rep


def test_a_blank_page_gains_no_ink(rng):
    """The OCR hallucination proxy: restoration must not turn noise on a
    blank page into ink an OCR engine would read."""
    blank = np.clip(1.0 + 0.08 * rng.normal(size=(200, 600)), 0, 1).astype(np.float32)
    raw = D.specks(D.sauvola(blank)) + int(np.sum(D.sauvola(blank)))
    r = D.restore(blank, steps=("denoise", "deblur", "binarize"))
    ink = r.image < 0.5
    assert int(np.sum(ink)) < 0.002 * ink.size
    assert D.specks(ink) + int(np.sum(ink)) < raw


def test_restore_file_writes_png_and_sidecar_only(tmp_path, page, rng):
    src = D.save_image(tmp_path / "scan.png", D.degrade(page, 2.0, 1.0, 0.05, rng))
    side = D.restore_file(src, tmp_path / "out" / "scan_restored.png")
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert files == ["scan_restored.png", "scan_restored.png.json"]
    js = json.loads((tmp_path / "out" / "scan_restored.png.json").read_text())
    assert js["tier"] == "cleaned"
    assert js["source_sha256"] == provenance.sha256_path(src)
    assert D.load_image(side["output"]).shape == page.shape
    with pytest.raises(ValueError, match="unknown step"):
        D.restore(page, steps=("sharpen_face",))


def test_ocr_accuracy_with_a_host_ocr(tmp_path, page, rng):
    """The host's OCR is a callable; here a stand-in that 'reads' a page
    correctly only when its ink matches the clean page, and reads noise
    as dots — enough to test the bookkeeping, not an OCR engine."""
    ref = D.sauvola(page)
    truth = "\n".join(LINES)

    def fake_ocr(path):
        ink = D.load_image(path) < 0.5
        f1 = D.ink_f1(ref, ink)
        if f1 > 0.75:
            return {"text": truth}
        sp = D.specks(ink)
        if sp and f1 < 0.3:                     # speckle read as punctuation
            return {"text": "." * min(40, max(1, sp // 10))}
        return {"text": "" if ink.sum() < 50 else "N0TICE T0 ALL"}

    deg = D.degrade(page, 3.0, 1.5, 0.08, rng)
    blank = np.clip(1 + 0.08 * rng.normal(size=page.shape), 0, 1)
    pages = [D.sauvola(deg) * -1.0 + 1.0, D.restore(deg).image,
             D.sauvola(blank) * -1.0 + 1.0, D.restore(blank, steps=("denoise", "binarize")).image]
    res = D.ocr_accuracy(pages, [truth, truth, "", ""], fake_ocr, work_dir=tmp_path / "ocr")
    rows = res["per_image"]
    assert rows[0]["cer"]["rate"] > 0.5 and rows[1]["cer"]["rate"] == 0.0
    assert res["blank_pages"] == 2
    assert rows[2]["text"].count(".") > rows[3]["text"].count(".")
    assert res["chars_on_blank_pages"] == len(rows[2]["text"]) + len(rows[3]["text"])
    assert res["chars_on_blank_pages"] > 0
    with pytest.raises(ValueError, match="work_dir"):
        D.ocr_accuracy([page], [truth], fake_ocr)
