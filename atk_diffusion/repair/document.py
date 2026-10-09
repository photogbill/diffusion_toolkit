# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Document restoration before OCR — plan §4.D5 (the forensics repair's
document half).

    *"D5 Forensics repair — document restoration before OCR (deblur,
    denoise, de-skew) and audio inpainting in the Media Lab."*  — plan §4.D5

WHAT, in the order it runs (each step is optional; each records its
parameters and a measured before/after number):

  1. **denoise** — median, bilateral (Tomasi & Manduchi 1998) or non-local
     means (Buades, Coll & Morel 2005); OpenCV when it is importable
     (`cv2.medianBlur`, `bilateralFilter`, `fastNlMeansDenoising`), a
     numpy/scipy implementation of the same filter when it is not, so the
     tool works in ATK's core environment either way. The strength follows
     the noise measured by Immerkær's (1996) estimator.
  2. **deskew** — the projection-profile search (Postl 1986): rotate the
     ink map through candidate angles and keep the one whose row profile is
     sharpest (sum of squared differences between rows — text lines become
     peaks only when they are level). Coarse 0.5° steps over ±10°, then
     0.05° steps.
  3. **deblur** — a Gaussian point-spread function whose width is ESTIMATED
     from the page (the gradient-ratio method: re-blurring a blurred edge by
     a known σr lowers its gradient by √(σ²+σr²)/σ — Zhuo & Sim 2011), then
     Wiener deconvolution (noise-to-signal ratio from the measured noise) or
     Richardson–Lucy (1972/1974) iterations.
  4. **binarise** — Sauvola & Pietikäinen (2000): a threshold from the local
     mean and deviation, T = m·(1 + k·(s/R − 1)), which follows uneven
     lighting and stains where a single global threshold cannot.

TIER: CLEANED. Every step is a deterministic filter of the scan itself: none
can add a character the page did not have. (The plan's line called D5's
output Invented; provenance.py's decision keeps INVENTED for generative
models, so an analyst who sees it learns it means something.) Deconvolution
can RING and amplify noise into specks, and binarisation decides what is
ink — so the original scan stays the record, and `ocr_accuracy` measures
characters produced on BLANK pages: the hallucination number for OCR
before and after.

NOT FOR IDENTIFYING PEOPLE (plan §2.4): this restores text on a page. It
does not sharpen faces, plates or signatures for identification, and is not
to be used to read redactions (plan §5).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov

for _m in ("bilateral", "nl_means", "sauvola", "richardson_lucy"):
    _prov.METHOD_TIERS.setdefault(_m, "cleaned")

DENOISERS = ("median", "bilateral", "nl_means")
DEBLURS = ("wiener", "richardson_lucy")


def _cv2():
    try:
        import cv2                                       # optional
        return cv2
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Images in and out: float32 grey, 0 = black ink, 1 = white paper
# ---------------------------------------------------------------------------
def to_gray(img) -> np.ndarray:
    """A PIL image, an array (uint8 / float, grey or RGB) or a path -> float32
    grey in [0, 1]."""
    if isinstance(img, (str, Path)):
        return load_image(img)
    if hasattr(img, "convert") and hasattr(img, "size"):
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].astype(np.float32)
        a = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
        a = a / 255.0 if a.max() > 1.5 else a
        return a.astype(np.float32)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    return np.clip(a.astype(np.float32), 0.0, 1.0)


def load_image(path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is needed to read image files; pass the "
                           "image as an array instead") from None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def save_image(path, img) -> Path:
    from PIL import Image
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(np.round(np.asarray(img, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(a, mode="L").save(p)
    return p


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def noise_sigma(img) -> float:
    """Immerkær (1996), "Fast noise variance estimation": the response to a
    mask that cancels smooth image structure, σ = √(π/2)/(6(W−2)(H−2))·Σ|r|.
    Text edges leak into it a little; it is a guide for the filter
    strength, reported with the steps."""
    from scipy.signal import convolve2d
    a = np.asarray(img, dtype=np.float64)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    N = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    r = convolve2d(a, N, mode="valid")
    return float(math.sqrt(math.pi / 2.0) * np.sum(np.abs(r))
                 / (6.0 * r.shape[0] * r.shape[1]))


def sharpness(img) -> float:
    """Variance of the Laplacian (Pech-Pacheco et al. 2000) — the focus
    measure: higher is sharper."""
    from scipy.ndimage import laplace
    return float(np.var(laplace(np.asarray(img, dtype=np.float64))))


def otsu(img) -> float:
    """Otsu's global threshold on a [0, 1] image."""
    a = np.clip(np.asarray(img, dtype=np.float64).ravel(), 0, 1)
    hist, edges = np.histogram(a, bins=256, range=(0, 1))
    w = np.cumsum(hist).astype(np.float64)
    mu = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2.0)
    total, mt = w[-1], mu[-1]
    w1 = total - w
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (mt * w / total - mu) ** 2 / (w * w1 / total)
    var = np.nan_to_num(var)
    return float(edges[int(np.argmax(var)) + 1])


def contrast(img) -> float:
    """Paper-to-ink contrast: median of the bright class minus median of the
    dark class (Otsu split). 1.0 is black ink on white paper."""
    a = np.asarray(img, dtype=np.float64)
    t = otsu(a)
    hi, lo = a[a > t], a[a <= t]
    if hi.size == 0 or lo.size == 0:
        return 0.0
    return float(np.median(hi) - np.median(lo))


def psnr(ref, img) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    img = np.asarray(img, dtype=np.float64)
    mse = float(np.mean((ref - img) ** 2))
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def ink_f1(ref_ink, ink) -> float:
    """F-measure of ink pixels (the binarisation contest's measure)."""
    r = np.asarray(ref_ink, dtype=bool)
    p = np.asarray(ink, dtype=bool)
    tp = float(np.sum(r & p))
    if tp == 0:
        return 0.0
    prec = tp / max(float(np.sum(p)), 1.0)
    rec = tp / max(float(np.sum(r)), 1.0)
    return 2 * prec * rec / (prec + rec)


def specks(ink, max_px: int = 4) -> int:
    """Connected ink components of at most `max_px` pixels — specks an OCR
    engine may read as punctuation that is not on the page."""
    from scipy.ndimage import label
    lab, n = label(np.asarray(ink, dtype=bool))
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel())[1:]
    return int(np.sum(sizes <= max_px))


# ---------------------------------------------------------------------------
# 1. Denoise
# ---------------------------------------------------------------------------
def _bilateral_np(a, radius: int, sigma_s: float, sigma_r: float):
    pad = np.pad(a, radius, mode="reflect")
    acc = np.zeros_like(a)
    wsum = np.zeros_like(a)
    H, W = a.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sh = pad[radius + dy:radius + dy + H, radius + dx:radius + dx + W]
            w = math.exp(-(dx * dx + dy * dy) / (2 * sigma_s ** 2)) * \
                np.exp(-((sh - a) ** 2) / (2 * sigma_r ** 2))
            acc += w * sh
            wsum += w
    return acc / np.maximum(wsum, 1e-12)


def _nlm_np(a, h: float, patch: int = 3, search: int = 5):
    """Non-local means: each pixel the weighted mean of pixels in a
    (2·search+1)² window, weighted by how alike their (2·patch+1)² patches
    are — exp(−max(d² − 2σ², 0)/h²) on the patch mean squared difference."""
    from scipy.ndimage import uniform_filter
    pad = np.pad(a, search, mode="reflect")
    H, W = a.shape
    acc = np.zeros_like(a)
    wsum = np.zeros_like(a)
    k = 2 * patch + 1
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            sh = pad[search + dy:search + dy + H, search + dx:search + dx + W]
            d2 = uniform_filter((a - sh) ** 2, size=k, mode="reflect")
            w = np.exp(-np.maximum(d2, 0.0) / (h * h))
            acc += w * sh
            wsum += w
    return acc / np.maximum(wsum, 1e-12)


def denoise(img, method: str = "nl_means", strength: float | None = None,
            use_cv2: bool | None = None) -> np.ndarray:
    """`strength` is in units of the [0, 1] image (default: from the
    measured noise). OpenCV is used when importable unless use_cv2=False."""
    if method not in DENOISERS:
        raise ValueError(f"unknown denoiser {method!r} — one of {', '.join(DENOISERS)}")
    a = to_gray(img).astype(np.float64)
    sig = noise_sigma(a)
    h = float(strength) if strength else max(sig, 0.01)
    cv2 = _cv2() if use_cv2 in (None, True) else None
    if use_cv2 is True and cv2 is None:
        raise RuntimeError("OpenCV was asked for and is not installed")
    u8 = np.clip(np.round(a * 255), 0, 255).astype(np.uint8)
    if method == "median":
        if cv2 is not None:
            return cv2.medianBlur(u8, 3).astype(np.float32) / 255.0
        from scipy.ndimage import median_filter
        return median_filter(a, size=3, mode="reflect").astype(np.float32)
    if method == "bilateral":
        if cv2 is not None:
            out = cv2.bilateralFilter(u8, d=5, sigmaColor=float(2.5 * h * 255),
                                      sigmaSpace=1.5)
            return out.astype(np.float32) / 255.0
        return _bilateral_np(a, 2, 1.5, 2.5 * h).astype(np.float32)
    # non-local means
    if cv2 is not None:
        out = cv2.fastNlMeansDenoising(u8, None, h=float(max(1.0, 1.2 * h * 255)),
                                       templateWindowSize=7, searchWindowSize=21)
        return out.astype(np.float32) / 255.0
    return _nlm_np(a, 1.2 * h).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Deskew
# ---------------------------------------------------------------------------
def _rotate(a, angle_deg: float, fill: float = 1.0, order: int = 1):
    from scipy.ndimage import rotate
    return rotate(a, angle_deg, reshape=False, order=order, mode="constant",
                  cval=fill, prefilter=False)


def _profile_score(ink) -> float:
    p = ink.sum(axis=1)
    return float(np.sum(np.diff(p) ** 2))


def estimate_skew(img, max_deg: float = 10.0, coarse: float = 0.5,
                  fine: float = 0.05, work_px: int = 800) -> tuple[float, float]:
    """(angle, score): the angle the TEXT is rotated by, counter-clockwise
    positive (PIL's and scipy's convention); `deskew` rotates by −angle."""
    a = to_gray(img).astype(np.float64)
    scale = min(1.0, work_px / max(a.shape))
    if scale < 1.0:
        from scipy.ndimage import zoom
        a = zoom(a, scale, order=1)
    ink = (a < otsu(a)).astype(np.float64)
    if ink.sum() < 10:
        return 0.0, 0.0

    def score(ang):
        return _profile_score(_rotate(ink, -ang, fill=0.0, order=1))
    grid = np.arange(-max_deg, max_deg + 1e-9, coarse)
    s = np.array([score(g) for g in grid])
    best = float(grid[int(np.argmax(s))])
    grid2 = np.arange(best - coarse, best + coarse + 1e-9, fine)
    s2 = np.array([score(g) for g in grid2])
    i = int(np.argmax(s2))
    return float(grid2[i]), float(s2[i])


def deskew(img, angle: float | None = None) -> tuple[np.ndarray, float]:
    a = to_gray(img)
    if angle is None:
        angle, _ = estimate_skew(a)
    if abs(angle) < 1e-6:
        return a.copy(), 0.0
    return _rotate(a.astype(np.float64), -angle, fill=1.0).astype(np.float32), float(angle)


# ---------------------------------------------------------------------------
# 3. Deblur
# ---------------------------------------------------------------------------
def estimate_blur_sigma(img, sigma_r: float = 1.0, top: float = 0.02,
                        pct: float = 5.0) -> float:
    """σ (pixels) of a Gaussian blur, from how much a known re-blur σr lowers
    the strongest edges' gradients: ρ = |∇I| / |∇(I∗Gσr)| = √(σ²+σr²)/σ,
    over the top 2 % gradient pixels; clipped to [0.3, 6].

    WHY THE 5TH PERCENTILE OF ρ, NOT THE MEDIAN: a text stroke is a bar, and
    the far edge of a thin bar pulls the re-blurred gradient down more than
    the blurred one — ρ comes out high and σ low. The isolated edges (the
    LOW end of ρ) are the honest ones. Measured on synthetic text pages:
    the median read 27 % low; the 5th percentile is within 3 % for σ from
    0.8 to 3 px on a clean page and about 8 % low after non-local-means
    denoising."""
    from scipy.ndimage import gaussian_filter, sobel
    a = to_gray(img).astype(np.float64)
    a = gaussian_filter(a, 0.5)                      # tame pixel noise first
    g1 = np.hypot(sobel(a, 0), sobel(a, 1))
    b = gaussian_filter(a, sigma_r)
    g2 = np.hypot(sobel(b, 0), sobel(b, 1))
    thr = np.quantile(g1, 1.0 - top)
    sel = (g1 >= thr) & (g2 > 1e-6)
    if sel.sum() < 20:
        return 0.3
    rho = float(np.percentile(g1[sel] / g2[sel], pct))
    if rho <= 1.0 + 1e-6:
        return 6.0
    s = sigma_r / math.sqrt(rho * rho - 1.0)
    # the 0.5 px pre-smoothing is part of what was measured; take it out
    s = math.sqrt(max(s * s - 0.25, 0.0))
    return float(min(6.0, max(0.3, s)))


def gaussian_psf(sigma: float) -> np.ndarray:
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1)
    g = np.exp(-x * x / (2.0 * sigma * sigma))
    k = np.outer(g, g)
    return k / k.sum()


def _otf(psf, shape):
    P = np.zeros(shape)
    kh, kw = psf.shape
    P[:kh, :kw] = psf
    P = np.roll(P, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.rfft2(P)


def deblur(img, method: str = "wiener", sigma: float | None = None,
           nsr: float | None = None, iterations: int = 20) -> tuple[np.ndarray, dict]:
    """Deconvolve a Gaussian blur. Returns (image, info{sigma, nsr|iterations})."""
    if method not in DEBLURS:
        raise ValueError(f"unknown deblur {method!r} — one of {', '.join(DEBLURS)}")
    a = to_gray(img).astype(np.float64)
    s = float(sigma) if sigma else estimate_blur_sigma(a)
    psf = gaussian_psf(s)
    pad = psf.shape[0]
    ap = np.pad(a, pad, mode="reflect")
    info = {"sigma_px": s}
    if method == "wiener":
        if nsr is None:
            # never below 0.01: after denoising the measured noise is tiny,
            # and a Wiener filter trusting it rings every stroke into specks
            # (measured: 0.0016 → 1,166 specks; 0.01 → 8, ink F1 0.64 → 0.85)
            nv = noise_sigma(a) ** 2
            sv = max(float(np.var(a)), 1e-6)
            nsr = float(min(0.1, max(0.01, nv / sv)))
        H = _otf(psf, ap.shape)
        G = np.fft.rfft2(ap)
        F = np.conj(H) / (np.abs(H) ** 2 + nsr) * G
        out = np.fft.irfft2(F, s=ap.shape)
        info["nsr"] = nsr
    else:
        from scipy.signal import fftconvolve
        u = np.maximum(ap, 1e-3)
        flip = psf[::-1, ::-1]
        g = np.maximum(ap, 1e-6)
        for _ in range(int(iterations)):
            est = np.maximum(fftconvolve(u, psf, mode="same"), 1e-6)
            u = u * fftconvolve(g / est, flip, mode="same")
            u = np.clip(u, 1e-4, 2.0)
        out = u
        info["iterations"] = int(iterations)
    out = out[pad:pad + a.shape[0], pad:pad + a.shape[1]]
    return np.clip(out, 0.0, 1.0).astype(np.float32), info


# ---------------------------------------------------------------------------
# 4. Binarise
# ---------------------------------------------------------------------------
def sauvola(img, window: int = 25, k: float = 0.2, R: float = 0.5) -> np.ndarray:
    """Ink mask (True = ink) by Sauvola & Pietikäinen (2000)."""
    from scipy.ndimage import uniform_filter
    a = to_gray(img).astype(np.float64)
    w = int(window) | 1
    m = uniform_filter(a, w, mode="reflect")
    m2 = uniform_filter(a * a, w, mode="reflect")
    s = np.sqrt(np.maximum(m2 - m * m, 0.0))
    T = m * (1.0 + k * (s / R - 1.0))
    return a < T


def binarize(img, window: int = 25, k: float = 0.2) -> np.ndarray:
    return np.where(sauvola(img, window, k), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
@dataclass
class Restored:
    image: np.ndarray
    steps: list = field(default_factory=list)
    tier: str = "cleaned"
    skew_deg: float = 0.0
    blur_sigma_px: float | None = None
    noise_before: float = 0.0

    def report(self) -> dict:
        d = asdict(self)
        d.pop("image", None)
        d["tier_words"] = _prov.TIER_WORDS[self.tier]
        return d


def restore(img, steps=("denoise", "deskew", "deblur", "binarize"),
            denoise_method: str = "nl_means", deblur_method: str = "wiener",
            skew_deg: float | None = None, blur_sigma: float | None = None,
            window: int = 25, k: float = 0.2, use_cv2: bool | None = None) -> Restored:
    """Run the chosen steps in the order denoise → deskew → deblur →
    binarise (each later step works better on the earlier one's output)."""
    unknown = set(steps) - {"denoise", "deskew", "deblur", "binarize"}
    if unknown:
        raise ValueError(f"unknown step(s): {', '.join(sorted(unknown))}")
    a = to_gray(img)
    rec = Restored(a, noise_before=noise_sigma(a))
    cur = a
    if "denoise" in steps:
        n0 = noise_sigma(cur)
        cur = denoise(cur, denoise_method, use_cv2=use_cv2)
        rec.steps.append({"step": "denoise", "method": denoise_method,
                          "tier": _prov.tier_for(denoise_method),
                          "opencv": _cv2() is not None and use_cv2 is not False,
                          "noise_before": n0, "noise_after": noise_sigma(cur)})
    if "deskew" in steps:
        cur, ang = deskew(cur, skew_deg)
        rec.skew_deg = ang
        rec.steps.append({"step": "deskew", "method": "deskew",
                          "tier": _prov.tier_for("deskew"), "angle_deg": ang,
                          "estimated": skew_deg is None})
    if "deblur" in steps:
        s0 = sharpness(cur)
        cur, info = deblur(cur, deblur_method, sigma=blur_sigma)
        rec.blur_sigma_px = info["sigma_px"]
        rec.steps.append({"step": "deblur", "method": deblur_method,
                          "tier": _prov.tier_for(deblur_method), **info,
                          "sharpness_before": s0, "sharpness_after": sharpness(cur)})
    if "binarize" in steps:
        c0 = contrast(cur)
        cur = binarize(cur, window, k)
        rec.steps.append({"step": "binarize", "method": "sauvola",
                          "tier": _prov.tier_for("sauvola"), "window": window,
                          "k": k, "contrast_before": c0,
                          "contrast_after": contrast(cur),
                          "specks": specks(cur < 0.5)})
    rec.image = cur
    return rec


def restore_file(in_path, out_path, **kw) -> dict:
    """Restore a scan into `out_path` (PNG) + `<out_path>.json` (the steps,
    tier CLEANED, the source's hash). The scan itself is not touched."""
    r = restore(load_image(in_path), **kw)
    p = save_image(out_path, r.image)
    side = {"source": str(in_path), "source_sha256": _prov.sha256_path(in_path),
            "output": str(p), **r.report(),
            "provenance": _prov.stamp("repair.document")}
    sp = Path(str(p) + ".json")
    sp.write_text(json.dumps(side, indent=2, default=float), encoding="utf-8")
    side["sidecar"] = str(sp)
    return side


# ---------------------------------------------------------------------------
# OCR accuracy, with the host's OCR
# ---------------------------------------------------------------------------
def ocr_accuracy(images, truths, ocr, work_dir=None, names=None) -> dict:
    """Character and word accuracy of a host-supplied OCR (ATK's Tesseract:
    `ocr(path) -> str | {"text"} | OcrResult`) on `images` against
    `truths`. Arrays are written as PNG under `work_dir` first (never a
    temporary folder). A truth of "" is a BLANK page: every character read
    there is a hallucination, reported separately."""
    from atk_diffusion.repair import wer as _wer
    images = list(images)
    truths = [str(x) for x in truths]
    if len(images) != len(truths):
        raise ValueError(f"{len(images)} images but {len(truths)} truths")
    rows = []
    cer_tot = _wer.Score(unit="character")
    wer_tot = _wer.Score()
    blank_chars = 0
    blank_pages = 0
    for k, (im, tr) in enumerate(zip(images, truths)):
        if isinstance(im, (str, Path)):
            path = Path(im)
        else:
            if work_dir is None:
                raise ValueError("arrays must be written somewhere for the OCR "
                                 "to read: give work_dir")
            nm = (names[k] if names else f"page_{k:03d}") + ".png"
            path = save_image(Path(work_dir) / nm, to_gray(im))
        try:
            text = _wer.text_of(ocr(str(path)))
            err = ""
        except Exception as e:                                 # noqa: BLE001
            text, err = "", f"{type(e).__name__}: {e}"
        c = _wer.cer(tr, text)
        w = _wer.wer(tr, text)
        if tr.strip() == "":
            # EVERY character read off a blank page is invented — punctuation
            # included (a speck read as "." is exactly the failure)
            blank_pages += 1
            blank_chars += len("".join(text.split()))
        else:
            cer_tot = cer_tot + c
            wer_tot = wer_tot + w
        rows.append({"image": str(path), "truth": tr, "text": text, "error": err,
                     "cer": c.to_json(), "wer": w.to_json()})
    return {"pages": len(rows), "cer": cer_tot.rate if cer_tot.ref_len else None,
            "wer": wer_tot.rate if wer_tot.ref_len else None,
            "char_accuracy": (1.0 - cer_tot.rate) if cer_tot.ref_len else None,
            "blank_pages": blank_pages,
            "chars_on_blank_pages": blank_chars,
            "chars_per_blank_page": (blank_chars / blank_pages) if blank_pages else None,
            "per_image": rows}


def synthetic_page(text_lines, size: tuple = (900, 320), font_px: int = 26,
                   margin: int = 30) -> np.ndarray:
    """A clean test page: black text on white, Pillow's own font. Used by the
    tests and experiments; it is not anybody's document."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("L", size, 255)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=font_px)
    except TypeError:                                    # Pillow < 10.1
        font = ImageFont.load_default()
    y = margin
    for line in text_lines:
        dr.text((margin, y), line, font=font, fill=0)
        y += int(font_px * 1.6)
    return np.asarray(im, dtype=np.float32) / 255.0


def degrade(page, skew_deg: float = 3.0, blur_sigma: float = 1.5,
            noise: float = 0.08, rng=None) -> np.ndarray:
    """The damage a phone photo or a fax adds: a rotation, a Gaussian blur,
    additive noise."""
    from scipy.ndimage import gaussian_filter
    rng = rng if rng is not None else np.random.default_rng(0)
    a = _rotate(np.asarray(page, dtype=np.float64), skew_deg, fill=1.0)
    a = gaussian_filter(a, blur_sigma) if blur_sigma > 0 else a
    a = a + noise * rng.normal(size=a.shape)
    return np.clip(a, 0, 1).astype(np.float32)
