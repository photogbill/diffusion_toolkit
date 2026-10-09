# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""What this installation can do, said in words (plan §2.9 — Bill-proof:
status in plain words; nothing happens silently).

Probing uses `importlib.util.find_spec` — it never imports PyTorch or
TorchSig, which take seconds and gigabytes, just to answer "is it there".
Version numbers come from package metadata for the same reason. The one
exception is `gpu()`, which must import PyTorch to ask it about CUDA: it
runs only when called (the command line's `status` calls it), never at
import.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib import metadata

#: module -> (distribution name, what it unlocks, where it is expected)
OPTIONAL = {
    "numpy":        ("numpy", "everything", "core"),
    "scipy":        ("scipy", "filters, decimation, CFAR, the cut", "core"),
    "onnxruntime":  ("onnxruntime", "CPU inference of trained models inside ATK", "core"),
    "tifffile":     ("tifffile", "GeoTIFF products (a minimal writer is built in)", "core"),
    "torch":        ("torch", "training and the diffusion models", "train"),
    "torchvision":  ("torchvision", "the FCOS 2D proposer", "train"),
    "torchaudio":   ("torchaudio", "audio models", "train"),
    "torchsig":     ("torchsig", "TorchSig synthetic data (the native generator "
                     "works without it)", "train"),
    "onnx":         ("onnx", "exporting trained models to ONNX", "train"),
    "transformers": ("transformers", "text diffusion backends (plan F)", "text"),
    "llama_cpp":    ("llama-cpp-python", "the autoregressive novelty backend "
                     "(ATK's own models)", "text"),
    "serial":       ("pyserial", "the ESP32 CSI sensor over USB serial", "core"),
}


def has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def version(module: str) -> str:
    dist = OPTIONAL.get(module, (module,))[0]
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return ""


def report() -> dict:
    """{module: {present, version, unlocks, env}} plus 'python'."""
    out = {"python": sys.version.split()[0]}
    for mod, (_dist, unlocks, env) in OPTIONAL.items():
        out[mod] = {"present": has(mod), "version": version(mod) if has(mod) else "",
                    "unlocks": unlocks, "env": env}
    return out


def lines(rep: dict | None = None) -> list[str]:
    """The report as lines an operator can read."""
    rep = rep or report()
    out = [f"Python {rep['python']}"]
    for mod, info in rep.items():
        if mod == "python":
            continue
        mark = "[OK]" if info["present"] else "[--]"
        ver = f" {info['version']}" if info["version"] else ""
        out.append(f"{mark} {mod}{ver} — {info['unlocks']}"
                   + ("" if info["present"] else f" (expected in the "
                      f"{info['env']} environment)"))
    return out


def can_train() -> tuple[bool, str]:
    if not has("torch"):
        return False, ("PyTorch is not in this environment. Training runs in "
                       "the toolkit's training environment (envs\\atk_diffusion, "
                       "built by get_diffusion.bat), never inside ATK's core.")
    return True, ""


def can_infer() -> tuple[bool, str]:
    if not has("onnxruntime"):
        return False, ("ONNX Runtime is not installed, so trained models cannot "
                       "run here; the classical detectors (energy, cyclic) "
                       "still work. Run get_diffusion.bat.")
    return True, ""


def gpu() -> dict:
    """Whether PyTorch can use a GPU here, in words. IMPORTS PyTorch (a few
    seconds) — call it when the answer is wanted, not to probe.
    -> {available, name, cuda, torch, words}."""
    out = {"available": False, "name": "", "cuda": "", "torch": "", "words": ""}
    if not has("torch"):
        out["words"] = ("GPU: not checked — PyTorch is not in this environment "
                        "(training runs in the toolkit's training environment).")
        return out
    try:
        import torch
        out["torch"] = str(getattr(torch, "__version__", ""))
        out["cuda"] = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
        out["available"] = bool(torch.cuda.is_available())
        if out["available"]:
            out["name"] = str(torch.cuda.get_device_name(0))
    except Exception as e:                                 # noqa: BLE001
        out["available"] = False
        out["words"] = (f"GPU: PyTorch is installed but did not load ({type(e).__name__}: "
                        f"{e}). Training cannot run until it does; rebuilding the "
                        "training environment usually mends it.")
        return out
    if out["available"]:
        out["words"] = (f"GPU: {out['name']} (CUDA {out['cuda']}) — training and the "
                        "diffusion models use it.")
    elif "+cpu" in out["torch"] or not out["cuda"]:
        out["words"] = (f"GPU: none — this PyTorch ({out['torch']}) is a CPU-only "
                        "build. Training runs on the CPU, many times slower; the "
                        "CUDA build is what install.bat installs without /cpu.")
    else:
        out["words"] = (f"GPU: none that PyTorch can use (PyTorch {out['torch']}, "
                        f"built for CUDA {out['cuda']}). Training runs on the CPU, "
                        "many times slower; check the NVIDIA driver.")
    return out
