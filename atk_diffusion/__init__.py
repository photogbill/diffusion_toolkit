# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""ATK Diffusion Toolkit — diffusion models and learned-RF tools for ATK.

The plan of record is ATK_DIFFUSION_PLAN.md; the detector's design is
DETECTION_DESIGN.md; docs/ARCHITECTURE.md maps both onto this package.

IMPORTING THIS PACKAGE IMPORTS NOTHING HEAVY. ATK imports it inside its own
core environment (numpy, scipy, onnxruntime) for the classical tools and CPU
inference; PyTorch and TorchSig exist only in the training environment and
are imported inside the functions that need them (`atk_diffusion.learn`).
`atk_diffusion.capabilities.report()` says, in words, what this
installation can and cannot do.
"""

__version__ = "0.1.0"

PACKAGE = "atk_diffusion"
REPOSITORY = "github.com/photogbill/diffusion_toolkit"
