# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The training library and everything that grows and trains from it — the
engine of ATK's Setup → RFMLS Training tab (2026-10-26).

    store       the folder: one subfolder per signal type, real originals
                and their copies, roles, removal and restore
    dirty       dirty copies — the real signal through simulated channel and
                receiver effects (noise, fading, frequency error, clipping,
                interference, dropouts …)
    fresh       fresh copies — the signal re-made by the native generator
                from its measured parameters, with new content
    generative  AI-generated variations — a diffusion model taught on the
                library (PyTorch: training environment only)
    build       the library -> a classifier dataset (held-out real examples
                kept apart) and -> wideband scenes for the signal finder
    train       one call per model: build, train, measure, and say in plain
                words whether it is ready

Importing this package imports nothing heavy (the toolkit's rule).
"""
