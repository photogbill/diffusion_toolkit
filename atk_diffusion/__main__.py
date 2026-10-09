# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""`python -m atk_diffusion <command> …` — the toolkit's command line
(atk_diffusion.cli). ATK runs jobs this way (`diffusion_host.run_job`), and
`atkdiff.bat` does the same from the toolkit's folder: always `-m`, never a
Scripts\\*.exe shim (ATK's rule).

The console is told to replace a character it cannot show rather than stop
the run: an analyst's command must never die on an encoding error halfway
through a night's training because a sentence held a '÷' or a '—'.
"""

from __future__ import annotations

import sys

from atk_diffusion.cli import main


def _console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):     # not a text console stream
            pass


if __name__ == "__main__":
    _console()
    sys.exit(main())
