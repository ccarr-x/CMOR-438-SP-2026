"""Ensure ``src/`` is on ``sys.path`` so ``rice_ml`` imports work without an editable install."""

from __future__ import annotations

import pathlib
import sys


def _ensure_src_on_path() -> None:
    here = pathlib.Path(__file__).resolve().parent
    for root in [here, *here.parents]:
        if (root / "pyproject.toml").is_file() and (root / "src").is_dir():
            src = root / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            return


_ensure_src_on_path()
