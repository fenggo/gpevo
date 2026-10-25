"""Helper for importing Python mirrors that live in numeric MATLAB folders (310)."""
from __future__ import annotations

import importlib


def module(fullname):
    return importlib.import_module(fullname)
