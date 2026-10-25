"""Pure-Python functional equivalent of the legacy Perl ``getStuff`` helper.

The original helper printed values from matching lines to stdout.  This module
keeps the same matching/column semantics as a regular Python function so it can
be called directly from the main code.
"""
from __future__ import annotations

import os
from typing import Iterable


def get_stuff(filename: str | os.PathLike[str], needle: str, column: int) -> list[str]:
    """Return ``column``-th whitespace-separated token for each line containing ``needle``.

    Parameters mirror the old CLI: the column is 1-based.  Missing tokens become
    empty strings, and an unreadable file raises ``OSError``.
    """
    path = os.fspath(filename)
    if column < 1:
        raise ValueError("column must be >= 1")

    values: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if needle in line:
                parts = line.split()
                values.append(parts[column - 1] if column <= len(parts) else "")
    return values
