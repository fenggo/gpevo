"""Python equivalent of ``FunctionFolder/sys/alignLine.m``."""
from __future__ import annotations


def alignLine(linein: str = "", blank_surround: int = 1) -> str:
    """Center ``linein`` in an 80-character line, mirroring MATLAB alignLine."""
    width = 80
    text = "" if linein is None else str(linein)
    if len(text) >= width:
        return text
    blanks_before = int(round(width / 2 - len(text) / 2))
    blanks_after = width - len(text) - blanks_before
    # The MATLAB implementation surrounds the text with one explicit
    # delimiter in addition to ``blanks_before - 1`` / ``blanks_after - 1``.
    return (" " * max(0, blanks_before)) + text + (" " * max(0, blanks_after))
