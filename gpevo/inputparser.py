"""Parser for USPEX INPUT.txt, compatible with FunctionFolder/getInput.py."""
from __future__ import annotations

import os
from typing import Iterable


def strip_comment(line: str) -> str:
    """Drop a trailing ``!`` / ``#`` / ``%`` comment from *line*.

    ``%`` is MATLAB's comment character, and INPUT.txt uses it in two ways:
    as a trailing comment (``11 : optType   % eV``) and as a block marker
    (``% atomType``).  Only the former is stripped -- a line whose first
    non-blank character is ``%`` is returned unchanged, so block detection
    keeps working.
    """
    if line.lstrip().startswith("%"):
        return line
    for mark in ("!", "#", "%"):
        pos = line.find(mark)
        if pos >= 0:
            line = line[:pos]
    return line


class InputParser:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            self.lines = fh.readlines()

    def find_line(self, begin_keyword: str) -> int | None:
        target = begin_keyword.lower()
        for i, raw in enumerate(self.lines):
            line = raw.lower()
            # Keyword must appear near beginning of the meaningful line.
            pos = line.find(target)
            if 0 <= pos < 10:
                return i
        return None

    def scalar(self, begin_keyword: str, col_number: int = 1):
        i = self.find_line(begin_keyword)
        if i is None:
            return None
        raw = strip_comment(self.lines[i]).strip()
        parts = raw.split()
        idx = int(col_number) - 1
        if idx >= len(parts):
            return None
        return parts[idx]

    def option(self, keyword: str):
        """Value of an INPUT option, either ``value : keyword`` or ``keyword : value``.

        USPEX writes options as ``11 : optType``; the GP options are also seen
        written the other way round (``den : 1.8``), so both sides of the colon
        are accepted.  The keyword is matched on the first token of a side, so
        a trailing comment or a unit (``den : 1.8   % g/cm3``,
        ``0.0 : ExternalPressure (GPa)``) does not hide the option.
        """
        target = str(keyword).strip().lower()
        for raw in self.lines:
            line = strip_comment(raw).strip()
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            left, right = left.strip(), right.strip()
            for side, other in ((right, left), (left, right)):
                tokens = side.split()
                if tokens and tokens[0].lower() == target:
                    values = other.split()
                    return values[0] if values else None
        return None

    def block(self, begin_keyword: str, end_keyword: str):
        begin = self.find_line(begin_keyword)
        if begin is None:
            return []
        for i in range(begin + 1, len(self.lines)):
            if end_keyword.lower() in self.lines[i].lower():
                end = i
                break
        else:
            return []
        out = []
        for line in self.lines[begin + 1:end]:
            line = strip_comment(line).strip()
            if not line:
                continue
            out.append(line)
        return out


def scannum(text):
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    pieces = s.replace(",", " ").replace(";", " ").split()
    try:
        vals = []
        for p in pieces:
            v = float(p)
            vals.append(int(v) if v.is_integer() else v)
        return vals[0] if len(vals) == 1 else vals
    except Exception:
        return None
