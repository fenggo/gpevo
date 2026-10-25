"""Python equivalent of createORG_Symmetry.m."""
from __future__ import annotations

from gpevo.runtime import G
from gpevo.inputparser import InputParser

# Highest space group number that can be requested.
NSYM_MAX = 230


def _parse_nsym(text, size=NSYM_MAX):
    """Turn the ``symmetries`` block into the 1-based flag vector ORG.nsym.

    Port of the numeric branch of createORG_Symmetry.m:68-97.  The block may
    list single groups (``4 14 19 29 7``) and ranges (``2-230``); every other
    token style is ignored, as in MATLAB.
    """
    flags = [0] * (size + 1)          # index 0 unused, groups are 1..size
    string = " ".join(str(text or "").split())
    if not string:
        return flags

    # Delimiter positions, 1-based, exactly like findstr() in MATLAB.
    delimiters = {i + 1 for i, ch in enumerate(string) if ch in " -"}
    boundaries = sorted(delimiters | {0}) + [len(string) + 1]

    ind1 = 1
    for i in range(1, len(boundaries)):
        start, stop = boundaries[i - 1] + 1, boundaries[i] - 1
        if start > stop:
            continue
        token = string[start - 1:stop]
        try:
            ind2 = int(float(token))
        except ValueError:
            continue
        if ind2 == 0:                 # "0" just resets the start of a range
            ind1 = 1
            continue
        if boundaries[i - 1] and string[boundaries[i - 1] - 1] == "-":
            for j in range(ind1, min(ind2, size) + 1):
                flags[j] = 1
        elif 0 < ind2 <= size:
            flags[ind2] = 1
        ind1 = ind2
    return flags


def createORG_Symmetry(inputFile):
    ORG = G["ORG_STRUC"]
    p = InputParser(inputFile)
    raw = p.scalar("doSpaceGroup")
    if raw is None:
        raw = 0 if ORG.dimension in (0, 2, -3) else 1
    ORG.doSpaceGroup = int(float(raw))
    raw = p.scalar("symmetrize")
    if raw is not None:
        ORG.symmetrize = int(float(raw))
    raw = p.scalar("SymTolerance")
    if raw is not None:
        if raw.lower() in ("high", "medium", "low"):
            ORG.SGtolerance = {"high": 0.05, "medium": 0.10, "low": 0.20}[raw.lower()]
        else:
            ORG.SGtolerance = float(raw)

    block = p.block("symmetries", "EndSymmetries")
    text = "".join(line + " " for line in block) if block else ""
    if text.strip() and int(getattr(ORG, "dimension", 3)) != 0:
        # dimension == 0 keeps the point-group *names* in MATLAB, which this
        # port does not implement, so the block is ignored there.
        ORG.nsym = _parse_nsym(text)
    elif int(getattr(ORG, "dimension", 3)) == 3:
        ORG.nsym = [0] + [1] * (NSYM_MAX - 1)   # 1-based active flags 1..230
    else:
        ORG.nsym = [1] * NSYM_MAX

    # createORG_Symmetry.m:93-95 -- the pool must never be empty, otherwise
    # Random_Init_310 has nothing to draw from (that is the ``tmp(0)`` crash
    # the Octave port hits).  Fall back to P1.
    if not any(ORG.nsym[1:]):
        ORG.nsym[1] = 1
    G["ORG_STRUC"] = ORG
