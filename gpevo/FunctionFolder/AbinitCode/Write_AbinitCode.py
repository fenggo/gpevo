"""Python equivalent of Write_AbinitCode.m (GULP path)."""
from __future__ import annotations

from gpevo.FunctionFolder.AbinitCode.Write_GULP import Write_GULP


def Write_AbinitCode(code, ind):
    if code == 3:
        return Write_GULP(ind)
    raise NotImplementedError(f"abinitioCode {code} is not ported yet; current INPUT.txt uses GULP")
