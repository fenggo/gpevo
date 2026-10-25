"""Python equivalent of Read_AbinitCode.m (GULP path)."""
from __future__ import annotations

from gpevo.FunctionFolder.AbinitCode.Read_GULP import Read_GULP


def Read_AbinitCode(code, flag, ID=None):
    if code == 3:
        return Read_GULP(flag, ID)
    raise NotImplementedError(f"abinitioCode {code} is not ported yet; current INPUT.txt uses GULP")
