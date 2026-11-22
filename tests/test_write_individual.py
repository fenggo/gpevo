"""WriteIndividual must tolerate the empty Initialize.py placeholder record.

USPEX.POPULATION[0] is seeded as an empty AttrDict() in Initialize.py.  A
structure rejected before it gets a Number (damaged molecule / fake residual)
never replaces it, so the record reaches WriteIndividual without Enthalpies.
That must be skipped, not crash the whole generation output -- the failure in
ct4 where WriteIndividual.py:23 raised AttributeError: Enthalpies.
"""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.USPEX.src.WriteIndividual import WriteIndividual, _row


def _record(number, gen=1, enthalpy=-1000.0):
    r = AttrDict()
    r.Number = number
    r.gen = gen
    r.Enthalpies = [enthalpy]
    r.numIons = [1]
    r.howCome = "Random"
    return r


def _setup(tmp_path, population):
    res = tmp_path / "results1"
    res.mkdir(exist_ok=True)
    USPEX = AttrDict()
    USPEX.POPULATION = population
    G["USPEX_STRUC"] = USPEX
    return res


def test_empty_placeholder_is_skipped_not_crashed(tmp_path):
    res = _setup(tmp_path, [AttrDict(), _record(1), _record(2)])
    WriteIndividual(str(res))
    lines = (res / "Individuals").read_text().splitlines()
    # header + units + two real rows; the empty record produced no line
    assert len(lines) == 4
    assert "1000.000" in lines[2] or "-1000.000" in lines[2]


def test_only_placeholder_writes_header_only(tmp_path):
    res = _setup(tmp_path, [AttrDict()])
    WriteIndividual(str(res))
    lines = (res / "Individuals").read_text().splitlines()
    assert len(lines) == 2                      # header + units, no rows


def test_row_of_empty_record_is_empty_string():
    assert _row(1, AttrDict()) == ""
    assert _row(1, None) == ""