"""Operator fractions must mirror createORG_EA.m / update_STUFF.m.

Regression for the fixed-composition 3D case: transmutation has no operator in
EA_310, so ``fracTrans`` must be zero there, otherwise update_STUFF_old counts
phantom offspring and starves latmutation (which only gets the leftover slots).
"""
from __future__ import annotations

import pytest

from gpevo.runtime import G, struct


def _org(tmp_path, varcomp=0, dimension=3, frac_trans_line=""):
    from gpevo.FunctionFolder.USPEX.src.createORGDefault import createORGDefault

    inp = tmp_path / "INPUT.txt"
    inp.write_text(
        "100 : populationSize\n0.2 : fracGene\n0.1 : fracRand\n"
        "0.1 : fracAtomsMut\n0.2 : fracRotMut\n0.1 : fracLatMut\n"
        "0.02 : fracPerm\n" + frac_trans_line
    )
    ORG = createORGDefault()
    ORG.atomType = [6, 8, 7, 1]
    ORG.numIons = [56, 48, 24, 40]
    ORG.numMols = [8]
    ORG.molecule = 0            # avoids needing STDMOL for GetPOP_MOL
    ORG.varcomp = varcomp
    ORG.dimension = dimension
    ORG.constLattice = 0
    ORG.homePath = str(tmp_path)
    ORG.resFolder = str(tmp_path)
    G["ORG_STRUC"] = ORG
    return ORG, str(inp)


def _ea(tmp_path, **kw):
    from gpevo.FunctionFolder.USPEX.src.createORG_EA import createORG_EA
    ORG, inp = _org(tmp_path, **kw)
    createORG_EA(inp)
    return ORG


def test_frac_trans_is_zero_for_fixed_composition_3d(tmp_path):
    ORG = _ea(tmp_path)
    assert ORG.fracTrans == 0.0
    assert ORG.howManyTrans == 0


def test_frac_trans_is_kept_for_variable_composition(tmp_path):
    ORG = _ea(tmp_path, varcomp=1, frac_trans_line="0.1 : fracTrans\n")
    assert ORG.fracTrans == pytest.approx(0.1)
    assert ORG.howManyTrans == 10


def test_frac_trans_is_kept_for_2d(tmp_path):
    ORG = _ea(tmp_path, dimension=2, frac_trans_line="0.1 : fracTrans\n")
    assert ORG.fracTrans == pytest.approx(0.1)


def test_frac_trans_from_input_is_ignored_in_3d_fixed(tmp_path):
    ORG = _ea(tmp_path, frac_trans_line="0.1 : fracTrans\n")
    assert ORG.fracTrans == 0.0


def test_update_stuff_logs_fractions_and_effective_quotas(tmp_path, monkeypatch):
    from gpevo.FunctionFolder.USPEX.src.update_STUFF import update_STUFF
    import gpevo.log_utils as log_utils

    ORG, inp = _org(tmp_path)
    POP = struct(generation=1, ranking=list(range(100)), bad_rank=0,
                 POPULATION=[struct(howCome="Random", Number=i + 1)
                             for i in range(100)])
    G["POP_STRUC"] = POP
    lines = []
    monkeypatch.setattr(log_utils, "write_log", lambda text: lines.append(str(text)))
    monkeypatch.chdir(tmp_path)

    from gpevo.FunctionFolder.USPEX.src.update_STUFF_old import update_STUFF_old
    update_STUFF_old(inp)          # set howMany* the way the static branch does
    update_STUFF(inp, goodFrac=0.7, ranking=POP.ranking)

    text = "".join(lines)
    assert "VARIATION OPERATORS" in text
    assert "transmutation" in text
    assert "structures produced by latmutation" in text
    assert ORG.fracTrans == 0.0
    assert ORG.howManyMutations > 0   # latmutation is no longer starved
