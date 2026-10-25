"""Python port of ``FunctionFolder/USPEX/src/SameBest.m``.

Counts how many of the previous ``stopCrit - 1`` generations had the same best
structure as the current one (compared through ``USPEX_STRUC``, i.e. by global
structure ``Number``).
"""
from __future__ import annotations

from gpevo.runtime import G


def SameBest():
    USPEX = G.get("USPEX_STRUC")
    ORG = G["ORG_STRUC"]
    if USPEX is None:
        return 0

    generations = list(getattr(USPEX, "GENERATION", []) or [])
    n_gen = len(generations)
    n_stop = int(getattr(ORG, "stopCrit", 0) or 0)
    if n_gen < 2 or n_stop <= 1:
        return 0

    try:
        from gpevo.FunctionFolder.sys.SameStructure import SameStructure
    except Exception:
        return 0

    best_ids = [int(v) for v in (generations[n_gen - 1].get("BestID", []) or []) if int(v) > 0]
    if not best_ids:
        return 0
    id1 = best_ids[0] - 1

    same_in = 0
    for i in range(max(0, n_gen - n_stop), n_gen - 1):
        other = [int(v) for v in (generations[i].get("BestID", []) or []) if int(v) > 0]
        if not other:
            continue
        try:
            if SameStructure(id1, other[0] - 1, USPEX):
                same_in += 1
        except Exception:
            continue
    return same_in
