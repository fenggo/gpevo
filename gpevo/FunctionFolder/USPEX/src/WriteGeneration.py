"""Python port of WriteGeneration.m.

Appends the quasi-entropy (and, for variable-composition runs, the composition
entropy) of the finished generation to ``OUTPUT.txt``.
"""
from __future__ import annotations

from gpevo.runtime import G
from gpevo.FunctionFolder.sys.alignLine import alignLine


def WriteGeneration(resFolder):
    USPEX = G.get("USPEX_STRUC")
    if USPEX is None:
        return
    generations = list(getattr(USPEX, "GENERATION", []) or [])
    if not generations:
        return
    entry = generations[-1]
    lines = [alignLine("Quasi entropy = %.4f" % float(entry.get("quasiEntropy", 0.0) or 0.0))]
    compos = entry.get("composEntropy", None)
    if compos is not None:
        lines.append(alignLine("Composition entropy = %.4f" % float(compos)))
    try:
        with open(f"{resFolder}/OUTPUT.txt", "a", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")
    except OSError:
        return
