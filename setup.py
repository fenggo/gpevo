"""Build script for GPEvo's Cython extensions.

Metadata (name, version, dependencies, package data) lives in pyproject.toml;
this file compiles the four ACTIVE extensions (random_cell_mol_cy,
molcheck_cy, bondtable_cy and rotation_kernels_cy).  None needs numpy headers:
they use the buffer protocol via typed memoryviews.

FunctionFolder/sys/uspex_fast_core.pyx is deliberately NOT built here.  It is
a dormant Octave-era fingerprint accelerator: gpevo computes its own
fingerprint in fingerprint_utils.py (pure numpy) and writes it into
Individuals.  Building it would activate an unused fast path whose erf
approximation differs from scipy's by ~1e-6.  See docs/fingerprint_method.md.
"""

import os

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


def _ext(name, pyx, c_fallback):
    if HAS_CYTHON and os.path.exists(pyx):
        sources = [pyx]
    elif os.path.exists(c_fallback):
        sources = [c_fallback]
    else:
        return None
    return Extension(name, sources=sources, extra_compile_args=["-O3"])


extensions = [
    _ext(
        "gpevo.FunctionFolder.spacegroup.random_cell_mol_cy",
        "gpevo/FunctionFolder/spacegroup/random_cell_mol_cy.pyx",
        "gpevo/FunctionFolder/spacegroup/random_cell_mol_cy.c",
    ),
    _ext(
        "gpevo.FunctionFolder.sys.molcheck_cy",
        "gpevo/FunctionFolder/sys/molcheck_cy.pyx",
        "gpevo/FunctionFolder/sys/molcheck_cy.c",
    ),
    _ext(
        "gpevo.FunctionFolder.sys.bondtable_cy",
        "gpevo/FunctionFolder/sys/bondtable_cy.pyx",
        "gpevo/FunctionFolder/sys/bondtable_cy.c",
    ),
    _ext(
        "gpevo.FunctionFolder.sys.rotation_kernels_cy",
        "gpevo/FunctionFolder/sys/rotation_kernels_cy.pyx",
        "gpevo/FunctionFolder/sys/rotation_kernels_cy.c",
    ),
]
extensions = [e for e in extensions if e is not None]

if HAS_CYTHON:
    extensions = cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )

setup(ext_modules=extensions)
