"""Build script for GPEvo's Cython extension.

Metadata (name, version, dependencies, package data) lives in pyproject.toml;
this file only compiles the ``random_cell_mol_cy`` Cython extension, which needs
no numpy headers.  (The uspexkit fingerprint extension was removed — gpevo
computes its own fingerprint and writes it into Individuals.)
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
