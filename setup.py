"""Build script for GPEvo's Cython extensions.

Metadata (name, version, dependencies, package data) lives in pyproject.toml;
this file only adds the compiled extensions, because they need numpy's include
path resolved at build time — something the static pyproject ext-modules table
cannot express portably.

Two extensions:
  * gpevo.FunctionFolder.spacegroup.random_cell_mol_cy  — USPEX random_cell_mol
  * gpevo.uspexkit.uspex_fast_core                    — vendored uspexkit fp core
"""

import os

from setuptools import Extension, setup

try:
    import numpy as np
    NUMPY_INCLUDE = np.get_include()
except ImportError:
    NUMPY_INCLUDE = os.path.join(
        os.path.dirname(__file__), "build", "numpy_include"
    )

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
    return Extension(
        name,
        sources=sources,
        include_dirs=[NUMPY_INCLUDE],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3"],
    )


extensions = [
    _ext(
        "gpevo.FunctionFolder.spacegroup.random_cell_mol_cy",
        "gpevo/FunctionFolder/spacegroup/random_cell_mol_cy.pyx",
        "gpevo/FunctionFolder/spacegroup/random_cell_mol_cy.c",
    ),
    _ext(
        "gpevo.uspexkit.uspex_fast_core",
        "gpevo/uspexkit/uspex_fast_core.pyx",
        "gpevo/uspexkit/uspex_fast_core.c",
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
