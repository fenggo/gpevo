#!/usr/bin/env python3
"""Python implementation of the USPEX.m entry point (calculationType=310)."""
import os
import sys

try:
    import numpy  # noqa: F401
except Exception:
    # Prefer the project Python interpreter that has the scientific stack.
    candidates = [
        "/home/feng/.local/anaconda/bin/python",
        sys.executable,
    ]
    for python in candidates:
        if os.path.exists(python):
            os.execv(python, [python, os.path.abspath(__file__)] + sys.argv[1:])
    raise SystemExit("numpy is required")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gpevo.FunctionFolder.USPEX.Start import Start


def main(argv=None):
    Start()


if __name__ == "__main__":
    main()
