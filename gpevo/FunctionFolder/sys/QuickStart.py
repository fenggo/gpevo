"""Empty MATLAB-struct-array element factory, mirroring QuickStart.m."""
from __future__ import annotations

from gpevo.runtime import AttrDict


def QuickStart(template=None):
    return AttrDict()
