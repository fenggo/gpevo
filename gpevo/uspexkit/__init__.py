"""USPeX Kit — USPEX crystal structure prediction post-processing toolkit."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "gp":
        from .core import gp as _gp
        return _gp
    if name == "calcdata":
        from .core import calcdata as _calcdata
        return _calcdata
    if name == "fixbroken":
        from .core import fixbroken as _fixbroken
        return _fixbroken
    if name == "add":
        from .core import add as _add
        return _add
    if name == "addall":
        from .core import addall as _addall
        return _addall
    if name == "supercell":
        from .core import supercell as _supercell
        return _supercell
    if name == "update":
        from .core import update as _update
        return _update
    if name == "pack":
        from .core import pack as _pack
        return _pack
    if name == "denevo":
        from .denevo import denevo as _denevo
        return _denevo
    if name == "smd":
        from .smd import smd as _smd
        return _smd
    if name == "pred":
        from .core import pred as _pred
        return _pred
    if name == "calc":
        from .core import calc as _calc
        return _calc
    if name == "traj":
        from .core import traj as _traj
        return _traj
    if name == "zmat":
        from .core import zmat as _zmat
        return _zmat
    if name == "fdf":
        from .core import fdf as _fdf
        return _fdf
    if name == "sample":
        from .core import sample as _sample
        return _sample
    if name == "info":
        from .core import info as _info
        return _info
    if name == "lib":
        from .core import lib as _lib
        return _lib
    if name == "ffield":
        from .core import ffield as _ffield
        return _ffield
    if name == "molinfo":
        from .core import molinfo as _molinfo
        return _molinfo
    if name == "md2pdf":
        from .md2pdf import md2pdf as _md2pdf
        return _md2pdf
    raise AttributeError(f"module 'uspexkit' has no attribute {name!r}")
