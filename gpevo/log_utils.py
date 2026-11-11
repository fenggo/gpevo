"""Output helpers: tee terminal messages into USPEX's ``OUTPUT.txt`` log."""
from __future__ import annotations

import builtins
import contextlib
import io
import os
import sys

_ORIGINAL_PRINT = builtins.print
_LOG_FP = None


def configure_log(path: str) -> None:
    """Open the main log and make every ``print`` also write to it."""
    global _LOG_FP
    close_log()
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _LOG_FP = open(path, "a", encoding="utf-8", errors="replace")
    builtins.print = _tee_print


def console_print(*args, **kwargs):
    """Print to the terminal only, bypassing the OUTPUT.txt tee.

    This is what MATLAB's ``disp`` does: the message shows up in the session
    but never in the log file, which only receives ``fprintf(fp, ...)`` output.
    """
    kwargs.setdefault("file", sys.stdout)
    _ORIGINAL_PRINT(*args, **kwargs)


def tee_print(*args, sep=" ", end="\n", **kwargs):
    """Print to the terminal AND to the active USPEX log (OUTPUT.txt).

    Unlike ``console_print`` (MATLAB ``disp`` semantics, terminal only), this
    is for Python-side diagnostics that must also be archived in the log.
    Writes straight to ``_LOG_FP``, so it is unaffected by ``console_only()``
    and degrades to a plain terminal print before ``configure_log()`` runs.
    """
    if _LOG_FP is not None:
        _LOG_FP.write(sep.join(str(a) for a in args) + end)
        _LOG_FP.flush()
    kwargs.setdefault("file", sys.stdout)
    _ORIGINAL_PRINT(*args, sep=sep, end=end, **kwargs)


@contextlib.contextmanager
def console_only():
    """Context manager: ``print`` inside it reaches the terminal only.

    Used when third-party code is called in-process instead of as a
    subprocess: a child's stdout never reaches OUTPUT.txt, so neither should
    the in-process equivalent.
    """
    original = builtins.print
    builtins.print = _ORIGINAL_PRINT
    try:
        yield
    finally:
        builtins.print = original


def write_log(text: str) -> None:
    """Write a message only to the active USPEX log file."""
    if _LOG_FP is not None:
        _LOG_FP.write(str(text))
        if not str(text).endswith("\n"):
            _LOG_FP.write("\n")
        _LOG_FP.flush()


def _tee_print(*args, sep=" ", end="\n", file=None, flush=False):
    if file is not None and file is not sys.stdout:
        _ORIGINAL_PRINT(*args, sep=sep, end=end, file=file, flush=flush)
        return

    if _LOG_FP is not None:
        buffer = io.StringIO()
        _ORIGINAL_PRINT(*args, sep=sep, end=end, file=buffer, flush=True)
        _LOG_FP.write(buffer.getvalue())
        _LOG_FP.flush()

    _ORIGINAL_PRINT(*args, sep=sep, end=end, file=sys.stdout, flush=flush)


def close_log() -> None:
    """Close the log and restore the normal ``print`` behaviour."""
    global _LOG_FP
    if _LOG_FP is not None:
        _LOG_FP.close()
        _LOG_FP = None
    builtins.print = _ORIGINAL_PRINT
