"""Python port of ``sys/createHeader.m`` / ``createHeader_wrap.m``.

Writes the USPEX banner, the "Output written" stamp and the citation block at
the top of ``OUTPUT.txt``, as ``createORGStruc.m`` does.
"""
from __future__ import annotations

import datetime
import os

WIDTH = 80

# FunctionFolder/USPEX_pic.txt
USPEX_PIC = [
    "                                                         ",
    "   ____ _____ ____ _____   _                            ",
    "  / ___| ____/ ___|_   _| / \\                           ",
    " | |  _|  _|| |     | |  / _ \\                          ",
    " | |_| | |__| |___  | | / ___ \\                         ",
    "  \\____|_____\\____| |_|/_/   \\_\\                        ",
    "                                                         ",
]
VERSION = "Version 0.1.0"
DESCRIPTION = "Evolutionary Algorithm Code for Structure Prediction"
URL = "more info at https://gitee.com/fenggo/gpevo"

CITATION_HEAD = [
    "Please cite the following suggested papers",
    "when you publish the results obtained from USPEX:",
]
CITATION_BODY = [
    "Oganov A.R., Glass C.W. (2006)",
    "Crystal structure prediction using evolutionary algorithms:",
    "Principles and applications. J. Chem. Phys. 124, 244704",
    "",
    "Oganov A.R., Stokes H., Valle M. (2011)",
    "How evolutionary crystal structure prediction works - and why.",
    "Acc. Chem. Res. 44, 227-237",
    "",
    "Lyakhov A.O., Oganov A.R., Stokes H., Zhu Q. (2013)",
    "New developments in evolutionary structure prediction algorithm USPEX.",
    "Comp. Phys. Comm., 184, 1172-1182",
]


def _separator():
    return "*" + "-" * (WIDTH - 2) + "*"


def _wrap(text_lines, position="center"):
    rows = [_separator()]
    for text in text_lines:
        text = str(text)
        before = 2 if position != "center" else max(0, (WIDTH // 2 - len(text) // 2 - 1))
        after = WIDTH - (1 + before + len(text)) - 1
        rows.append("|" + " " * before + text + " " * max(0, after) + "|")
    rows.append(_separator())
    return rows


def createHeader():
    """Banner block: picture, version, description, URL."""
    return _wrap(list(USPEX_PIC) + ["", VERSION, "", DESCRIPTION, URL, ""])


def createHeader_wrap(text_lines, position="center"):
    return _wrap(text_lines, position)


def write_output_header(log_path):
    """Create/overwrite ``OUTPUT.txt`` with the standard USPEX header."""
    directory = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    rows.extend(createHeader())
    rows.extend(createHeader_wrap([f"Output written {stamp}"]))
    rows.append("")
    rows.extend(createHeader_wrap(CITATION_HEAD))
    rows.extend(createHeader_wrap(CITATION_BODY, "left"))
    rows.append("")
    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(rows) + "\n")
    return log_path
