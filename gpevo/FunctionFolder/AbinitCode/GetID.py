"""Python equivalent of GetID.m."""
from __future__ import annotations


def GetID(some_ind, mtype_list, types_alist):
    # some_ind is 1-based atom index in the concatenated molecular atom list
    count = 0
    for i, mtype in enumerate(mtype_list):
        mol_len = len(types_alist) // len(mtype_list) if len(mtype_list) else 0
        if count + mol_len >= int(some_ind):
            j = int(some_ind) - count
            return i + 1, j
        count += mol_len
    return len(mtype_list), int(some_ind) - count
