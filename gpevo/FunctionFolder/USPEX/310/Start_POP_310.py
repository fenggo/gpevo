"""Python equivalent of Start_POP_310.m."""
from __future__ import annotations

import numpy as np

from gpevo.runtime import G, AttrDict
from gpevo.FunctionFolder.sys.ReRank import ReRank


def Start_POP_310():
    ORG = G["ORG_STRUC"]
    POP = G["POP_STRUC"]
    n_steps = len(ORG.abinitioCode)
    POP.DoneOrder = [0] * len(POP.POPULATION)

    for ind, s in enumerate(POP.POPULATION):
        if not hasattr(s, "Step") or s.Step is None or (
            isinstance(s.Step, list) and len(s.Step) == 0
        ):
            s.Step = 1
            s.Enthalpies = np.array([100000.0] * n_steps)
            s.K_POINTS = np.ones((n_steps, 3))
        s.Error = 0
        s.Folder = 0
        s.ToDo = 1
        s.Done = 0
        s.Number = 0
        if getattr(s, "symg", None) is None or str(getattr(s, "symg", "")).strip().upper() == "N/A":
            s.symg = 1

        num_mols = [int(x) for x in s.numMols]
        catted = []
        for j in range(sum(num_mols)):
            catted.append(np.asarray(s.MOLECULES[j].MOLCOORS, dtype=float).reshape(-1, 3))
        if not catted:
            catted = np.zeros((0, 3))
        else:
            catted = np.concatenate(catted, axis=0)
        lat = np.asarray(s.LATTICE, dtype=float).reshape(3, 3)
        s.COORDINATES = catted @ np.linalg.inv(lat)

        new_coords = None
        for atom_id in ORG.atomType:
            mask = [i for i, t in enumerate(s.typesAList) if t == atom_id]
            if mask:
                block = s.COORDINATES[mask, :]
                new_coords = block if new_coords is None else np.concatenate([new_coords, block], axis=0)
        if new_coords is None:
            new_coords = s.COORDINATES
        s.COORDINATES = new_coords
        s.INIT_COORD = np.asarray(s.COORDINATES).copy()
        s.INIT_LAT = np.asarray(s.LATTICE).copy()
        s.INIT_numIons = list(s.numIons)

    ReRank()
    G["POP_STRUC"] = POP
