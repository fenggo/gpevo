#!/usr/bin/env python
import subprocess
from os import getcwd, listdir
from os.path import isdir, exists
import sys
import argparse
import numpy as np
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator


parser = argparse.ArgumentParser(description='options')
parser.add_argument('--t', default='structures.traj',type=str, help='trajectory file')
# 0: total 5:eangle  8: etor 10: evdw 11: ehb 12:ecoul
args = parser.parse_args(sys.argv[1:])
traj = args.t

images      = Trajectory(traj)
traj_       = TrajectoryWriter('structures_supercell.traj',mode='w')

for i,atoms in enumerate(images):
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density_ = masses/volume/0.602214129
    energy = atoms.get_potential_energy()

    atoms_ = atoms*(2,1,1)
    atoms_.calc = SinglePointCalculator(atoms_,energy=2*energy)
    traj_.write(atoms=atoms_)

    atoms_ = atoms*(1,2,1)
    atoms_.calc = SinglePointCalculator(atoms_,energy=2*energy)
    traj_.write(atoms=atoms_)

    atoms_ = atoms*(1,1,2)
    atoms_.calc = SinglePointCalculator(atoms_,energy=2*energy)
    traj_.write(atoms=atoms_)
    
traj_.close()

