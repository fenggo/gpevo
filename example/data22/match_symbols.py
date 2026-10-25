#!/usr/bin/env python
import subprocess
from ase.io import read
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance


def check_symbols(atoms,atoms_):
    consistance = True
    if len(atoms)==len(atoms_):
       for i,atom in enumerate(atoms):
           atom_ = atoms_[i]
           if atom.symbol != atom_.symbol:
              consistance = False
              break
    else:
       consistance = False
    return consistance


ind = [0, 1, 2, 3, 4, 20, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 41, 39, 38, 40, 22, 21, 23, 56, 34, 32, 33, 30, 31, 29, 27, 28, 25, 26, 24, 53, 54, 55, 42, 43, 44, 45, 49, 50, 51, 52, 46, 47, 48, 35, 36, 37] 
ind_= [13, 31, 0, 14, 1, 50, 2, 3, 4, 5, 46, 32, 15, 16, 6, 47, 48, 49, 33, 17, 18, 22, 39, 38, 21, 7, 51, 8, 56, 36, 12, 55, 11, 54, 35, 10, 53, 9, 52, 34, 45, 29, 30, 40, 41, 23, 24, 43, 44, 27, 28, 42, 25, 26, 37, 19, 20]


subprocess.call('cp structures.traj structures.backup.traj',shell=True)
images = Trajectory('structures.backup.traj')
atoms_ = images[0]
natom  = len(atoms_)
# elem   = atoms.get_chemical_symbols()
# elem_  = atoms_.get_chemical_symbols()
traj       = TrajectoryWriter('structures.traj',mode='w')

for i,atoms in enumerate(images):
    consistance = check_symbols(atoms,atoms_)
    # print(i,consistance)
    
    if consistance:
       traj.write(atoms=atoms)
    else:  
       # print(i,',',d[0],',',d[1],',',d[2],',',d[3],',',d[4],',',d[5],',',d[6],file=fd)  
       # ositions = atoms.posictions
       positions  = [None for i in range(natom)]
       positions_ = []
       elems_     = []
        
       for i,i_ in zip(ind,ind_):
           positions[i_] = atoms.positions[i]
       elem = atoms_.get_chemical_symbols()

       positions_.extend(positions)
       elems_.extend(elem)

       A = Atoms(elems_, positions_)
       A.set_cell(atoms.get_cell())
       A.set_pbc([True,True,True])
       A.calc = SinglePointCalculator(A,energy=atoms.get_potential_energy())
       # A.write('POSCAR.mat')
       traj.write(atoms=A)
    
traj.close()


# natom  = len(atoms)
# cell   = atoms.get_cell()

# positions_ = []
# elems_     = []

# positions = [None for i in range(natom)]
  
# for i,i_ in zip(ind,ind_):
#     positions[i_] = atoms.positions[i]
# elem = atoms_.get_chemical_symbols()

# positions_.extend(positions)
# elems_.extend(elem)

# A = Atoms(elems_, positions_)
# A.set_cell(cell)
# A.set_pbc([True,True,True])

# A.write('POSCAR.mat')

