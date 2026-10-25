#!/usr/bin/env python
import subprocess
from os import getcwd, listdir
from os.path import isdir, exists
import sys
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read 
from irff.irff_np import IRFF_NP
# from irff.molecule import press_mol,Molecules,enlarge # , moltoatoms
from irff.md.gulp import opt,get_reax_energy,write_gulp_in
from irff.deb.compare_energies import deb_gulp_energy

parser = argparse.ArgumentParser(description='options')
parser.add_argument('--s', default=1000,type=int, help='step of optimization')
parser.add_argument('--n', default=8,type=int, help='ncpu')
# 0: total 5:eangle  8: etor 10: evdw 11: ehb 12:ecoul
args = parser.parse_args(sys.argv[1:])
# nmol = args.m
step = args.s
n    = args.n

nframe = None
images         = Trajectory('structures.traj')
images_        = Trajectory('structures_mlp.traj')
    
data = np.loadtxt('feature.csv',delimiter=',',skiprows=1)  ## get crystal feature data
data_= np.loadtxt('feature_mlp.csv',delimiter=',',skiprows=1)  ## get crystal feature data

traj_          = TrajectoryWriter('structures_mlp.traj',mode='w')

fd = open('feature_mlp.csv','w') 
print(', etot, eang, etor, evdw, ehb, ecoul, density',file=fd)

e0 = data_[0][1]

for i,atoms in enumerate(images):
    fea = data_[i]
    if abs(data_[i][1]-e0)>1.0:
      
       atoms = opt(atoms=atoms,step=step,l=1,t=0.000001,n=n, lib='reaxff_nn')
       write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',lib='reaxff_nn')
       if n==1:
          subprocess.call('gulp<inp-gulp>out',shell=True)
       else:
          subprocess.call('mpirun -n {:d} gulp<inp-gulp>out'.format(n),shell=True)
       e = get_reax_energy(fo='out')
       # atoms = read('gulp.cif')
       atoms.calc = SinglePointCalculator(atoms,energy=e[0])
       masses = np.sum(atoms.get_masses())
       volume = atoms.get_volume()
       density = masses/volume/0.602214129
       print('ID {:4d}: etol {:8.4f} ebond: {:8.4f} eang: {:8.4f} etor: {:8.4f} evdw: {:8.4f} ehb: {:8.4f} ecoul: {:8.4f}'
             ' density: {:9.6}'.format(i,e[0],e[1],e[5],e[8],e[10],e[11],e[12],density))
       print(i,',',e[0],',',e[1],',',e[5],',',e[8],',',e[10],',',e[11],',',e[12],',',density,file=fd) 
    else:
       print(i,',',fea[1],',',fea[2],',',fea[3],',',fea[4],',',fea[5],',',fea[6],',',fea[7],',',fea[8],file=fd) 

    traj_.write(atoms=atoms)
      
    # e,ebond,elp,eover,eunder,eang,epen,tconj,etor,fconj,evdw,ehb,ecl,esl
    #  mols,ehb,ev,ec,e = bind_energy(atoms)

fd.close()
