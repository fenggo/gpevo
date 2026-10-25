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
parser.add_argument('--t', default='structures.traj',type=str, help='trajectory file')
parser.add_argument('--s', default=1000,type=int, help='step of optimization')
parser.add_argument('--n', default=8,type=int, help='ncpu')
parser.add_argument('--m', default=2,type=int, help='number of molecule')
parser.add_argument('--p', default=0,type=int, help='plot which energy') 
# 0: total 5:eangle  8: etor 10: evdw 11: ehb 12:ecoul
args = parser.parse_args(sys.argv[1:])
p    = args.p
nmol = args.m
step = args.s
n    = args.n
traj = args.t

if p==0:
   lab = 'Etotal'
elif p==5:
   lab = 'Eangle'
elif p==8:
   lab = 'Etorsion'
elif p==9:
   lab = 'Evdw'
elif p==10:
   lab = 'Ehbond'
elif p==11:
   lab = 'Ecoul'  # Coulomb
else:
   lab = 'E'

nframe = None
images         = Trajectory(traj)

if exists('structures_mlp.traj'):
   images_     = Trajectory('structures_mlp.traj')
   nframe      = len(images_)
   if nframe == len(images):
      raise SystemExit(1)
   traj_       = TrajectoryWriter('structures_mlp.traj',mode='a')
else:
   traj_       = TrajectoryWriter('structures_mlp.traj',mode='w')
    
if not exists('feature_mlp.csv'):   
   with open('feature_mlp.csv','w') as fd:
        print(', etot, ebond, eang, etor, evdw, ehb, ecoul, density',file=fd)

for i,atoms in enumerate(images):
    if nframe is not None:
       if i<nframe:
          continue
    atoms = opt(atoms=atoms,step=step,l=1,t=0.000001,n=n, lib='reaxff_nn')
    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',lib='reaxff_nn')
    if n==1:
       subprocess.call('gulp<inp-gulp>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} gulp<inp-gulp>out'.format(n),shell=True)
    e = get_reax_energy(fo='out')
    atoms = read('gulp.cif')
    atoms.calc = SinglePointCalculator(atoms,energy=e[0])
    traj_.write(atoms=atoms)
    # e,ebond,elp,eover,eunder,eang,epen,tconj,etor,fconj,evdw,ehb,ecl,esl
    #  mols,ehb,ev,ec,e = bind_energy(atoms)
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129

    print('ID {:4d}: etol {:8.4f} ebond: {:8.4f} eang: {:8.4f} etor: {:8.4f} evdw: {:8.4f} ehb: {:8.4f} ecoul: {:8.4f}'
          ' density: {:9.6}'.format(i,e[0],e[1],e[5],e[8],e[10],e[11],e[12],density))
    with open('feature_mlp.csv','a') as fd:
         print(i,',',e[0],',',e[1],',',e[5],',',e[8],',',e[10],',',e[11],',',e[12],',',density,file=fd) 
        
traj_.close()

# nohup ./mlp_data.py --n=20 --s=1200>py.log 2>&1 &
