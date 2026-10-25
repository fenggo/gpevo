#!/usr/bin/env python
import subprocess
import sys
import argparse
import numpy as np
from ase.io import read
from ase.data import atomic_numbers, atomic_masses
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.md.gulp import opt,get_reax_energy,write_gulp_in
from irff.dft.siesta import siesta_opt #


def update_structure(atomes_dft,inde=None,step=1000,tolerance=0.01,ncpu=1):
    subprocess.call('cp structures.traj structures.backup.traj',shell=True)
    masses  = np.sum(atoms_dft.get_masses())
    volume  = atoms_dft.get_volume()
    density = masses/volume/0.602214129
    energy  = atoms_dft.get_potential_energy()

    atoms = opt(atoms=atoms_dft,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')              ## compute feature
    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',lib='reaxff_nn')   ## compute feature
    if ncpu==1:
       subprocess.call('gulp<inp-gulp>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} gulp<inp-gulp>out'.format(ncpu),shell=True)         ## compute feature
    e = get_reax_energy(fo='out')

    volume   = atoms.get_volume()
    density_ = masses/volume/0.602214129
    
    data = np.loadtxt('feature_mlp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
    data_= np.loadtxt('feature.csv',delimiter=',',skiprows=1)          ## get crystal feature data
    d    = data[:,1:]         # 去掉索引
    d_   = data_[:,1:]   
     
    images     = Trajectory('structures.backup.traj')
    # images_  = Trajectory('structures_mlp.traj')
    
    feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density_])
    res  = np.sum(np.square(d - feature),axis=1)
    ind  = np.where(res<tolerance)
    if inde is None:
       ind_ = ind[0]
    else:
       ind_ = [inde]
    if len(ind_)>0:
       traj  = TrajectoryWriter('structures.traj',mode='w')
       fd    = open('feature.csv','w') 
       print(', etot, eang, etor, evdw, ehb, ecoul, density',file=fd)
       for i,d in enumerate(d_):
           if i in ind_:
              print(i,data_[i][1],energy,data_[i][7],density)
              print(i,',',energy,',',d[1],',',d[2],',',d[3],',',d[4],',',d[5],',',d[6],',',density,file=fd)  
              traj.write(atoms=atoms_dft)
           else:  
              print(i,',',d[0],',',d[1],',',d[2],',',d[3],',',d[4],',',d[5],',',d[6],d[7],',',file=fd)  
              traj.write(atoms=images[i])
       fd.close()
       traj.close()
    else:
       print(f'Specified structure not found in database!') 
       print(f'energy: {energy}')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='./add_structure.py --t=md.traj')
    parser.add_argument('--t',default='id_354.traj',type=str, help='data file directory')
    parser.add_argument('--n',default=8,type=int, help='number of CPU to be used')
    parser.add_argument('--s',default=1000,type=int, help='total MD step')
    parser.add_argument('--tolerance',default=0.01,type=float, help='tolerance')
    parser.add_argument('--i',default=None,type=int, help='the ID of structure to be updated')
    args = parser.parse_args(sys.argv[1:])

    atoms_dft = read(args.t,-1)
    update_structure(atoms_dft,inde=args.i,step=args.s,tolerance=args.tolerance,ncpu=args.n)

