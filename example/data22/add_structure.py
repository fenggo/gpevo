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


def add_structure(atomes_dft,step=1000,tolerance=0.01,ncpu=1):
    # subprocess.call('cp structures.traj structures.traj.backup',shell=True)
    # subprocess.call('cp structures_mlp.traj structures_mlp.traj.backup',shell=True)
    # subprocess.call('cp feature.csv feature.csv.backup',shell=True)
    # subprocess.call('cp feature_mlp.csv feature_mlp.csv.backup',shell=True)
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
    i    = int(data[-1][0])   # 获取索引
    # print(cry)
    feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density_])
    res  = np.sum(np.square(d - feature),axis=1)
    ind  = np.where(res<tolerance)
    print('ID {:4d}: etol {:8.4f} ebond: {:8.4f} eang: {:8.4f} etor: {:8.4f} evdw: {:8.4f} ehb: {:8.4f} ecoul: {:8.4f}'
          ' density: {:9.6}'.format(i,e[0],e[1],e[5],e[8],e[10],e[11],e[12],density))
    if len(ind[0])>0:
       print(f'Structure already in database with index {ind[0]}!') 
       print(f'energy: {d[ind[0],0]}')
    else:
       with open('feature_mlp.csv','a') as fd:
            print(i,',',feature[0],',',feature[1],',',feature[2],',',feature[3],',',
                     feature[4],',',feature[5],',',feature[6],',',feature[7],file=fd) 
       with open('feature.csv','a') as fd:
            print(i,',',energy,',',feature[1],',',feature[2],',',
               feature[3],',',feature[4],',',feature[5],',',feature[6],',',density,file=fd)  
    
       atoms.calc = SinglePointCalculator(atoms,energy=e[0])
       with TrajectoryWriter('structures_mlp.traj',mode='a') as traj:
            traj.write(atoms=atoms)
       with TrajectoryWriter('structures.traj',mode='a') as traj:
            traj.write(atoms=atoms_dft)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='./add_structure.py --t=md.traj')
    parser.add_argument('--t',default='id_354.traj',type=str, help='data file directory')
    parser.add_argument('--n',default=24,type=int, help='number of CPU to be used')
    parser.add_argument('--s',default=1000,type=int, help='total MD step')
    parser.add_argument('--tolerance',default=0.01,type=float, help='tolerance')
    args = parser.parse_args(sys.argv[1:])

    # images      = Trajectory(args.t)
    atoms_dft = read(args.t,-1)
    # for atoms_dft in images:
    add_structure(atoms_dft,step=args.s,tolerance=args.tolerance,ncpu=args.n)

