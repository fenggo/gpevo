#!/usr/bin/env python
from os import system
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read,write
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase import Atoms
from ase.calculators.lammpslib import LAMMPSlib
from irff.irff_np import IRFF_NP
from irff.md.gulp import write_gulp_in,get_reax_energy


def get_gulp_energy(atoms,libfile='reaxff_nn.lib'):
    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',
                lib=libfile)
    system('gulp<inp-gulp>out')
    e_ = get_reax_energy(fo='out')
    return e_[0]


def trajplot(traj='structures.traj',nn=True,i=0,j=1,tolerance=0.2):
    ffield = 'ffield.json' if nn else 'ffield'
    images         = Trajectory(traj)
    step,eg,ei,e2,e  = [],[],[],[],[]
    e3             = []
    r              = []

    ir = IRFF_NP(atoms=images[0],
              libfile=ffield,
              nn=nn)

    ir.calculate(images[0])
    natom = ir.natom  
    e0    = images[0].get_potential_energy()

    for i_,atoms in enumerate(images):
        if len(atoms) != natom:
           continue
        e_ = atoms.get_potential_energy()
        if abs(e_-e0)>2.0:
           continue
        
        step.append(i_)
        e.append(e_)

        ir.calculate(atoms)
        ei.append(ir.E)
        r.append(ir.r[i][j])

        eg.append(get_gulp_energy(atoms,libfile='reaxff_nn.lib'))
        print(i_, "Energy: ", eg[-1],ir.E,e[-1])

    e_max = min(eg)
    eg = np.array(eg) - e_max
    e_max = min(ei)
    ei    = np.array(ei) - e_max
    e_max = min(e)
    e     = np.array(e) - e_max

    traj_ = TrajectoryWriter('structures_to_train.traj',mode='w')
    for ii,atoms in enumerate(images):
        print(ii,abs(eg[ii]-e[ii]))
        dif = abs(eg[ii]-e[ii])
        if dif>tolerance:
           traj_.write(atoms=atoms)
    traj_.close()
    
    if i==0 and j==0:
       r = step
    plt.figure()   
    plt.ylabel(r'$Energy$ ($eV$)')
    plt.xlabel(r'$Crystal$ $ID$')

    plt.plot(r,eg,alpha=0.8,
             linestyle='-',marker='o',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='k',markersize=10,
             color='k',label=r'$ReaxFF-nn$')
    
    plt.plot(r,e,alpha=0.8,
             linestyle='-',marker='s',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='r',markersize=10,
             color='r',label=r'$DFT(SIESTA$')
    # ediff = np.abs(e - ei)
    # plt.fill_between(r,ei - ediff, ei + ediff, color='palegreen',
    #                  alpha=0.2)

    # pdiff = np.abs(pdft - preax)
    # plt.fill_between(v_, pdft - pdiff, pdft + pdiff, color='palegreen',
    #                  alpha=0.2)

    plt.text( 0.0, 0.5, '%.3f' %e_max, fontdict={'size':10.5, 'color': 'k'})
    plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
    plt.savefig('{:s}'.format(traj.replace('traj','pdf')),transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./tplot.py to run it'''

   parser = argparse.ArgumentParser(description='stretch molecules')
   parser.add_argument('--t', default='structures.traj',type=str, help='trajectory file')
   parser.add_argument('--i', default=0,type=int, help='atom i')
   parser.add_argument('--j', default=0,type=int, help='atom j')
   parser.add_argument('--d', default=0.2,type=float, help='energy difference tolerance that to be trained')
   args = parser.parse_args(sys.argv[1:])
   trajplot(traj=args.t,i=args.i,j=args.j,tolerance=args.d)


