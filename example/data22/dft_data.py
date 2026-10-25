#!/usr/bin/env python
import subprocess
import sys
import argparse
from os import getcwd,listdir # ,system
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from irff.irff_np import IRFF_NP
from irff.md.gulp import opt,get_reax_energy,write_gulp_in

# from irff.molecule import press_mol

parser = argparse.ArgumentParser(description='options')
parser.add_argument('--r', default='results1',type=str, help='results file label')
parser.add_argument('--n', default=8,type=int, help='ncpu')
# parser.add_argument('--m', default=2,type=int, help='number of molecule')
parser.add_argument('--p', default=0,type=int, help='plot which energy') 
# 0: total 5:eangle  8: etor 10: evdw 11: ehb 12:ecoul
args = parser.parse_args(sys.argv[1:])

p       = args.p
ncpu    = args.n
label   = args.r

cdir    = getcwd()
cdir_   = '/'.join(cdir.split('/')[:-1])
dirs    = listdir(cdir_)
# print(dirs)

if not exists('structures.traj'):
   traj    = TrajectoryWriter('structures.traj',mode='w')
else:
   traj    = TrajectoryWriter('structures.traj',mode='a')

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
    
if not exists('feature.csv'):
   with open('feature.csv','w') as fd:
        print(', etot, ebond, eang, etor, evdw, ehb, ecoul, density',file=fd)
       
D,E = [],[]
ii  = 0
e0  = None
efeature = []

# print(dirs,lab)

for dir_ in dirs:
    d = dir_.split('-')
    # print(d[0],lab)
    if d[0]==label : #and d[1].isalnum(): # d[1].isalpha(): isalnum()
       # print('parsing dir',d[0])
       if exists('{:s}/{:s}/density.log'.format(cdir_,dir_)):
          with open('{:s}/{:s}/density.log'.format(cdir_,dir_),'r') as f:
              for i,line in enumerate(f.readlines()):
                 if i==0:
                    continue
                 l   = line.split()
                 den = float(l[1])
                 id_ = l[0]

                 # if exists('{:s}/{:s}/{:s}/POSCAR.{:s}_opt'.format(cdir_,dir_,id_,id_)):
                 # print('reading {:s}/{:s}/{:s}/id_{:s}.traj ... '.format(cdir_,dir_,id_,id_))
                 if exists('{:s}/{:s}/{:s}/id_{:s}.traj'.format(cdir_,dir_,id_,id_)):
                    atoms = read('{:s}/{:s}/{:s}/id_{:s}.traj'.format(cdir_,dir_,id_,id_))
                    e_ = atoms.get_potential_energy()
                    if e0 is None:
                       e0 = e_
                    elif abs(e_-e0)>1.5:
                       continue
                    
                    #atoms = read('{:s}/{:s}/{:s}/POSCAR.{:s}_opt'.format(cdir_,dir_,id_,id_))
                    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',lib='reaxff_nn')
                    if ncpu==1:
                       subprocess.call('gulp<inp-gulp>out',shell=True)
                    else:
                       subprocess.call('mpirun -n {:d} gulp<inp-gulp>out'.format(ncpu),shell=True)
                    e = get_reax_energy(fo='out')

                    masses = np.sum(atoms.get_masses())
                    volume = atoms.get_volume()
                    density = masses/volume/0.602214129
                    D.append(density)
                    E.append(e[p])

                    ef = [e[0],e[1],e[5],e[8],e[10],e[11],e[12],density]
                    if efeature:
                       ef_   = np.array(ef)
                       efe   = np.array(efeature)
                       resi  = np.sum(np.square(efe-ef_),axis=1)    #  判断是否重复
                       ind   = np.where(resi<0.01)
                       if len(ind[0])==0:
                          efeature.append(ef)    
                       else:
                          continue
                    else:
                       efeature.append(ef)
                    atoms.calc = SinglePointCalculator(atoms,energy=e_)
                    traj.write(atoms=atoms)
                    print('ID {:4d}: etol {:8.4f} ebond: {:8.4f} eang: {:8.4f} etor: {:8.4f} evdw: {:8.4f} ehb: {:8.4f} ecoul: {:8.4f}'
                          ' density: {:9.6}'.format(ii,e_,e[1],e[5],e[8],e[10],e[11],e[12],density))
                    with open('feature.csv','a') as fd:
                         print(ii,',',e_,',',e[1],',',e[5],',',e[8],',',e[10],',',e[11],',',e[12],',',density,file=fd) 
                    ii += 1
traj.close()

plt.figure()   
plt.rcParams['ytick.labelsize'] = 8  
plt.xlabel(r'$Density$ ($g/cm^3$)')
plt.ylabel('{:s} Energy (eV)'.format(lab))
plt.scatter(D,E,alpha=0.8,
            edgecolor='#035f37', s=35,marker='^',
            color='none',
            label=lab)
# x = np.linspace(0.59,0.77)
# y = x*0.67 + 1.41
# plt.plot(x,y,color='k',linestyle='-.')
plt.legend(loc='best',edgecolor='yellowgreen',ncol=3,fontsize=9) # loc = lower left upper right best
plt.savefig('energies.pdf',transparent=True) 
plt.close()
