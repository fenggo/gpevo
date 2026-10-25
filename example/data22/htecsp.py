#!/usr/bin/env python
import subprocess
from os.path import isfile
from os import system, getcwd,listdir
import sys
import argparse
import numpy as np
from ase.io import read
from ase import Atoms
from ase.data import atomic_numbers, atomic_masses
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase
from irff.md.gulp import write_gulp_in,get_reax_energy
from irff.dft.dftb import dftb_opt
from irff.dft.siesta import siesta_opt #

''' A work flow in combination with USPEX 
    High-Throughput Evolutionary Crystal Structure Prediction Method
'''

def run_gulp(atoms=None,n=1,inp=None,step=200,l=1,p=0,T=300,t=0.0001,lib='reaxff_nn'):
    if inp is not None:
       if n==1:
          subprocess.call('gulp<{:s}>output'.format(inp),shell=True) 
       else:
          subprocess.call('mpirun -n {:d} gulp<{:s}>output'.format(n,inp),shell=True)  # get initial crystal structure
    else:
       if l==1 or p>0.0000001:
          runword= 'opti conp qiterative stre atomic_stress'
       elif l==0:
          runword='opti conv qiterative'
 
       write_gulp_in(atoms,runword=runword,
                  T=T,maxcyc=step,pressure=p,
                  gopt=t,
                  lib=lib)
       print('\n-  running gulp optimize ...')
       if n==1:
          subprocess.call('gulp<inp-gulp>output',shell=True)
       else:
          subprocess.call('mpirun -n {:d} gulp<inp-gulp>output'.format(n),shell=True)
    # xyztotraj('his.xyz',mode='w',traj='md.traj',checkMol=c,scale=False) 
    # atoms = arctotraj('his_3D.arc',traj='md.traj',checkMol=c)

def write_input(inp='inp-grad',keyword='grad nosymmetry conv qiterative'):
    with open('input','r') as f:
      lines = f.readlines()
    with open(inp,'w') as f:
      for i,line in enumerate(lines):
          if i==0 :
             print(keyword,file=f)
          # elif line.find('maxcyc')>=0:
          #    print('maxcyc 0',file=f)
          else:
             print(line.rstrip(),file=f)

def write_output(e=None):
    if e is None:
       with open('output','r') as f:
         for line in f.readlines():
             if line.find('Total lattice energy')>=0 and line.find('eV')>0:
                e = float(line.split()[4])
    with open('output','w') as f:
         print('  Cycle:      0 Energy:       {:f}'.format(e),file=f)

def write_geometry(gen='optimized.gen',atoms=None):
    if atoms is None:
       atoms = read(gen)
    cell = atoms.get_cell()
    angles = cell.angles()
    lengths = cell.lengths()
    cell = cell[:].astype(dtype=np.float32)
    rcell     = np.linalg.inv(cell).astype(dtype=np.float32)
    positions = atoms.get_positions()
    xf        = np.dot(positions,rcell)
    xf        = np.mod(xf,1.0)
    symbols = atoms.get_chemical_symbols()

    with open('optimized.structure','w') as gf:
         print('opti nosymmetry conp qiterative conjugate  ',file=gf)
         print(' ',file=gf)
         print('cell  ',file=gf)
         #   6.80240161   5.69664152   5.91581126  99.91236580 104.21459462 103.96779224   
         print(' {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(lengths[0],
                              lengths[1],lengths[2],angles[0],angles[1],angles[2]),file=gf)
         print('fractional  1  ',file=gf)
         for i,x in enumerate(xf):
             print('{:1s}     core {:12.9f} {:12.9f} {:12.9f}    0.0 1.0 0.0'.format(symbols[i],
                                                                      x[0],x[1],x[2]),file=gf)
         print(' ',file=gf)
         print('dump every      1 optimized.structure',file=gf)   

def optimize(atoms,calc=2,ncpu=8,step=1000):
   if calc==1: # dftb
      dftb_opt(atoms=atoms,step=step,skf_dir='./')
      output = subprocess.check_output('grep \'Total Energy:\' dftb.out | tail -1',shell=True)
      e = float(output.split()[-2])
      write_output(e=e)
      write_geometry(gen='dftb.gen')
   elif calc==2: # siesta
      img = siesta_opt(atoms,ncpu=ncpu,us='F',VariableCell='true',tstep=step,
                       xcf='GGA',xca='PBE',basistype='split')
                       # xcf='VDW',xca='DRSLL',basistype='split')
      atoms = img[-1]
      subprocess.call('rm siesta.* *.xml INPUT_TMP.* fdf-*',shell=True)
      energy  = atoms.get_potential_energy()
      write_output(e=energy)
      write_geometry(atoms=atoms)
   elif calc==0:
      run_gulp(n=ncpu,atoms=atoms,l=1,step=step)
   else:
      print('calculator not supported!')
      raise SystemExit(1)
   return atoms 

def add_structure(i,atomes_dft,atoms_mlp,feature=None,data=None):
    with TrajectoryWriter('../{:s}/structures_mlp.traj'.format(data),mode='a') as traj:
         traj.write(atoms=atoms_mlp)
    with TrajectoryWriter('../{:s}/structures.traj'.format(data),mode='a') as traj_:
         traj_.write(atoms=atoms_dft)

    masses  = np.sum(atoms_dft.get_masses())
    volume  = atoms_dft.get_volume()
    density = masses/volume/0.602214129
    energy  = atoms_dft.get_potential_energy()

    with open('../{:s}/feature_mlp.csv'.format(data),'a') as fd:
         print(i,',',feature[0],',',feature[1],',',feature[2],',',feature[3],',',
                     feature[4],',',feature[5],',',feature[6],',',feature[7]，file=fd) 
    with open('../{:s}/feature.csv'.format(data),'a') as fd:
         print(i,',',energy,',',feature[1],',',feature[2],',',
               feature[3],',',feature[4],',',feature[5],',',feature[6]，',',density,file=fd)
        

parser = argparse.ArgumentParser(description='./htecsp.py --g=md.traj')
parser.add_argument('--c',default=2,type=int, help='calculator')
parser.add_argument('--t',default=0.005,type=float, help='tolerance')
parser.add_argument('--d',default=1.80,type=float, help='density tolerance')
parser.add_argument('--n',default=1,type=int, help='the number of cpu used in this calculation')
parser.add_argument('--g',default='gulp.cif',type=str, help='geometry file')
parser.add_argument('--x',default=1,type=int, help='X')
parser.add_argument('--y',default=1,type=int, help='Y')
parser.add_argument('--z',default=1,type=int, help='Z')
parser.add_argument('--step',default=5000,type=int, help='Time Step')
parser.add_argument('--struc',default='mlp',type=str, help='returen structure with dft/mlp optimization')
parser.add_argument('--data',default='data',type=str, help='data file directory')
args = parser.parse_args(sys.argv[1:])

tolerance = args.t
datafile  = args.data
step      = args.step
ncpu      = args.n
calc      = args.c
gen       = args.g
den       = args.d
struc     = args.struc

write_input(inp='inp-grad',keyword='grad conv qiterative verb')
run_gulp(n=ncpu,inp='inp-grad')
e = get_reax_energy(fo='output')
write_output(e=e[0])

atoms = read(gen)
masses = np.sum(atoms.get_masses())
volume = atoms.get_volume()
density = masses/volume/0.602214129
atoms.calc = SinglePointCalculator(atoms,energy=e[0])

feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])
# python htecsp.py  --n=20 --step=500 --d=1.72
# python htecsp.py  --b=1  --step=500

data = np.loadtxt('../{:s}/feature_mlp.csv'.format(datafile),delimiter=',',skiprows=1)  ## get crystal feature data
data_= np.loadtxt('../{:s}/feature.csv'.format(datafile),delimiter=',',skiprows=1)      ## get crystal feature data
images = Trajectory('../{:s}/structures.traj'.format(datafile))
d    = data[:,1:]    # 去掉索引
# print(cry)
res  = np.sum(np.square(d - feature),axis=1)
ind  = np.where(res<tolerance)

if len(ind[0])>0:
   im = np.argmin(res[ind])
   # print(ind[0][im])
   # print(res[ind[0][im]])
   density_= data_[ind[0][im],8]
   energy  = data_[ind[0][im],1]
   write_output(e=energy)
   cell = atoms.get_cell()
   cell = cell*np.power(density/density_,1/3)
   atoms.set_cell(cell)
   if struc=='mlp': 
      write_geometry(atoms=atoms)
   else:
      write_geometry(atoms=images[ind[0][im]])
else:
   if density<den:
      atoms_dft = optimize(atoms,calc=calc,ncpu=ncpu,step=20)
    
      volume  = atoms_dft.get_volume()
      density = masses/volume/0.602214129
      if density>=den:
         atoms_dft = optimize(atoms,calc=calc,ncpu=ncpu,step=step)
   else:
      atoms_dft = optimize(atoms,calc=calc,ncpu=ncpu,step=step)
 
   add_structure(len(d),atoms_dft,atoms,feature=feature,data=datafile)
