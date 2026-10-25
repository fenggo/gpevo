#!/usr/bin/env python
import subprocess
from os.path import exists #isfile
from os import system, getcwd,listdir
import sys
import argparse
import pickle
import numpy as np
from ase.io import read
from ase import Atoms
from ase.data import atomic_numbers, atomic_masses
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase
from irff.md.gulp import write_gulp_in,get_reax_energy
from irff.molecule import Molecules,enlarge # SuperCell,moltoatoms
#from irff.md.lammps import writeLammpsData


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
       # print('\n-  running gulp optimize ...')
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
         print(' {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(lengths[0],
                              lengths[1],lengths[2],angles[0],angles[1],angles[2]),file=gf)
         print('fractional  1  ',file=gf)
         for i,x in enumerate(xf):
             print('{:1s}     core {:12.9f} {:12.9f} {:12.9f}    0.0 1.0 0.0'.format(symbols[i],
                                                                      x[0],x[1],x[2]),file=gf)
         print(' ',file=gf)
         print('dump every      1 optimized.structure',file=gf)   


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
                     feature[4],',',feature[5],',',feature[6],',',feature[7],file=fd) 
    with open('../{:s}/feature.csv'.format(data),'a') as fd:
         print(i,',',energy,',',feature[1],',',feature[2],',',
               feature[3],',',feature[4],',',feature[5],',',feature[6],',',density,file=fd)
        

parser  = argparse.ArgumentParser(description='./check_broken.py --g=md.traj')
parser.add_argument('--n',default=1,type=int, help='number of cpu')
parser.add_argument('--data',default='data',type=str, help='data file directory')
parser.add_argument('--b',default=1.5,type=float, help='energy tolerance that structure is broken')
parser.add_argument('--s',default=1.2,type=float, help='scale factor')
args    = parser.parse_args(sys.argv[1:])
broken  = args.b
datafile= args.data
scale_f = args.s
ncpu    = args.n

write_input(inp='inp-grad',keyword='grad conv qiterative verb')
run_gulp(n=ncpu,inp='inp-grad')
e = get_reax_energy(fo='output')
write_output(e=e[0])

atoms  = read('gulp.cif')
# atoms  = opt(atoms=atoms,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')
masses = np.sum(atoms.get_masses())
volume = atoms.get_volume()
density = masses/volume/0.602214129
atoms.calc = SinglePointCalculator(atoms,energy=e[0])

feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])

data   = np.loadtxt('../{:s}/feature_mlp.csv'.format(datafile),delimiter=',',skiprows=1)  ## get crystal feature data
data_  = np.loadtxt('../{:s}/feature.csv'.format(datafile),delimiter=',',skiprows=1)      ## get crystal feature data
images = Trajectory('../{:s}/structures.traj'.format(datafile))
d      = data[:,1:]    # 去掉索引

# Train a Gaussian Process 
# res    = np.sum(np.square(d - feature),axis=1)
# ind    = np.where(res<tolerance)
# imin   = np.argmin(res)
### prepare data 
X_raw  = data[:,1:]
y      = data_[:,-1]
y_eng  = data_[:,1]
# x_mean = np.mean(X,axis=0) 
d_scale= np.mean(y)/np.mean(data[:,-1])
e_mean = np.mean(data[:,1])
e_scale= e_mean - np.mean(y_eng)
# X      = X - x_mean

if e_mean-e[0]>broken:
   if exists("molecule.pkl"):
      with open("molecule.pkl", "rb") as f:
           m_ = pickle.load(f)
      for m in m_:
          for i,na in enumerate(m.mol_index):
              # elems[na] = mol.atom_name[i]
              m.mol_x[i] = atoms.positions[na]
              # print('set positions: ',na)
      for m in m_:
          m.center       = np.sum(m.mol_x,axis=0)/m.natom
       
      nmol    = len(m_)
      cell    = atoms.get_cell()
      _,atoms = enlarge(m_,cell=cell,fac=scale_f,supercell=[1,1,1])
else:
   if not exists("molecule.pkl"):
      m_  = Molecules(atoms,rcut={"H-H":1.0,"H-O":1.02,"O-O":1.4,"H-N":1.22,"H-C":1.35,
                            "others": 1.75},check=True)
      with open("molecule.pkl", "wb") as f:
           pickle.dump(m_, f)
 
energy = e[0]
    
write_output(e=energy)
write_geometry(atoms=atoms)

 