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

def nvt(atoms,T=350,tdump=100,timestep=0.1,step=100,gen='poscar.gen',i=-1,c=0,
        free=' ',dump_interval=10,
        x=1,y=1,z=1,n=1,lib='ffield',thermo_fix=None,
        r=0):
    # atoms = read(gen,index=i)*(x,y,z)
    atoms = atoms*(x,y,z)
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp      = ' '.join(species)
    freeatoms = free.split()
    freeatoms = [int(i)+1 for i in freeatoms]
    masses    = {s:atomic_masses[atomic_numbers[s]] for s in species }

    pair_style = 'reaxff control nn yes checkqeq yes'   # without lg set lgvdw no
    pair_coeff = '* * {:s} {:s}'.format(lib,sp)
    units      = "real"
    atom_style = 'charge'
 
    if thermo_fix is None:
       thermo_fix = 'fix   1 all nvt temp {:f} {:f} {:d} '.format(T,T,tdump) 

    thermo_style = 'thermo_style  custom step temp epair etotal press vol \
       cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz'

    if r == 0:
       r_=None
       data = 'data.lammps'
       writeLammpsData(atoms,data='data.lammps',specorder=None, 
                       masses=masses,
                       force_skew=False,
                       velocities=False,units=units,atom_style=atom_style)
    else:
       r_ = 'restart'
       data = None

    writeLammpsIn(log='lmp.log',timestep=timestep,total=step,restart=r_,
              species=species,
              pair_style= pair_style,  # without lg set lgvdw no
              pair_coeff=pair_coeff,
              fix = thermo_fix,
              freeatoms=freeatoms,natoms=len(atoms),
              fix_modify = ' ',
              dump_interval=dump_interval,more_commond = ' ',
              thermo_style =thermo_style,
              units=units,atom_style=atom_style,
              data=data,T=T,
              restartfile='restart')
    print('\n-  running lammps ...')
    if n==1:
       subprocess.call('lammps<in.lammps>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} lammps -i in.lammps>out'.format(n),shell=True)
    atoms = lammpstraj_to_ase('lammps.trj',inp='in.lammps',recover=c,units=units)
    return atoms

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

def npt(atoms,T=350,tdump=100,timestep=0.1,step=100,gen='poscar.gen',i=-1,c=0,
        p=0.0,x=1,y=1,z=1,n=1,lib='ffield',free=' ',dump_interval=10,r=0):
    thermo_fix = 'fix   1 all npt temp {:f} {:f} {:d} iso {:f} {:f} {:d}'.format(T,
                  T,tdump,p,p,tdump)
    
    atoms = nvt(atoms,T=T,tdump=tdump,timestep=timestep,step=step,gen=gen,i=i,c=c,
                free=free,dump_interval=dump_interval,
                 x=x,y=y,z=z,n=n,lib=lib,thermo_fix=thermo_fix,r=r)
    
    if x>1 or y>1 or z>1:
       ncell     = x*y*z
       natoms    = int(len(atoms)/ncell)
       species   = atoms.get_chemical_symbols()
       positions = atoms.get_positions()
       forces    = atoms.get_forces()
       cell      = atoms.get_cell()
       cell      = [cell[0]/x, cell[1]/y,cell[2]/z]
       u         = np.linalg.inv(cell)
       pos_      = np.dot(positions[0:natoms], u)
       posf      = np.mod(pos_, 1.0)          # aplling simple pbc conditions
       pos       = np.dot(posf, cell)
       atoms     = Atoms(species[0:natoms],pos,#forces=forces[0:natoms],
                         cell=cell,pbc=[True,True,True])
   
    atoms.write('POSCAR.lammps')
    return atoms

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
                     feature[4],',',feature[5],',',feature[6],file=fd) 
    with open('../{:s}/feature.csv'.format(data),'a') as fd:
         print(i,',',energy,',',feature[1],',',feature[2],',',
               feature[3],',',feature[4],',',feature[5],',',density,file=fd)
        

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

feature = np.array([e[0],e[5],e[8],e[10],e[11],e[12],density])
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
   # print('\n residual: \n',res[ind])
   im = np.argmin(res[ind])
   # print(ind[0][im])
   # print(res[ind[0][im]])
   density_= data_[ind[0][im],-1]
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
