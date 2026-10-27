"""Utility classes and functions for USPEX data processing."""
from os.path import exists
import re
import subprocess
# import torch
# import gpytorch
import numpy as np
from ase.io.trajectory import TrajectoryWriter
from irff.md.gulp import write_gulp_in
from irff.md.lammps import writeLammpsData,writeLammpsIn,lammpstraj_to_ase


class Stack:
    def __init__(self, entry=None):
        self.entry = entry if entry is not None else []

    def push(self, x):
        self.entry.append(x)

    def pop(self):
        return self.entry.pop()

    def close(self):
        self.entry = None


def read_individuals(individuals='Individuals',g=None):
    """Parse USPEX Individuals file, return list of (index, enthalpy, density, fitness)."""
    gene = {}
    with open(individuals) as f:
        for line in f.readlines():
            st = Stack([])
            for x in line:
                if x != "]":
                    st.push(x)
                else:
                    x_ = " "
                    while x_ != "[":
                        x_ = st.pop()
            line = "".join(st.entry)
            l = line.split()

            if len(l) >= 10 and l[0] != "Gen":
                g_ = int(l[0])
                i = int(l[1])
                e = float(l[3])
                d = float(l[5])
                f_ = 99999 if l[6] == "N/A" else float(l[6])
                if g_ in gene:
                    gene[g_].append((i, e, d, f_))
                else:
                    gene[g_] = [(i, e, d, f_)]
        st.close()
    if gene:
       k_ = max(gene.keys()) if g is None else g
       generation = gene[k_]
    else:
       generation = None
    return generation 


def search_structure(feature, D, tolerance=0.01):
    """Search for matching structure in feature database."""
    if D.ndim == 2:
        res = np.sum(np.square(D - feature), axis=1)
        ind = np.where(res < tolerance)
        imin = np.argmin(res)
        res_ = res[imin]
    else:
        res = np.sum(np.square(D - feature))
        ind = [[0]] if res < tolerance else [[]]
        imin = 0
        res_ = res
    return ind, imin, res_


def generate_hbond_lib(elements, src='reaxff_nn.lib', dst=None,
                       hbond_energy=-10.0):
    '''从 src 复制生成 dst，只保留 elements 指定的氢键项，能量替换为 hbond_energy。
    
    elements: 如 'H core C core O core'，自动推导 dst 文件名后缀。
    若 elements 在原始氢键段中不存在，则以 'H core C core O core' 为模板替换元素名。
    '''
    if dst is None:
        elems = [p.lower() for p in elements.split()
                 if len(p) == 1 and p.isalpha()]
        suffix = 'ch' + elems[2]  # C-H-X 命名约定
        dst = f'reaxff_nn_{suffix}.lib'
    if exists(dst):
        return dst

    template = 'H core C core O core'

    with open(src, 'r') as f:
        lines = f.readlines()

    # 检查 elements 是否在氢键段中存在
    in_hbond = False
    found = False
    for line in lines:
        if line.startswith('reaxff3_hbond'):
            in_hbond = True
            continue
        if in_hbond and (line.startswith('#') or line.startswith('reaxff4')):
            break
        if in_hbond and line.startswith(elements):
            found = True
            break

    in_hbond = False
    hbond_done = False
    with open(dst, 'w') as f:
        for line in lines:
            if line.startswith('reaxff3_hbond'):
                in_hbond = True
                f.write(line)
                continue
            if in_hbond and not hbond_done:
                if found:
                    if line.startswith(elements):
                        # 替换该行第二个数值（能量项），保留原始空格
                        pat = r'^(' + re.escape(elements) + r'\s+[\d.-]+\s+)[\d.-]+'
                        line = re.sub(pat, r'\g<1>' + str(hbond_energy), line)
                        f.write(line)
                        f.write('\n')
                        hbond_done = True
                else:
                    if line.startswith(template):
                        # 用模板行替换元素名 + 能量
                        pat = r'^(' + re.escape(template) + r'\s+[\d.-]+\s+)[\d.-]+'
                        line = re.sub(pat, r'\g<1>' + str(hbond_energy), line)
                        line = line.replace(template, elements, 1)
                        f.write(line)
                        f.write('\n')
                        hbond_done = True
                continue
            if in_hbond and hbond_done:
                if line.startswith('#') or line.startswith('reaxff4'):
                    in_hbond = False
                    f.write(line)
                continue
            if not in_hbond:
                f.write(line)
    return dst

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
    
def lammps_opt_mtp(atoms,n=4,step=5000,lib='pot.almtp',T=5.0,P=0.0,tdump=100):
    pair_style = 'mlip load_from={:s}'.format(lib)
    pair_coeff = '* * # C O N H'
    units      = "metal"
    atom_style = 'atomic'
    specorder  = ['C','O','N','H']
    #############  run npt ##################
    writeLammpsData(atoms,data='data.lammps',specorder=specorder,
                    masses={'Al':26.9820,'C':12.0000,'H':1.0080,'O':15.9990,
                             'N':14.0000,'F':18.9980},
                    force_skew=False,
                    velocities=False,units=units,atom_style=atom_style)
    writeLammpsIn(log='lmp.log',timestep=1.0,total=10000,restart=None,
              dump_interval=10,
              species=specorder,
              pair_style = pair_style,  # without lg set lgvdw no
              pair_coeff = pair_coeff,
              fix = 'fix   1 all npt temp {:f} {:f} {:d} iso {:f} {:f} {:d}'.format(T,T,tdump,P,P,tdump),
              fix_modify = ' ',
              # minimize   = '1e-5 1e-5 2000 2000',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              data='data.lammps',units=units,atom_style=atom_style,
              restartfile='restart')
    print('\n-  running lammps minimize ...')
    if n==1:
       subprocess.call('lammps<in.lammps>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} lammps -i in.lammps>out'.format(n),shell=True)
    atoms = lammpstraj_to_ase('lammps.trj',inp='in.lammps',units=units)
   #  line = subprocess.check_output('grep \"Total Energy:\" lmp.log',shell=True)
   
   #############  run minimize ##################
    writeLammpsData(atoms,data='data.lammps',specorder=specorder,
                    masses={'Al':26.9820,'C':12.0000,'H':1.0080,'O':15.9990,
                             'N':14.0000,'F':18.9980},
                    force_skew=False,
                    velocities=False,units=units,atom_style=atom_style)
    writeLammpsIn(log='lmp.log',timestep=0.1,total=3000,restart=None,
              dump_interval=10,
              species=specorder,
              pair_style = pair_style,  # without lg set lgvdw no
              pair_coeff = pair_coeff,
              fix = ' ',
              fix_modify = ' ',
              minimize   = '1e-5 1e-5 3000 3000',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              data='data.lammps',units=units,atom_style=atom_style,
              restartfile='restart')
    print('\n-  running lammps minimize ...')
    if n==1:
       subprocess.call('lammps<in.lammps>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} lammps -i in.lammps>out'.format(n),shell=True)
    atoms = lammpstraj_to_ase('lammps.trj',inp='in.lammps',units=units)
    return atoms
    
# def add_structure(i,atomes_dft,atoms_mlp,feature=None,data=None):
#     with TrajectoryWriter('../{:s}/structures_mlp.traj'.format(data),mode='a') as traj:
#          traj.write(atoms=atoms_mlp)
#     with TrajectoryWriter('../{:s}/structures.traj'.format(data),mode='a') as traj_:
#          traj_.write(atoms=atoms_dft)

#     masses  = np.sum(atoms_dft.get_masses())
#     volume  = atoms_dft.get_volume()
#     density = masses/volume/0.602214129
#     energy  = atoms_dft.get_potential_energy()

#     with open('../{:s}/feature_mlp.csv'.format(data),'a') as fd:
#          print(i,',',feature[0],',',feature[1],',',feature[2],',',feature[3],',',
#                      feature[4],',',feature[5],',',feature[6],',',feature[7],file=fd) 
#     with open('../{:s}/feature.csv'.format(data),'a') as fd:
#          print(i,',',energy,',',feature[1],',',feature[2],',',
#                feature[3],',',feature[4],',',feature[5],',',feature[6],',',density,file=fd)

# class GP(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(GP, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = (
#             gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel()) +
#             gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=train_x.shape[1]))
#         )

#     def forward(self, x):
#         mean = self.mean_module(x)
#         covar = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean, covar)
