#!/usr/bin/env python
import subprocess
from os.path import exists #isfile
from os import system, getcwd,listdir
import sys
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,DotProduct, WhiteKernel,
                                              ConstantKernel as C,RationalQuadratic,
                                              Matern,
                                              ExpSineSquared)
from sklearn.ensemble import RandomForestRegressor
from ase.io import read
from ase import Atoms
from ase.data import atomic_numbers, atomic_masses
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase
from irff.md.gulp import write_gulp_in,get_reax_energy,opt
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
                     feature[4],',',feature[5],',',feature[6],',',feature[7],file=fd) 
    with open('../{:s}/feature.csv'.format(data),'a') as fd:
         print(i,',',energy,',',feature[1],',',feature[2],',',
               feature[3],',',feature[4],',',feature[5],',',feature[6],',',density,file=fd)
        

parser = argparse.ArgumentParser(description='./htecsp.py --g=md.traj')
parser.add_argument('--pred',default='gp',type=str, help='calculation method, 0: only filter the broken structure; 1: gp predict; 2: random forest')
parser.add_argument('--b',default=1.5,type=float, help='energy tolerance that structure is broken')
parser.add_argument('--t',default=0.01,type=float, help='tolerance')
parser.add_argument('--u',default=0.05,type=float, help='uncertainty')
parser.add_argument('--d',default=1.80,type=float, help='density tolerance')
parser.add_argument('--dft',default=0,type=float, help='dft calculation is applied')
parser.add_argument('--n',default=1,type=int, help='the number of cpu used in this calculation')
parser.add_argument('--g',default='gulp.cif',type=str, help='geometry file')
parser.add_argument('--x',default=1,type=int, help='X')
parser.add_argument('--y',default=1,type=int, help='Y')
parser.add_argument('--z',default=1,type=int, help='Z')
parser.add_argument('--f',default=1,type=int, help='which feature to be used')
parser.add_argument('--step',default=5000,type=int, help='Time Step')
# parser.add_argument('--i',default=1,type=int, help='Structure ID')
parser.add_argument('--struc',default='mlp',type=str, help='returen structure with dft/mlp optimization')
parser.add_argument('--data',default='data',type=str, help='data file directory')
parser.add_argument('--r',default='result1',type=str, help='gaussian process fit results file directory')
args = parser.parse_args(sys.argv[1:])

tolerance = args.t
uncertainty= args.u
datafile  = args.data
step      = args.step
ncpu      = args.n
if args.pred=='gp':
   gp     = 1 
elif args.pred=='rf':
   gp     = 2
else:
   gp     = 0
gen       = args.g
den       = args.d
struc     = args.struc
dft       = args.dft
# id_       = args.i
resf      = args.r
broken    = args.b
fea       = args.f

write_input(inp='inp-grad',keyword='grad conv qiterative verb')
run_gulp(n=ncpu,inp='inp-grad')
e = get_reax_energy(fo='output')
write_output(e=e[0])

atoms  = read(gen)
# atoms  = opt(atoms=atoms,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')
masses = np.sum(atoms.get_masses())
volume = atoms.get_volume()
density = masses/volume/0.602214129
atoms.calc = SinglePointCalculator(atoms,energy=e[0])

if fea==1:
   feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])
else:
   feature = np.array([e[0],e[5],e[8],e[10],e[11],e[12],density])
# python htecsp.py  --n=20 --step=500 --d=1.72
# python htecsp.py  --b=1  --step=500

data   = np.loadtxt('../{:s}/feature_mlp.csv'.format(datafile),delimiter=',',skiprows=1)  ## get crystal feature data
data_  = np.loadtxt('../{:s}/feature.csv'.format(datafile),delimiter=',',skiprows=1)      ## get crystal feature data
images = Trajectory('../{:s}/structures.traj'.format(datafile))
d      = data[:,1:]    # 去掉索引

# Train a Gaussian Process 
res    = np.sum(np.square(d - feature),axis=1)
ind    = np.where(res<tolerance)
imin   = np.argmin(res)
# print(ind)
### prepare data 
X_raw  = data[:,1:]
y      = data_[:,-1]
y_eng  = data_[:,1]
# x_mean = np.mean(X,axis=0) 
d_scaler= np.mean(y)/np.mean(data[:,-1])
e_mean = np.mean(data[:,1])
e_scaler= e_mean - np.mean(y_eng)
# X      = X - x_mean
scaler = preprocessing.StandardScaler().fit(X_raw)
X      = scaler.transform(X_raw)
# rng              = np.random.RandomState(1)     ### 选取一些点作图
# training_indices = rng.choice(np.arange(y.size), size=20, replace=False)
# rf = RandomForestRegressor(n_estimators=100,max_depth=10,oob_score=True).fit(X,Y)
# score_   =rf.score(X,Y)      # cross_val_score(rfr,x,y,cv=10).mean()
if not exists('gpr_density.pkl'):
   # kernel =  C(1.0)*RBF(length_scale=[1.0, 1.0, 1.0,1.0,1.0,1.0,1.0], length_scale_bounds=(1e-5, 1e4))
   # kernel = (C(1.0) * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))**2 + 
   #           C(1.0) * RBF(length_scale=[1.0, 1.0, 1.0,1.0,1.0,1.0,1.0]) +
   #           WhiteKernel(noise_level=0.01) )
   # 如果你的7维数据来自于工程实验或物理模拟，数据往往不如 RBF 预期的那样“无限平滑”。
   # 此时，使用 Matern 5/2 核通常能获得比 RBF 更稳健的泛化效果。
   # 适用场景：当数据存在局部快速波动或不那么平滑时（例如传感器数据、流体压力等）
   # kernel = C(1.0) * Matern(length_scale=[1.0]*7, nu=2.5) + WhiteKernel(noise_level=0.1) + DotProduct()**2
    
   # 应对高维耦合：加法核 (Additive Kernel)
   # 在7维空间中，如果某些特征之间是独立的，或者只存在低阶交互，组合核会更有效。
   # DotProduct（点积核）是非平稳核的一种
   # 在高维空间，纯 RBF/Matern 核有时很难拟合简单的线性斜率。DotProduct 负责处理“大趋势”，
   # 让 Matern 核可以专注于拟合更精细的维度耦合（残差）。
   # 如果 DotProduct 的 sigma_0 很大且权重高，说明你的7维问题本质上更接近线性。
   # 如果 Matern 的某些 length_scale 极小，说明这些维度存在强烈的非线性相互作用。
   # 在代码中我使用了 DotProduct(...)**2。在 scikit-learn 中，这允许模型捕捉特征的二次交互，这对于 7 维空间的完全耦合非常有效。
   if fea==1:
      length_scale = [0.0525, 0.0525,0.0493, 0.01, 0.0439, 0.163, 1.0, 1.0]
   else:
      length_scale = [0.0525, 0.0493, 0.01, 0.0439, 0.163, 1.0, 1.0]
       
   kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50)) +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
              0.35**2 * Matern(length_scale=length_scale, nu=2.5) +         # 局部耦合
              WhiteKernel(noise_level=0.031,noise_level_bounds=(1e-8, 1e-1))    )                                   # 噪声补偿
   gpr_density = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=1e-10,normalize_y=True)
   gpr_density.fit(X,y)
    
   kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50)) +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
              0.35**2 * Matern(length_scale=length_scale, nu=2.5) +         # 局部耦合
              WhiteKernel(noise_level=0.031,noise_level_bounds=(1e-8, 1e-1))    )     # 噪声补偿
   gpr_energy = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=1e-10,normalize_y=True)
   gpr_energy.fit(X,y_eng)
   # score  =  gaussian_process.score(X, y)
   with open('gpr_density.pkl', 'wb') as f:
        pickle.dump(gpr_density, f)
   with open('gpr_energy.pkl', 'wb') as f:
        pickle.dump(gpr_energy, f)
   with open('../{:s}/gpcsp.log'.format(resf),'w') as fl:
        print(gpr_density.kernel_,file=fl)
        print(gpr_density.log_marginal_likelihood(),file=fl)
        print(gpr_energy.kernel_,file=fl)
        print(gpr_energy.log_marginal_likelihood(),file=fl)
        # for hyperparameter in kernel.hyperparameters:
            # print(kernel.kernel_,file=fl)
            # print(hyperparameter,file=fl)
else:
   with open('gpr_density.pkl', 'rb') as f:
        gpr_density = pickle.load(f)
   with open('gpr_energy.pkl', 'rb') as f:
        gpr_energy = pickle.load(f)
       
if not exists('rfr_density.pkl'):
   rfr_density = RandomForestRegressor(random_state=37, n_estimators=300,
                                       min_weight_fraction_leaf=0.0,
                                       oob_score=True)
   rfr_density.fit(X, y)  # train
   feature_importances = rfr_density.feature_importances_
   with open('rfr_density.pkl', 'wb') as f:
        pickle.dump(rfr_density, f)
else:
   with open('rfr_density.pkl', 'rb') as f:
        rfr_density = pickle.load(f)

if not exists('../{:s}/gpcsp.csv'.format(resf)):
   with open('../{:s}/gpcsp.csv'.format(resf),'w') as fd:
        print(',   index,          residual,        density_min,         density_rf,   density_gp,'
              '          uncertainty,           energy_min,       eng_pred,        uncertainty_eng',file=fd)

# X_ = np.concatenate((X,np.expand_dims(feature,axis=0)))  #X_train.extend(feature)
X_ = scaler.transform(np.expand_dims(feature,axis=0))
mean_prediction, std_prediction = gpr_density.predict(X_, return_std=True)
mean_eng_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)
density_rf = rfr_density.predict(X_)
# print('95% confidence interval: \n', 1.96 * std_prediction)
         
with open('../{:s}/gpcsp.csv'.format(resf),'a') as fd:
     # id_ = fd.tell()
     print(0,',',imin,',',res[imin],',',data_[imin][-1],',',
           density_rf[0],',',mean_prediction[0],',',
           1.96*std_prediction[0],',',data_[imin][1],',',mean_eng_pred[0],',',1.96*std_eng_pred[0],
           file=fd)
    
if not gp:
    with open('../{:s}/gpcsp.log'.format(resf),'a') as fd:
       # id_ = fd.tell()
       print(0,imin,res[imin],data_[imin][-1],data_[imin][1],
             mean_eng_pred[0],1.96*std_eng_pred[0],
             e_mean,e[0],
             file=fd)  # for debug
    if e_mean-e[0]>broken:
       energy = 100000
    else:
       energy = e[0]
    write_output(e=energy)
    write_geometry(atoms=atoms)
else:
    if gp==1:
       density_= mean_prediction[0] # data_[ind[0][im],-1]
       if density_>np.max(y)*1.1 and (density_/density>1.2 or  density/density_>1.2):
          if density_rf[0]/density>1.2 or  density/densityrf[0]>1.2:
             density_ = density*d_scaler
          else:
             density_ = density_rf[0]
    else:
       density_= density_rf[0] # data_[ind[0][im],-1]
    energy  = -density_ # mean_eng_pred[0]
    write_output(e=energy)
    write_geometry(atoms=atoms)

 