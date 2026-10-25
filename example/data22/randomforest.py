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
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,DotProduct, WhiteKernel,
                                              ConstantKernel as C,RationalQuadratic,
                                              Matern,
                                              ExpSineSquared)
from ase.io import read
from ase import Atoms
from ase.data import atomic_numbers, atomic_masses
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase
from irff.md.gulp import write_gulp_in,get_reax_energy,opt
from irff.dft.dftb import dftb_opt
from irff.dft.siesta import siesta_opt #

''' compares the Gaussian Process and Random Forest machine learning model
'''

parser = argparse.ArgumentParser(description='./htecsp.py --g=md.traj')
parser.add_argument('--s',default=0,type=int, help='start index of trajectory file')
parser.add_argument('--e',default=-1,type=int, help='end index of trajectory file')
parser.add_argument('--n',default=1,type=int, help='number of CPU')
parser.add_argument('--t',default='Individuals.traj',type=str, help='trajectory file')
args   = parser.parse_args(sys.argv[1:])
traj   = args.t
s      = args.s
e      = args.e
n      = args.n

tolerance = 0.001
images = Trajectory(traj)
data   = np.loadtxt('feature_mlp.csv',delimiter=',',skiprows=1)  ## get crystal feature data
data_  = np.loadtxt('feature.csv',delimiter=',',skiprows=1)      ## get crystal feature data
d      = data[:,1:]    # 去掉索引

# Train a Gaussian Process 

### prepare data 
X_raw  = data[:,1:]
y      = data_[:,8]
y_eng  = data_[:,1]

scaler = preprocessing.StandardScaler().fit(X_raw)
X      = scaler.transform(X_raw)

rf     = RandomForestRegressor(n_estimators=100,max_depth=10,oob_score=True).fit(X,y)
score_ = rf.oob_score_ # rf.score(X,y)  # cross_val_score(rfr,x,y,cv=10).mean()

if not exists('gpr_density.pkl'):
   kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))**2 +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
              0.35**2 * Matern(length_scale=[0.0526, 0.0525, 0.0493, 1.0, 0.0439, 0.163, 1.0, 1.0], nu=1.5) +         # 局部耦合
              WhiteKernel(noise_level=0.031)    )                                   # 噪声补偿
   gpr_density = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,normalize_y=True)
   gpr_density.fit(X,y)
   with open('gpr_density.pkl', 'wb') as f:
        pickle.dump(gpr_density, f)
   with open('gpr_energy.pkl', 'wb') as f:
        pickle.dump(gpr_energy, f)
   with open('gpcsp.log','w') as fl:
        print(gpr_density.kernel_,file=fl)
        print(gpr_density.log_marginal_likelihood(),file=fl)
        print(gpr_energy.kernel_,file=fl)
        print(gpr_energy.log_marginal_likelihood(),file=fl)
else:
   with open('gpr_density.pkl', 'rb') as f:
        gpr_density = pickle.load(f)
   with open('gpr_energy.pkl', 'rb') as f:
        gpr_energy = pickle.load(f)
# if not exists('gpcsp.csv'):
#    with open('gpcsp.csv','w') as fd:
#         print(', index, residual, density_min, predicted_density, uncertainty,energy_min,eng_pred,uncertainty_eng',file=fd)
if e == -1:
   e = len(images)
for i in range(s,e):
    atoms   = images[i]# read(gen)
    masses  = np.sum(atoms.get_masses())
    volume  = atoms.get_volume()
    density = masses/volume/0.602214129
    
    # atoms = opt(atoms=atoms,step=100,l=1,t=0.000001,n=n, lib='reaxff_nn')
    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',lib='reaxff_nn')
    if n==1:
       subprocess.call('gulp<inp-gulp>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} gulp<inp-gulp>out'.format(n),shell=True)
    e = get_reax_energy(fo='out')
    
    feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])
    res     = np.sum(np.square(d - feature),axis=1)
    ind     = np.where(res<tolerance)
    imin    = np.argmin(res)
    
    # X_ = np.concatenate((X,np.expand_dims(feature,axis=0)))  #X_train.extend(feature)
    X_ = scaler.transform(np.expand_dims(feature,axis=0))
    mean_prediction, std_prediction = gpr_density.predict(X_, return_std=True)
    # mean_eng_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)
    rf_prediction = rf.predict(X_)
    print(i+1,mean_prediction[0],std_prediction[0], rf_prediction[0])
             
    # with open('gpcsp.csv','a') as fd:
    #      print(0,',',imin,',',res[imin],',',data_[imin][-1],',',mean_prediction[0],',',
    #            1.96*std_prediction[0],',',data_[imin][1],',',mean_eng_pred[0],',',1.96*std_eng_pred[0],
    #            file=fd)




 