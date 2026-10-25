

#!/usr/bin/env python
from os.path import exists #isfile
import subprocess
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
from ase.io import read
from ase.data import atomic_numbers, atomic_masses
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.md.gulp import opt,get_reax_energy,write_gulp_in
from irff.dft.siesta import siesta_opt #


def search_structure(atomes_dft,inde=None,step=1000,tolerance=0.01,ncpu=1):
   #  subprocess.call('cp structures.traj structures.backup.traj',shell=True)
    masses  = np.sum(atoms_sea.get_masses())
    volume  = atoms_sea.get_volume()
    density = masses/volume/0.602214129
    # energy= atoms_sea.get_potential_energy()

    atoms = opt(atoms=atoms_sea,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')              ## compute feature
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
    D    = data[:,1:]         # 去掉索引
    D_   = data_[:,1:]   

    images     = Trajectory('structures.traj')
    # images_  = Trajectory('structures_mlp.traj')

    feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density_])
    
    X_raw  = data[:,1:]
    y      = data_[:,8]
    y_eng  = data_[:,1]
    scaler = preprocessing.StandardScaler().fit(X_raw)
    X      = scaler.transform(X_raw)
    
    if not exists('gpr_density.pkl'):
       kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))**2 +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
                  0.35**2 * Matern(length_scale=[0.0526, 0.0525, 0.0493, 0.01, 0.0439, 0.163, 0.1, 0.1], nu=1.5) +       # 局部耦合
                  WhiteKernel(noise_level=0.1)    )                                      # 噪声补偿
       # kernel = ( # 1. 线性趋势项：捕捉全局线性关系
       #           C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50)) + 
       #           # 2. 非线性多尺度项：RQ 核开启 ARD (length_scale 设为 7 维列表)
       #           # alpha 保持宽边界，适应不同的衰减速度
       #           RationalQuadratic(length_scale=[1.0]*7, 
       #                             length_scale_bounds=(1e-2, 1e3), 
       #                              alpha=0.1, 
       #                             alpha_bounds=(1e-5, 1e15) ) + 
       #           WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.5))  )
        
       gpr_density = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,normalize_y=True)
       gpr_density.fit(X,y)
       with open('gpr_density.pkl', 'wb') as f:
            pickle.dump(gpr_density, f)
    else:
       with open('gpr_density.pkl', 'rb') as f:
            gpr_density = pickle.load(f)
       # score  =  gaussian_process.score(X, y)
    
    if not exists('gpr_energy.pkl'):
       kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))**2 +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
                  0.35**2 * Matern(length_scale=[0.0526, 0.0525, 0.0493, 0.01, 0.0439, 0.163, 0.1, 0.1], nu=1.5) +       # 局部耦合
                  WhiteKernel(noise_level=0.1)    )                                      # 噪声补偿
       gpr_energy = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,normalize_y=True)
       gpr_energy.fit(X,y_eng)
       with open('gpr_energy.pkl', 'wb') as f:
            pickle.dump(gpr_energy, f)
    else:
       with open('gpr_energy.pkl', 'rb') as f:
            gpr_energy = pickle.load(f)
           
    with open('gpcsp.log','w') as fl:
        print(gpr_density.kernel_,file=fl)
        print(gpr_density.log_marginal_likelihood(),file=fl)
        print(gpr_energy.kernel_,file=fl)
        print(gpr_energy.log_marginal_likelihood(),file=fl)
       
    res  = np.sum(np.square(D - feature),axis=1)
    ind  = np.where(res<tolerance)
    imin = np.argmin(res)
    d_   = D_[imin]
    if inde is None:
       ind_ = ind[0]
    else:
       ind_ = [inde]

    X_ = scaler.transform(np.expand_dims(feature,axis=0))
    mean_prediction, std_prediction = gpr_density.predict(X_, return_std=True)
    mean_eng_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)

    e_mean = np.mean(D[:,0])
    # print('{:.4f} {:.4f}'.format(e_mean,e[0]),end=' ')
    sigma  = abs(e_mean - e[0])
    
    if len(ind_)>0:
       traj  = TrajectoryWriter('matched_structures.traj',mode='w')
       # for i in ind_:
       #     d_ = D_[i]
       #     print('{:5d} {:11.6f} {:11.6f} {:11.6f} {:11.6f} {:11.6f} {:11.6f} {:11.6f} {:11.6f} {:11.6f}'.format(i,
       #           d_[0],d_[1],d_[2],d_[3],d_[4],d_[5],d_[6],d_[7],res[i]))  
       #     traj.write(atoms=images[i])
       print('{:5d} mt {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:7.4f} {:7.4f} {:7.4f}'.format(imin,
             d_[0],d_[1],d_[2],d_[3],d_[4],d_[5],d_[6],d_[7],res[imin],sigma))  
       traj.write(atoms=images[imin])
       traj.close()
    else:
       # print(f'Specified structure not found in database!') 
       # print(f'energy: {energy}')
       print('{:5d} gp {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:7.4f} {:7.4f} {:7.4f}'.format(imin,
                    mean_eng_pred[0],d_[1],d_[2],d_[3],d_[4],d_[5],d_[6],mean_prediction[0],1.96*std_prediction[0],sigma))  
    return mean_eng_pred[0],1.96*std_eng_pred[0],mean_prediction[0],1.96*std_prediction[0],d_[0],d_[7],res[imin],sigma
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='./search_structure.py --t=Individuals.traj --')
    parser.add_argument('--g',default='structure.gen',type=str, help='data file directory')
    parser.add_argument('--t',default=None,type=str, help='trajectory file')
    parser.add_argument('--n',default=8,type=int, help='number of CPU to be used')
    parser.add_argument('--s',default=500,type=int, help='total MD step')
    parser.add_argument('--tolerance',default=0.01,type=float, help='tolerance')
    parser.add_argument('--i',default=-1,type=int, help='the ID of structure in trajectory to be searched')
    parser.add_argument('--e',default=-1,type=int, help='the end ID of structure in trajectory to be searched')
    parser.add_argument('--d',default=1.85,type=float, help='the minimum density to be computed')
    args = parser.parse_args(sys.argv[1:])
    
    if args.t is None:
       atoms_sea = read(args.g,args.i)
       search_structure(atoms_sea,step=args.s,tolerance=args.tolerance,ncpu=args.n)
    else:
       images = Trajectory(args.t)
       e = len(images) if args.e<0 else args.e

       E_pred,U_e,D_pred,U_d,E_mat,D_mat,R = [],[],[],[],[],[],[]
       for i in range(args.i,e):
           atoms_sea = images[i]
           masses  = np.sum(atoms_sea.get_masses())
           volume  = atoms_sea.get_volume()
           density = masses/volume/0.602214129
           if density > args.d:
              print('CryID: {:4d} '.format(i+1),end='')
              e_pred,uncert_e,d_pred,uncert_d,e_mat,d_mat,res,sigma = search_structure(atoms_sea,
                                                    step=args.s,
                                                    tolerance=args.tolerance,
                                                    ncpu=args.n)
              if sigma<1.0:
                 E_pred.append(e_pred)
                 U_e.append(uncert_e)
                 D_pred.append(d_pred)
                 U_d.append(uncert_d)
                 E_mat.append(e_mat)
                 D_mat.append(d_mat)
                 R.append(res)
       # plot results
       # mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
       ids   = [i for i in range(len(E_pred))]
       R     = np.array(R)
       inds  = np.where(R<0.01)
       ids   = np.array(ids)
       D_mat = np.array(D_mat)
       E_mat = np.array(E_mat)
       D_pred= np.array(D_pred)
       E_pred= np.array(E_pred)
       U_d   = np.array(U_d)
       U_e   = np.array(U_e)
        
       plt.figure()
       plt.plot(ids, D_pred, label="Mean prediction")
       plt.scatter(ids[inds], D_mat[inds], label="Observations")
       plt.fill_between(ids,
            D_pred - U_d,
            D_pred + U_d,
            alpha=0.5,
            label=r"95% confidence interval",
        )
       plt.legend()
       plt.xlabel(r"$Crystal$ $ID$")
       plt.ylabel(r"$Density$ ($g/mol$)")
       plt.title("Gaussian Process Regression on Crystal density Predict")
       plt.savefig('gaussianprocess.svg')


