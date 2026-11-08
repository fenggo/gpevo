"""Core commands: pred, calc, traj, zmat, fdf, sample, add, addall, gp."""
import os
import subprocess
import pickle
import numpy as np
from os import getcwd, chdir, mkdir, listdir
from os.path import exists

from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,DotProduct, WhiteKernel,
                                              ConstantKernel as C,RationalQuadratic,
                                              Matern,
                                              ExpSineSquared)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
# from ase import build
from ase.io import read,write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from irff.md.gulp import opt,get_reax_energy,write_gulp_in
from .utils import (read_individuals, search_structure,generate_hbond_lib,
                            write_input,run_gulp, # add_structure,
                            lammps_opt_mtp,
                            write_output,write_geometry)
from pymatgen.io.ase import AseAtomsAdaptor
from gpevo.runtime import RESIDUAL_CUT
# from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase
from irff.md.gulp import write_gulp_in,get_reax_energy ,opt
# from irff.dft.dftb import dftb_opt
from irff.dft.siesta import siesta_opt
from irff.molecule import Molecules,enlarge, SuperCell,repulse,reassemble # moltoatoms


''' A work flow in combination with USPEX 
    High-Throughput Evolutionary Crystal Structure Prediction Method
'''

def raw_residual(feature, d, scaler):
    """Squared distance to every labelled row, in the ORIGINAL feature space.

    ``d`` is the raw training matrix (``data[:, 1:]``); the residual is
    ``sum((d - feature)**2)`` with no scaling -- the §2.3 standardisation
    experiment was reverted, so ``RESIDUAL_CUT`` (=10.0) again refers to this
    raw distance.  Returns ``(res, imin, X_)`` where ``X_`` is the
    StandardScaler-transformed query row kept only because the GP models are
    fitted on standardised X; ``predict(X_)`` needs the same transform.
    """
    f = np.asarray(feature, dtype=float).reshape(1, -1)
    X_ = scaler.transform(f)
    res = np.sum(np.square(np.asarray(d, dtype=float) - f), axis=1)
    return res, int(np.argmin(res)), X_


def supercell(gen=None,traj=None,x=1,y=1,z=1,i=None):
    if traj is None:
        A = read(gen)
        # build.make_supercell(A,[2,2,2])
        _,atoms = SuperCell(A,fac=1.0,supercell=[x,y,z])
        write(f'POSCAR.supercell_{x}_{y}_{z}',atoms)
    else:
        images = Trajectory(traj)
        # A = images[-1]
        # build.make_supercell(A,[2,2,2])
        his    = TrajectoryWriter(f'{traj.split(".")[0]}_{x}{y}{z}.traj',mode='w')

        if i is not None:
           A = images[i]
           atoms   = A*(x,y,z)
           atoms.calc = SinglePointCalculator(atoms,energy=A.get_potential_energy()*x*y*z)
           his.write(atoms=atoms)
        else:
           for i,A in enumerate(images):
               # _,atoms = SuperCell(A,fac=1.0,supercell=[x,y,z])
               atoms   = A*(x,y,z)
               atoms.calc = SinglePointCalculator(atoms,energy=A.get_potential_energy()*x*y*z)
               his.write(atoms=atoms)
        his.close()


def addall(traj='structures.traj',step=1000,tolerance=0.005,ncpu=1,f=1):
    images = Trajectory(traj)
    for atoms_dft in images:
        add(atoms_dft,step=step,tolerance=tolerance,ncpu=ncpu,f=f)


def add(atoms_dft=None,traj='structures.traj',step=1000,tolerance=0.005,i=-1,ncpu=1,f=1):
    if atoms_dft is None:
       atoms_dft = read(traj,i)
    masses  = np.sum(atoms_dft.get_masses())
    volume  = atoms_dft.get_volume()
    density = masses/volume/0.602214129
    energy  = atoms_dft.get_potential_energy()

    atoms = opt(atoms=atoms_dft,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')              ## compute feature
    e     = get_feature(atoms,n=ncpu,lib='reaxff_nn')
    e_cho = get_hbond_feature(atoms,n=ncpu,elements='H core C core O core')
    e_chn = get_hbond_feature(atoms,n=ncpu,elements='H core C core N core')
    e_chc = get_hbond_feature(atoms,n=ncpu,elements='H core C core C core')

    volume   = atoms.get_volume()
    density_ = masses/volume/0.602214129

    # Initialize database files if they don't exist yet (fresh data directory)
    if not exists('feature_mlp.csv'):
        with open('feature_mlp.csv','w') as fd:
            print(', etot, ebond, eang, etor, evdw, ehb_cho,ehb_chn,ehb_chc, ecoul, density',file=fd)
    if not exists('feature.csv'):
        with open('feature.csv','w') as fd:
            print(', etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn,ehb_chc, ecoul, density',file=fd)
    if not exists('structures_mlp.traj'):
        TrajectoryWriter('structures_mlp.traj',mode='w').close()
    if not exists('structures.traj'):
        TrajectoryWriter('structures.traj',mode='w').close()

    data = np.loadtxt('feature_mlp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
    data_= np.loadtxt('feature.csv',delimiter=',',skiprows=1)          ## get crystal feature data
    # print(cry)
    sc    = get_nmol(atoms) if f==2 else 1.0
    feature = np.array([e[0]/sc,e[1]/sc,e[5]/sc,e[8]/sc,e[10]/sc,
                        e_cho[11]/sc,e_chn[11]/sc,e_chc[11]/sc,e[12]/sc,density_])
    if data.ndim == 1:           # 表头后 0 或 1 行数据时 loadtxt 返回一维数组
        if data.size == 0:
            d = data.reshape(0, len(feature))   # 空数据库
            i = 0
        else:
            d = data[1:].reshape(1, -1)         # 仅一行数据
            i = int(data[0]) + 1
    else:
        d = data[:,1:]           # 去掉索引
        i = int(data[-1][0])+1   # 获取索引
    res  = np.sum(np.square(d - feature),axis=1)
    ind  = np.where(res<tolerance)
    
    if len(ind[0])>0:
       print(f'Structure already in database with index {ind[0]}!') 
       print(f'energy: {d[ind[0],0]}')
    else:
       with open('feature_mlp.csv','a') as fd:
            print(i,',',feature[0],',',feature[1],',',feature[2],',',feature[3],',',feature[4],',',
                     feature[5],',',feature[6],',',feature[7],',',
                     feature[8],',',feature[9],
                     file=fd) 
       with open('feature.csv','a') as fd:
            print(i,',',energy/sc,',',feature[1],',',feature[2],',',feature[3],',',feature[4],',',
                  feature[5],',',feature[6],',',feature[7],',',
                  feature[8],',',density,file=fd)  
    
       atoms.calc = SinglePointCalculator(atoms,energy=e[0])
       with TrajectoryWriter('structures_mlp.traj',mode='a') as traj:
            traj.write(atoms=atoms)
       with TrajectoryWriter('structures.traj',mode='a') as traj:
            traj.write(atoms=atoms_dft)


def get_feature(atoms,n=1,lib='reaxff_nn'):
    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',lib=lib)
    if n==1:
       subprocess.call('gulp<inp-gulp>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} gulp<inp-gulp>out'.format(n),shell=True)
    e = get_reax_energy(fo='out')
    return e


def get_hbond_feature(atoms,n=1,elements='H core C core O core'):
    lib = generate_hbond_lib(elements)
    e = get_feature(atoms,n=n,lib=lib)
    return e


def get_nmol(atoms):
    '''返回晶胞内分子数，检测方式与 fixbroken 一致 (相同 rcut)。'''
    if not exists("molecule.pkl"):
       m_ = Molecules(atoms, rcut={"H-H": 1.0, "H-O": 1.02, "O-O": 1.4,
                                "H-N": 1.22, "H-C": 1.35, "others": 1.75},
                   check=True)
       with open("molecule.pkl", "wb") as f:
            pickle.dump(m_, f)
    else:
       with open("molecule.pkl", "rb") as f:
            m_ = pickle.load(f)
    return len(m_)


def gp(tolerance=0.005,step=1000,n=1,b=1.5,u=0.04,f=1,dat='data',dft=0,den=1.82,pop=100,ref='results1',id_=None,k=1,mode='ei',optype=11,
       backend="sklearn",training_iter=200,learning_rate=0.1,log_dir=None):
    ''' Gaussian Process '''
    write_input(inp='inp-grad',keyword='grad conv qiterative verb')
    run_gulp(n=n,inp='inp-grad')
    e = get_reax_energy(fo='output')
    write_output(e=e[0])

    atoms  = read('gulp.cif')
    # atoms  = opt(atoms=atoms,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    atoms.calc = SinglePointCalculator(atoms,energy=e[0])

    e_cho = get_hbond_feature(atoms,n=n,elements='H core C core O core')
    e_chn = get_hbond_feature(atoms,n=n,elements='H core C core N core')
    e_chc = get_hbond_feature(atoms,n=n,elements='H core C core C core')

    if f==1:
       feature = np.array([e[0],e[1],e[5],e[8],e[10],e_cho[11],e_chn[11],e_chc[11],e[12],density])
    elif f==2:
       nmol = get_nmol(atoms)
       feature = np.array([e[0],e[1],e[5],e[8],e[10],e_cho[11],e_chn[11],e_chc[11],e[12],density], dtype=float)
       feature[:-1] /= nmol      # 能量项按分子数平均, 密度(末位)不变
    else:
       feature = np.array([e[0],e[5],e[8],e[10],e[11],e[12],density])
 
    data   = np.loadtxt('../{:s}/feature_mlp.csv'.format(dat),delimiter=',',skiprows=1)  ## get crystal feature data
    data_  = np.loadtxt('../{:s}/feature.csv'.format(dat),delimiter=',',skiprows=1)      ## get crystal feature data
    # The database trajectory used to be read here too; nothing below uses it,
    # so a per-structure call no longer pays for it.
    d      = data[:,1:]    # 去掉索引
    # f==2 只对查询 feature 做 nmol 平均, 训练数据 d 保持原始总能量不动

    ### prepare data 
    X_raw  = d
    y      = data_[:,-1]
    y_eng  = data_[:,1]

    d_scaler= np.mean(y)/np.mean(data[:,-1])
    e_mean = np.mean(data[:,1])
    e_scaler= e_mean - np.mean(y_eng)

    scaler = preprocessing.StandardScaler().fit(X_raw)
    X      = scaler.transform(X_raw)
    # Residual in the ORIGINAL feature space (§2.3 reverted); X_ is the
    # standardised query row the GP predict() consumes.
    res, imin, X_ = raw_residual(feature, d, scaler)
    ind    = np.where(res < tolerance)

    gpr_energy, gpr_density = load_gaussian_process(
        X, y, y_eng, backend=backend, training_iter=training_iter,
        learning_rate=learning_rate, log_dir=log_dir)
    rfr_density = load_rfr(X, y)
    if not exists('gp.csv'):
        with open('gp.csv','w') as fd:
             print(',   index,          residual,        density_min,         density_rf,   density_gp,'
                '          uncertainty,           energy_min,       eng_pred,        uncertainty_eng,'
                '      density_gp_updated,  uncertainty_updated',file=fd)

    mean_prediction, std_prediction = gpr_density.predict(X_, return_std=True)
    mean_eng_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)
    density_rf = rfr_density.predict(X_)
    # print('95% confidence interval: \n', 1.96 * std_prediction)
    
    density_= mean_prediction[0] # data_[ind[0][im],-1]
    # if ((density_>np.max(y)*1.1 and (density_/density>1.5 or  density/density_>1.5)) or 
    #     (density_>np.max(y) and res[imin]>RESIDUAL_CUT) ):
    if res[imin] > RESIDUAL_CUT:
       if density_rf[0]/density>1.5 or  density/density_rf[0]>1.5:
          density_ = density*d_scaler
       else:
          density_ = density_rf[0]

    # 晶体全局 ID：优先使用 USPEX submitJob 下发的 bodyCount+1（--id）；
    # 未提供时回退到 Individuals 最后一行（兼容手动运行 uspexkit gp 的场景）
    if id_ is not None:
        id_ = int(id_)
    else:
        try:
            last_line = subprocess.run(['tail', '-1', f'../{ref}/Individuals'],
                                       capture_output=True, text=True,
                                       check=True).stdout.split()
            id_ = int(last_line[1]) + 1 if len(last_line) >= 2 and last_line[1].isdigit() else 1
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
            id_ = 1
           
    # DFT 分支运行标记：该晶体已 DFT 并加入训练集, gp.csv 只能写一行
    dft_done = False
    # 更新后的后验 (该点作为训练点): (mean_updated, 1.96*std_updated)
    density_updated = float('nan')
    uncert_updated = float('nan')

    if dft:
       data_pred = np.loadtxt('gp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
       if data_pred.size > 0:
          if  data_pred.ndim==2:
              if data_pred.shape[0]>pop:
                 U         = data_pred[:,6]
                 R         = data_pred[:,2]
                 Density   = np.where(np.logical_and(U<0.2,R<RESIDUAL_CUT),data_pred[:,5],data_pred[:,4])
                 imax      = np.argmax(Density)
                 # with open('gp.log','a') as fg:
                 #     print(data_pred.ndim,file=fg)
                 #     print(data_pred.shape[0],file=fg)
                 #     print(density_,0.9*Density[imax],file=fg)
                 if density_ >= 0.98*Density[imax] and density_ >= den:
                    if std_prediction[0]>u and res[imin] < RESIDUAL_CUT:
                       subprocess.call("cp ../Specific/*.psf ./", shell=True)
                       img = siesta_opt(atoms, ncpu=n, us="F", VariableCell="true", tstep=step,
                                             xcf="GGA", xca="PBE", basistype="split")
                       subprocess.call(f"mv siesta.traj id_{id_}.traj", shell=True)
                       subprocess.call("rm siesta.* ", shell=True)
                       subprocess.call("rm *.xml ", shell=True)
                       subprocess.call("rm INPUT_TMP.* ", shell=True)
                       subprocess.call("rm fdf-* ", shell=True)
                       # img[0].write(f"POSCAR.{s}")
                       atoms_opt = img[-1]
                       # atoms_opt.write(f"POSCAR.{s}_opt")
                       masses = np.sum(atoms_opt.get_masses())
                       volume = atoms_opt.get_volume()
                       density_ = masses / volume / 0.602214129
                       energy = atoms_opt.get_potential_energy()
                       # with open('refit','w') as fr:
                       #      print(1,file=fr)
                       subprocess.call("rm gpr_density.pkl gpr_energy.pkl", shell=True) 
                       subprocess.call("rm rfr.pkl", shell=True) 
                              
                       # add structure to database
                       # chdir(data_dir)
                       with open(f"../{dat}/feature_mlp.csv", "a") as fd:
                            print(f"{id_},{feature[0]},{feature[1]},{feature[2]},{feature[3]},"
                                  f"{feature[4]},{feature[5]},{feature[6]},{feature[7]},{feature[8]},{feature[9]}", file=fd)
                       with open(f"../{dat}/feature.csv", "a") as fd:
                            print(f"{id_},{energy},{feature[1]},{feature[2]},"
                                  f"{feature[3]},{feature[4]},{feature[5]},{feature[6]},{feature[7]},{feature[8]},{density_}",
                                  file=fd)

                       # atoms_opt.calc = SinglePointCalculator(atoms_opt, energy=energy)
                       with TrajectoryWriter(f"../{dat}/structures_mlp.traj", mode="a") as traj_w:
                            traj_w.write(atoms=atoms)
                       with TrajectoryWriter(f"../{dat}/structures.traj", mode="a") as traj_w:
                            traj_w.write(atoms=atoms_opt)

                       # ---- 更新后的后验 ----
                       # 新 DFT 点已追加进训练集。用当前 GP 的超参冻结
                       # (optimizer=None, 不重新优化 LML) 拟合一次, 对该点
                       # 自身预测 -> 更新高斯均值 + 不确定度。带 WhiteKernel
                       # 观测噪声时训练点处后验 σ 不为零, 正是三角上误差棒
                       # 的长度。失败则保留 nan (绘图端回退为 σ=0)。
                       # This fit reuses sklearn's fitted kernel_/alpha; a
                       # gpytorch kernel_ is a gpytorch object and cannot be
                       # fed back to GaussianProcessRegressor.  Skip it there:
                       # density_updated / uncert_updated stay nan and the
                       # plots already fall back to sigma=0.
                       if hasattr(gpr_density, "L_"):
                           try:
                               X_upd = np.vstack([X, X_[0]])
                               y_upd = np.concatenate([y, np.array([density_])])
                               gpr_upd = GaussianProcessRegressor(
                                   kernel=gpr_density.kernel_,
                                   alpha=gpr_density.alpha,
                                   optimizer=None,
                                   normalize_y=True)
                               gpr_upd.fit(X_upd, y_upd)
                               mu_upd, sd_upd = gpr_upd.predict(X_, return_std=True)
                               density_updated = float(mu_upd[0])
                               uncert_updated = float(1.96 * sd_upd[0])
                           except Exception as _err:
                               print(f'[gp] updated posterior failed for id {id_}: {_err}')

                       dft_done = True

    if not dft_done:
       with open('gp.csv','a') as fd:
           # id_ = fd.tell()
           print(id_,',',imin,',',res[imin],',',data_[imin][-1],',',
               density_rf[0],',',mean_prediction[0],',',
               1.96*std_prediction[0],',',data_[imin][1],',',mean_eng_pred[0],',',1.96*std_eng_pred[0],
               ',', '',',','',
               file=fd)
    else:
       # DFT 已完成: 只写一行, 末尾两列为更新后后验均值与 ±1.96σ
       with open('gp.csv','a') as fd:
           print(id_,',',imin,',',res[imin],',',data_[imin][-1],',',
               density_rf[0],',',mean_prediction[0],',',
               1.96*std_prediction[0],',',data_[imin][1],',',mean_eng_pred[0],',',1.96*std_eng_pred[0],
               ',', f'{density_updated:.8f}',',',f'{uncert_updated:.8f}',
               file=fd)
    if optype==11:
       energy = -density_ # mean_eng_pred[0]
    elif optype==1:
       energy = mean_eng_pred[0]
    else:
       energy = feature[0]
    write_output(e=energy)
    write_geometry(atoms=atoms)

    # Return exactly what was appended to gp.csv, so in-process callers do not
    # have to re-parse the file (column order identical to the csv row).
    return {
        'id': id_,
        # The 10-D query feature, so a caller that needs it again (pred() after
        # the training set grew) can reuse it instead of re-running ReaxFF.
        'feature': [float(value) for value in feature],
        'nearest': int(imin),
        'residual': float(res[imin]),
        'density_min': float(data_[imin][-1]),
        'density_rf': float(density_rf[0]),
        'density_gp': float(mean_prediction[0]),
        'uncertainty': float(1.96*std_prediction[0]),
        'energy_min': float(data_[imin][1]),
        'energy_pred': float(mean_eng_pred[0]),
        'uncertainty_eng': float(1.96*std_eng_pred[0]),
        'density_updated': density_updated,
        'uncertainty_updated': uncert_updated,
        'dft': bool(dft_done),
        'density': float(density_),     # density actually used for the fitness
        'energy': float(energy),        # value written to `output`
        'optype': optype,
    }
    

def calcdata(traj='structures.traj',n=8,c='nn',step=1000,f=1):
    ''' c: calculator, which mathine learning potential to be used
        f: feature 模式 (1=原始总能量; 2=能量项按分子数平均, 密度除外) '''
    images      = Trajectory(traj)
    traj_       = TrajectoryWriter('structures_mlp.traj',mode='w')

    with open('feature_mlp.csv','w') as fd:
        print(', etot, ebond, eang, etor, evdw, ehb_cho,ehb_chn,ehb_chc, ecoul, density',file=fd)
    with open('feature.csv','w') as fd_:
        print(', etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn,ehb_chc, ecoul, density',file=fd_)

    for i,atoms in enumerate(images):
        masses = np.sum(atoms.get_masses())
        volume = atoms.get_volume()
        density_ = masses/volume/0.602214129
        energy = atoms.get_potential_energy()
        if c=='nn':
           atoms = opt(atoms=atoms,step=step,l=1,t=0.000001,n=n, lib='reaxff_nn')
        elif c=='mtp':
           atoms = lammps_opt_mtp(atoms=atoms,step=step,n=n,lib='pot.almtp')
        else:
           raise RuntimeError("Caluclator not supported!") 
        e     = get_feature(atoms,n=n,lib='reaxff_nn')
        e_cho = get_hbond_feature(atoms,n=n,elements='H core C core O core')
        e_chn = get_hbond_feature(atoms,n=n,elements='H core C core N core')
        e_chc = get_hbond_feature(atoms,n=n,elements='H core C core C core')

        sc = get_nmol(atoms) if f==2 else 1.0

        # atoms = read('gulp.cif')
        atoms.calc = SinglePointCalculator(atoms,energy=e[0])
        traj_.write(atoms=atoms)
        # e,ebond,elp,eover,eunder,eang,epen,tconj,etor,fconj,evdw,ehb,ecl,esl
        
        volume = atoms.get_volume()
        density = masses/volume/0.602214129
        #  print(e)
        print('ID {:4d}: etol {:8.4f} ebond: {:8.4f} eang: {:8.4f} etor: {:8.4f} evdw: {:8.4f} '
            'ehb: {:8.4f}  {:8.4f} {:8.4f} {:8.4f} ' 
            'ecoul: {:8.4f} density: {:9.6}'.format(i,e[0],e[1],e[5],e[8],e[10],
                                        e[11],e_cho[11],e_chn[11],e_chc[11],
                                        e[12],density))
        with open('feature_mlp.csv','a') as fd:
            print(i,',',e[0]/sc,',',e[1]/sc,',',e[5]/sc,',',e[8]/sc,',',e[10]/sc,',',e_cho[11]/sc,',',e_chn[11]/sc,',',e_chc[11]/sc,',',
                e[12]/sc,',',density,file=fd) 
        with open('feature.csv','a') as fd_:
            print(i,',',energy/sc,',',e[1]/sc,',',e[5]/sc,',',e[8]/sc,',',e[10]/sc,',',e_cho[11]/sc,',',e_chn[11]/sc,',',e_chc[11]/sc,',',
                e[12]/sc,',',density_,file=fd_) 
            
    traj_.close()

# ──────────────────────────────────────────────
#  fix broken molecule
# ──────────────────────────────────────────────
def fixbroken(broken=1.5,dat='data',scale=1.2,ncpu=1):
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
    # feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])

    data   = np.loadtxt('../{:s}/feature_mlp.csv'.format(dat),delimiter=',',skiprows=1)  ## get crystal feature data
    data_  = np.loadtxt('../{:s}/feature.csv'.format(dat),delimiter=',',skiprows=1)      ## get crystal feature data
    # The database trajectory used to be read here too; nothing below uses it,
    # so a per-structure call no longer pays for it.
    d      = data[:,1:]    # 去掉索引

    ### prepare data 
    X_raw  = data[:,1:]
    y      = data_[:,-1]
    y_eng  = data_[:,1]
    d_scale= np.mean(y)/np.mean(data[:,-1])
    e_mean = np.mean(data[:,1])
    e_scale= e_mean - np.mean(y_eng)

    if e_mean-e[0]>broken:
       cell  = atoms.get_cell()
       inv_c = np.linalg.inv(cell)
       if exists("molecule.pkl"):
          with open("molecule.pkl", "rb") as f:
               m_ = pickle.load(f)
          for m in m_:
              for i,na in enumerate(m.mol_index):
                  m.mol_x[i] = atoms.positions[na]
              m.center = np.sum(m.mol_x,axis=0)/m.natom

          irun   = 0
          d_safe = 2.0
          while e_mean-e[0]>broken and irun < 4:
                # wrap each molecule as a rigid body back into the cell
                for mol in m_:
                    fc    = mol.center @ inv_c
                    shift = np.floor(fc) @ cell
                    if np.any(shift != 0.0):
                       mol.move(-shift)
                       mol.center -= shift

                m_,d_min,niter = repulse(m_, cell=cell, d_safe=d_safe)
                atoms_new = reassemble(m_, cell=cell, scale=scale)
                atoms_new,e,density = get_gulp_energy(atoms_new, ncpu=ncpu,o=False)
                print('fixbroken iter {:d}: d_min={:.3f} A (d_safe={:.2f}), '
                      'E deviation={:.3f} eV'.format(irun+1,d_min,d_safe,e_mean-e[0]))
                atoms  = atoms_new
                irun  += 1
                d_safe += 0.3
       else:
          # no cached fragments – detect from the current structure and cache
          m_ = Molecules(atoms,rcut={"H-H":1.0,"H-O":1.02,"O-O":1.4,"H-N":1.22,
                                     "H-C":1.35,"others":1.75},check=True)
          with open("molecule.pkl","wb") as f:
               pickle.dump(m_,f)

          irun, d_safe = 0, 2.0
          while e_mean-e[0]>broken and irun < 4:
                for mol in m_:
                    fc    = mol.center @ inv_c
                    shift = np.floor(fc) @ cell
                    if np.any(shift != 0.0):
                       mol.move(-shift)
                       mol.center -= shift

                m_,d_min,niter = repulse(m_, cell=cell, d_safe=d_safe)
                atoms_new = reassemble(m_, cell=cell, scale=scale)
                atoms_new,e,density = get_gulp_energy(atoms_new, ncpu=ncpu,o=False)
                print('fixbroken iter {:d}: d_min={:.3f} A (d_safe={:.2f}), '
                      'E deviation={:.3f} eV'.format(irun+1,d_min,d_safe,e_mean-e[0]))
                atoms  = atoms_new
                irun  += 1
                d_safe += 0.3
    else:
       if not exists("molecule.pkl"):
          m_  = Molecules(atoms,rcut={"H-H":1.0,"H-O":1.02,"O-O":1.4,"H-N":1.22,"H-C":1.35,
                                "others": 1.75},check=True)
          with open("molecule.pkl", "wb") as f:
               pickle.dump(m_, f)
    
    write_output(e=e[0])
    write_geometry(atoms=atoms)

# ──────────────────────────────────────────────
#  GULP energy helper
# ──────────────────────────────────────────────

def get_gulp_energy(atoms, ncpu=8,o=True):
    if o:
       atoms_opt = opt(atoms=atoms, step=1000, l=1, t=0.000001, n=ncpu, lib="reaxff_nn")
    else:
       atoms_opt = atoms
    write_gulp_in(atoms_opt, runword="gradient nosymmetry conv qite verb", lib="reaxff_nn")
    if ncpu == 1:
        subprocess.call("gulp<inp-gulp>out", shell=True)
    else:
        subprocess.call(f"mpirun -n {ncpu:d} gulp<inp-gulp>out", shell=True)
    e = get_reax_energy(fo="out")
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses / volume / 0.602214129
    return atoms_opt, e, density

# ──────────────────────────────────────────────
#  ML model helpers
# ──────────────────────────────────────────────

# Kernel spec revision.  Bump whenever a kernel below changes: the fitted
# models are cached in gpr_*.pkl, and without this stamp a cached model would
# silently keep the old kernel (the old code required deleting the pkls by
# hand -- see the commented-out "refit" logic this replaces).
_KERNEL_VERSION = 2
# Cache tag for the opt-in GPyTorch backend.  Keeping the sklearn tag at 2
# means switching backends always refits, without invalidating existing
# sklearn caches.
_GPYTORCH_VERSION = "gpytorch-1"


def _load_or_fit_gpr(path, X, y, kernel, backend="sklearn", core="matern",
                     training_iter=200, learning_rate=0.1):
    """Fit a GPR with `kernel`, caching it under a backend-specific version.

    A cache entry written by an older revision, by the other backend, or a bare
    model pickle from before versioning is retrained and overwritten.  The
    default (sklearn) path keeps the original construction byte-for-byte.
    """
    version = _KERNEL_VERSION if backend == "sklearn" else _GPYTORCH_VERSION
    if exists(path):
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            if isinstance(blob, dict) and blob.get("version") == version:
                return blob["model"]
        except Exception:
            pass                        # unreadable/old format -> refit
    if backend == "gpytorch":
        from .gp import GPyTorchRegressor   # lazy: sklearn path never imports torch
        gpr = GPyTorchRegressor(kernel=kernel, n_restarts_optimizer=10,
                                normalize_y=True, random_state=0,
                                training_iter=training_iter,
                                learning_rate=learning_rate, core=core)
    else:
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                       normalize_y=True, random_state=0)
    gpr.fit(X, y)
    with open(path, "wb") as f:
        pickle.dump({"version": version, "model": gpr}, f)
    return gpr


def _write_gp_diagnostics(fl, gpr, X, y, label, backend):
    """Append the training report for one surrogate to ``gpcsp.log``.

    Everything is read from state the fit already stored; nothing is refitted
    here, because load_gaussian_process runs once per structure in gp().
    """
    print(f"  [{label}] backend={backend}  n_train={X.shape[0]}  "
          f"n_features={X.shape[1]}", file=fl)
    info = getattr(gpr, '_fit_info', None)
    if not info or 'restart_lmls' not in info:
        return                       # older pickle cache without diagnostics
    lmls = [v for v in info['restart_lmls'] if v is not None]
    spread = max(lmls) - min(lmls) if len(lmls) > 1 else 0.0
    print(f"  restarts={info['n_restarts']}  failed={info['failed_restarts']}  "
          f"max_iter={info['training_iter']}  "
          f"best_lbfgs_evals={info['best_evals']}", file=fl)
    print(f"  restart_LML=[{', '.join(f'{v:.4f}' for v in lmls)}]  "
          f"spread={spread:.4f}", file=fl)
    scale_note = (f"  y_std={info['y_std']:.6g}"
                  if info.get('y_std') else "")
    print(f"  best_LML={info['best_lml']:.6f}/point "
          f"(normalised y){scale_note}", file=fl)
    if 'noise_sigma' not in info:
        print("  (hyperparameter/fit diagnostics unavailable)", file=fl)
        return
    print(f"  noise_sigma={info['noise_sigma']:.6f}  "
          f"mean_constant={info['mean_constant']:+.6f}  "
          f"linear_outputscale={info['linear_outputscale']:.6g}  "
          f"core_outputscale={info['core_outputscale']:.6g}", file=fl)
    ls = ', '.join(f'{v:.3f}' for v in info['lengthscale'])
    print(f"  lengthscale(ARD)=[{ls}]", file=fl)
    print(f"  train_RMSE={info['train_rmse']:.6f}  "
          f"train_R2={info['train_r2']:.6f}  "
          f"max|resid|={info['train_max_resid']:.6f}", file=fl)
    print(f"  train_posterior_sigma_mean="
          f"{info['train_posterior_sigma_mean']:.6f}", file=fl)
    print("  LOO_RMSE=n/a (gpytorch backend: no sklearn Cholesky factor)",
          file=fl)


def load_gaussian_process(X, y, y_eng, backend="sklearn",
                          training_iter=200, learning_rate=0.1,
                          log_dir=None):
    length_scale = [0.1 for i in range(X.shape[1])]
    # density: Matern 5/2 + linear + noise.  LOO on the training set showed
    # this target is insensitive to the core (RBF is within noise), so the
    # original spec is kept.
    kernel_density = (
        0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))
        + 0.35**2 * Matern(length_scale=length_scale, nu=2.5)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e-1))
    )
    # energy: RBF instead of Matern 5/2.  Same-composition polymorphs have a
    # smoother DFT energy surface than nu=2.5 assumes; LOO gave RMSE 0.233 vs
    # 0.254 eV and R^2 0.821 vs 0.788.  The linear term only contributes once
    # the core is smooth (with Matern the optimizer collapsed sigma_0 to ~0).
    # The LML surface is multimodal (the linear term either survives or
    # collapses, ~3 LML apart); random_state=0 pins the restart sequence so
    # fits are reproducible and land on the better optimum for this data.
    kernel_energy = (
        0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))
        + 0.35**2 * RBF(length_scale=length_scale)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e-1))
    )
    gpr_density = _load_or_fit_gpr("gpr_density.pkl", X, y, kernel_density,
                                   backend=backend, core="matern",
                                   training_iter=training_iter,
                                   learning_rate=learning_rate)
    gpr_energy = _load_or_fit_gpr("gpr_energy.pkl", X, y_eng, kernel_energy,
                                  backend=backend, core="rbf",
                                  training_iter=training_iter,
                                  learning_rate=learning_rate)

    # The original uspexkit wrote ../<resFolder>/gpcsp.log; the bare relative
    # path landed in CalcFoldN for gp() and in <data> for pred().  Take the
    # run's results folder explicitly so both entry points agree.
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "gpcsp.log")
    else:
        log_path = "gpcsp.log"
    with open(log_path, "w") as fl:
        print("density:", gpr_density.kernel_, file=fl)
        print("LML:", gpr_density.log_marginal_likelihood(), file=fl)
        _write_gp_diagnostics(fl, gpr_density, X, y, "density", backend)
        print("energy:", gpr_energy.kernel_, file=fl)
        print("LML:", gpr_energy.log_marginal_likelihood(), file=fl)
        _write_gp_diagnostics(fl, gpr_energy, X, y_eng, "energy", backend)

    return gpr_energy, gpr_density

# def load_mlp(X, y):
#     mlp = MLPRegressor((16, 8), max_iter=20000)
#     mlp.fit(X, y)
#     return mlp

def load_rfr(X, y):
    if not exists("rfr.pkl"):
       rfr = RandomForestRegressor(random_state=37, n_estimators=300,
                                min_weight_fraction_leaf=0.0, oob_score=True)
       rfr.fit(X, y)
       with open("rfr.pkl", "wb") as f:
            pickle.dump(rfr, f)
    else:
       with open("rfr.pkl", "rb") as f:
            rfr = pickle.load(f)
    return rfr

# ──────────────────────────────────────────────
#  pred — 高斯过程预测
# ──────────────────────────────────────────────

def pred(t="Individuals.traj", g=None, gen=None, f=1, den=1.88, ids=None,x=-1,
         c='nn',step=300, ncpu=8, dat="data", tolerance=0.001, features=None,
         backend="sklearn",training_iter=200,learning_rate=0.1,log_dir=None):
    """
    Predict density and energy using Gaussian Process + MLP + RandomForest.

    Args:
        t: trajectory file name
        g: geometry structure file name (None = use trajectory)
        gen: generation number (None = latest)
        f: feature flag (1 = 8D feature, else 7D)
        den: unused -- every crystal with a negative fitness is predicted, the
             density threshold no longer drops any of them
        ids: comma/space separated crystal indices
        step: optimization steps
        ncpu: number of CPUs
        dat: data directory name
        tolerance: structure matching tolerance
        features: optional ``{crystal id: 10-D feature}`` already computed for
            these very structures (the GP step of this generation).  A crystal
            found here skips the ReaxFF re-optimisation and its four GULP runs;
            the geometry is the relaxed one the trajectory holds either way.
    """
    if g is not None:
       atoms  = read(g,index=x)
       images = [atoms]
    else:
       images = Trajectory(t)
    
    if g is not None:
       ids_list = [1]
    elif not ids:
        ids_list = []
        res = read_individuals(g=gen)
        for i, _e, _d, _f in res:
            # No density gate: a low-density crystal still has to reach
            # density_predict.log, since the reports read every EI pick from it
            # and a missing row reads as "this crystal never got predicted".
            if _f < 0.0:
                ids_list.append(i)
    else:
        ids_list = [int(i) for i in ids.split()]

    root_dir = getcwd()
    if not exists("density_predict.log"):
        with open("density_predict.log", "w") as fd:
            print("# Crystal_id Residual Density_mlp Density_rf Density_gp std_den Energy std_eng", file=fd)

    masses = np.sum(images[0].get_masses())
    # Collected per crystal and returned, so callers (gpevo) can store the
    # predictions on the structure instead of re-parsing density_predict.log.
    results = {}
    if not ids_list:
        return results

    # The database and the fitted models are the same for every crystal of one
    # call, so they are read once here.  They used to be read inside the loop:
    # both csv files, the whole structures.traj (whose handle was never used)
    # and the model pickles were reloaded once per crystal.
    dir_list = root_dir.split("/")
    rootdir = "/".join(dir_list[:-1])
    if exists(f"{root_dir}/{dat}"):
       data_dir = f"{root_dir}/{dat}"
    else:
       data_dir = f"{rootdir}/{dat}"
    chdir(data_dir)
    assert exists("structures.traj"), "Error, datafile not found in data directory!"
    data  = np.loadtxt("feature_mlp.csv", delimiter=",", skiprows=1)
    data_ = np.loadtxt("feature.csv", delimiter=",", skiprows=1)
    # f==2 只对查询 feature 做 nmol 平均, 训练数据 X_raw 保持原始总能量不动
    X_raw = data[:, 1:]
    D     = X_raw
    y     = data_[:, -1]
    y_eng = data_[:, 1]
    scaler = preprocessing.StandardScaler().fit(X_raw)
    X      = scaler.transform(X_raw)
    gpr_energy, gpr_density = load_gaussian_process(
        X, y, y_eng, backend=backend, training_iter=training_iter,
        learning_rate=learning_rate, log_dir=log_dir or root_dir)
    rfr = load_rfr(X, y)
    chdir(root_dir)

    for s in ids_list:
        # print(images,s)
        atoms = images[s - 1]

        chdir(data_dir)
        cached = features.get(s) if features else None
        if cached is not None:
            # The GP step of this generation already computed this feature from
            # the same relaxed geometry; re-running ReaxFF would only re-walk
            # the minimiser to the same local minimum.
            atoms_mlp = atoms
            volume  = atoms_mlp.get_volume()
            density = masses / volume / 0.602214129
            feature = np.asarray(cached, dtype=float)
        else:
            if c=='nn':
               # atoms_mlp, e, density = get_gulp_energy(atoms, ncpu=ncpu)
               atoms_mlp = opt(atoms=atoms,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')
            elif c=='mtp':
               # atoms_mlp, e, density = get_gulp_energy(atoms, ncpu=ncpu)
               atoms_mlp = lammps_opt_mtp(atoms=atoms,step=step,n=ncpu,lib='pot.almtp')
            else:
               raise RuntimeError("Caluclator not supported!")

            volume  = atoms_mlp.get_volume()
            density = masses / volume / 0.602214129
            e       = get_feature(atoms_mlp,n=ncpu,lib='reaxff_nn')
            e_cho   = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core O core')
            e_chn   = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core N core')
            e_chc   = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core C core')

            # if f == 1:
            if f==1:
                feature = np.array([e[0],e[1],e[5],e[8],e[10],e_cho[11],e_chn[11],e_chc[11],e[12],density])
            elif f==2:
                nmol = get_nmol(atoms)
                feature = np.array([e[0],e[1],e[5],e[8],e[10],e_cho[11],e_chn[11],e_chc[11],e[12],density], dtype=float)
                feature[:-1] /= nmol      # 能量项按分子数平均, 密度不变
            else:
                feature = np.array([e[0],e[5],e[8],e[10],e[11],e[12],density])

        # ``search_structure``'s distance is a structure-matching measure; the
        # residual reported to gpevo is the raw feature-space one (§2.3 reverted).
        ind, imin, _ = search_structure(feature, D, tolerance=tolerance)
        res, imin, X_ = raw_residual(feature, D, scaler)
        res_ = float(res[imin])

        density_pred, std_den_pred = gpr_density.predict(X_, return_std=True)
        energy_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)
        energy_pred  = energy_pred[0]
        density_pred = density_pred[0]
        std_den_pred = std_den_pred[0]
        std_eng_pred = std_eng_pred[0]
        density_rf   = rfr.predict(X_)[0]
        # density_mlp  = mlp.predict(X_)[0]

        # if f == 1:
        print(f"{s:5d} -- {imin:5d} res: {res_}  "
              f"rf: {density_rf:7.4f}  "
              f"gp(den): {density_pred:7.4f} uncert: {std_den_pred:7.4f}  "
              f"gp(eng): {energy_pred:7.4f} uncert: {std_eng_pred:7.4f}" )
        # else:
        #     print(f"{s:5d} rf: {density_rf:9.4f} "
        #           f"{feature[3]:9.4f} {feature[4]:9.4f} {feature[5]:9.4f} {feature[6]:9.4f} "
        #           f"gp: {density_pred:7.4f} uncert: {std_den_pred:7.4f}")

        chdir(root_dir)
        with open("density_predict.log", "a") as fd:
             print(f"{s:5d} {res_} "
                   f"{density:7.4f} "
                   f"{density_rf:7.4f} "
                   f"{density_pred:7.4f} {std_den_pred:7.4f} "
                   f"{energy_pred:7.4f} {std_eng_pred:7.4f} " ,file=fd)

        try:
            residual = float(res_)
        except (TypeError, ValueError):
            residual = float("nan")
        results[int(s)] = {
            'id': int(s),
            'residual': residual,
            'density_mlp': float(density),
            'density_rf': float(density_rf),
            'density_gp': float(density_pred),
            'std_den': float(std_den_pred),
            'uncertainty': float(1.96 * std_den_pred),
            'energy_pred': float(energy_pred),
            'std_eng': float(std_eng_pred),
            'uncertainty_eng': float(1.96 * std_eng_pred),
        }

    return results

# ──────────────────────────────────────────────
#  calc — DFT 高通量计算
# ──────────────────────────────────────────────

def calc(t="Individuals.traj", den=1.88, ids=None, step=500,
         ncpu=8, dat="data", tolerance=0.01, gen=None):
    """
    High-throughput DFT calculation with structure matching.

    Args:
        t: trajectory file name
        den: density threshold
        ids: comma/space separated crystal indices
        step: MD steps
        ncpu: number of CPUs
        dat: data directory name
        tolerance: structure matching tolerance
        gen: generation number (None = latest generation)
    """
    images = Trajectory(t)
    if not ids:
        ids_list = []
        res = read_individuals(g=gen)
        for i, e, d, _f in res:
            if d > den and _f < 0.0:
                ids_list.append(i)
    else:
        ids_list = [int(i) for i in ids.split()]

    root_dir = getcwd()
    if not exists("density.log"):
        with open("density.log", "w") as fd:
            print("# Crystal_id Density Energy", file=fd)

    for s in ids_list:
        dir_list = root_dir.split("/")
        rootdir = "/".join(dir_list[:-1])
        data_dir = f"{rootdir}/{dat}"
        work_dir = os.path.join(root_dir, str(s))
        atoms = images[s - 1]

        if exists(str(s)):
            continue
        else:
            mkdir(str(s))

        chdir(data_dir)
        atoms_mlp, e, density = get_gulp_energy(atoms, ncpu=ncpu)
        e_cho = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core O core')
        e_chn = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core N core')
        e_chc = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core C core')

        feature = np.array([e[0], e[1], e[5], e[8], e[10], e_cho[11], e_chn[11], e_chc[11],e[12], density])

        if exists("structures.traj"):
            data = np.loadtxt("feature_mlp.csv", delimiter=",", skiprows=1)
            data_ = np.loadtxt("feature.csv", delimiter=",", skiprows=1)
            struc = Trajectory("structures.traj")
            try:
                D = data[:, 1:]
                D_ = data_[:, 1:]
            except IndexError:
                D = data[1:]
                D_ = data_[1:]
            ind, imin, res_ = search_structure(feature, D, tolerance=tolerance)
        else:
            ind = [[]]
            with open("feature_mlp.csv", "w") as fd:
                print(", etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density", file=fd)
            with open("feature.csv", "w") as fd_:
                print(", etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density", file=fd_)
            masses = np.sum(atoms.get_masses())
            volume = atoms.get_volume()
            density = masses / volume / 0.602214129
            res_ = 0.0

        chdir(work_dir)
        if len(ind[0]) > 0:
            atoms.write(f"POSCAR.{s}")
            struc[imin].write(f"POSCAR.{s}_opt")
            if D.ndim == 2:
                energy = D_[imin, 0]
                density = D_[imin, -1]
            else:
                energy = D_[0]
                density = D_[-1]
            print(f"{s:5d} mt {energy:9.4f} {feature[1]:9.4f} {feature[2]:9.4f} "
                  f"{feature[3]:9.4f} {feature[4]:9.4f} {feature[5]:9.4f} "
                  f"{feature[6]:9.4f} {density:7.4f} {res_:7.4f}")
            traj_w = TrajectoryWriter(f"id_{s}.traj", mode="w")
            traj_w.write(atoms=struc[imin])
            traj_w.close()
        else:
            subprocess.call(f"cp {rootdir}/Specific/*.psf ./", shell=True)
            img = siesta_opt(atoms, ncpu=ncpu, us="F", VariableCell="true", tstep=step,
                             xcf="GGA", xca="PBE", basistype="split")
            subprocess.call(f"mv siesta.out siesta-{s}.out", shell=True)
            subprocess.call(f"mv siesta.MDE siesta-{s}.MDE", shell=True)
            subprocess.call(f"mv siesta.MD_CAR siesta-{s}.MD_CAR", shell=True)
            subprocess.call(f"mv siesta.traj id_{s}.traj", shell=True)
            subprocess.call("rm siesta.* ", shell=True)
            subprocess.call("rm *.xml ", shell=True)
            subprocess.call("rm INPUT_TMP.* ", shell=True)
            subprocess.call("rm fdf-* ", shell=True)
            img[0].write(f"POSCAR.{s}")
            atoms_opt = img[-1]
            atoms_opt.write(f"POSCAR.{s}_opt")
            masses = np.sum(atoms_opt.get_masses())
            volume = atoms_opt.get_volume()
            density = masses / volume / 0.602214129
            energy = atoms_opt.get_potential_energy()

            print(f"{s:5d} cl {energy:9.4f} {feature[1]:9.4f} {feature[2]:9.4f} "
                  f"{feature[3]:9.4f} {feature[4]:9.4f} {feature[5]:9.4f} "
                  f"{feature[6]:9.4f} {density:7.4f} {res_:7.4f}")

            chdir(data_dir)
            with open("feature_mlp.csv", "a") as fd:
                print(f"0,{feature[0]},{feature[1]},{feature[2]},{feature[3]},"
                      f"{feature[4]},{feature[5]},{feature[6]},{feature[7]},{feature[8]},{feature[9]}", file=fd)
            with open("feature.csv", "a") as fd:
                print(f"0,{energy},{feature[1]},{feature[2]},"
                      f"{feature[3]},{feature[4]},{feature[5]},{feature[6]},{feature[7]},{feature[8]},{density}", file=fd)

            atoms_opt.calc = SinglePointCalculator(atoms_opt, energy=energy)
            with TrajectoryWriter("structures_mlp.traj", mode="a") as traj_w:
                traj_w.write(atoms=atoms_mlp)
            with TrajectoryWriter("structures.traj", mode="a") as traj_w:
                traj_w.write(atoms=atoms_opt)

        chdir(root_dir)
        with open("density.log", "a") as fd:
            print(f"{s:5d} {density:10.6f} {energy:10.8f}", file=fd)

# ──────────────────────────────────────────────
#  pack — gather POSCAR.* into USPEX POSCARS
# ──────────────────────────────────────────────

def _write_poscar_block(fposcars, atoms, id_str):
    """Append one Atoms object to an open gatheredPOSCARS file handle.

    The structure is canonicalized through pymatgen (written to a transient
    ``POSCAR``) and emitted with an ``EA<id>`` header carrying lattice info.
    """
    structure = AseAtomsAdaptor.get_structure(atoms)
    structure.to(filename="POSCAR")
    cell = atoms.get_cell()
    angles = cell.angles()
    lengths = cell.lengths()
    with open("POSCAR", "r") as f:
        lines = f.readlines()

    card = False
    for i, line in enumerate(lines):
        if line.find("direct") >= 0:
            card = True
        if card and line.find("direct") < 0:
            print(line[:-3], file=fposcars)
        elif i == 0:
            print("EA{:s} {:.6f} {:.6f} {:.6f} {:.3f} {:.3f} {:.3f} Sym.group: 1".format(id_str,
                    lengths[0], lengths[1], lengths[2],
                    angles[0], angles[1], angles[2]), file=fposcars)
        else:
            print(line[:-1], file=fposcars)


def _parse_range(range_str):
    """Parse a frame range into a list of 0-based integer indices.

    Supports: a single index (``5``), a comma list (``0,2,4``) and a
    Python-style slice (``start:stop`` or ``start:stop:step``).
    """
    range_str = range_str.strip()
    if ":" in range_str:
        parts = range_str.split(":")
        start = int(parts[0]) if len(parts) > 0 and parts[0] else 0
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        if stop is None:
            raise ValueError("range must specify a stop, e.g. 0:10")
        return list(range(start, stop, step))
    return [int(i) for i in range_str.replace(",", " ").split()]


def pack(output="POSCARS", traj=None, range_=None):
    """
    Pack structures into a single USPEX-format gatheredPOSCARS file.

    Two modes:
      * ``traj=None`` (default): scan the current directory for ``POSCAR.*``
        files and pack them (legacy behaviour).
      * ``traj=<file>``: read the given ASE trajectory and pack the frames
        selected by ``range_``.

    Args:
        output: gathered POSCARS output file name.
        traj: ASE trajectory file name; None keeps the POSCAR.* behaviour.
        range_: frame range for trajectory mode — an index (``5``), a comma
            list (``0,2,4``) or a slice (``0:10`` / ``0:10:2``).
    """
    if traj is not None:
        if range_ is None:
            raise ValueError("trajectory mode requires --range, e.g. --range=0:10")
        indices = _parse_range(range_)
        images = Trajectory(traj)
        with open(output, "a") as fposcars:
            for idx in indices:
                _write_poscar_block(fposcars, images[idx], str(idx + 1))
        return

    cdir = getcwd()
    poscars = sorted(f for f in listdir(cdir) if f.split(".")[0] == "POSCAR")

    with open(output, "a") as fposcars:
        for p in poscars:
            p_ = p.split(".")
            if len(p_) <= 1:
                continue
            i_ = p_[1]
            atoms = read(p)
            _write_poscar_block(fposcars, atoms, i_)

# ──────────────────────────────────────────────
#  update structure
# ──────────────────────────────────────────────

def update(traj,inde=None,step=1000,tolerance=0.005,ncpu=1):
    atoms_dft = read(traj,-1)
    subprocess.call('cp structures.traj structures.backup.traj',shell=True)
    subprocess.call('cp feature.csv feature.backup.csv',shell=True)
    subprocess.call('cp feature_mlp.csv feature_mlp.backup.csv',shell=True)
    masses  = np.sum(atoms_dft.get_masses())
    volume  = atoms_dft.get_volume()
    density = masses/volume/0.602214129
    energy  = atoms_dft.get_potential_energy()

    atoms = opt(atoms=atoms_dft,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')              ## compute feature
    e     = get_feature(atoms,n=ncpu,lib='reaxff_nn')
    e_cho = get_hbond_feature(atoms,n=ncpu,elements='H core C core O core')
    e_chn = get_hbond_feature(atoms,n=ncpu,elements='H core C core N core')
    e_chc = get_hbond_feature(atoms,n=ncpu,elements='H core C core C core')
    
    volume   = atoms_dft.get_volume()
    density_ = masses/volume/0.602214129
    
    data = np.loadtxt('feature_mlp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
    data_= np.loadtxt('feature.csv',delimiter=',',skiprows=1)          ## get crystal feature data
    d    = data[:,1:]         # 去掉索引
    d_   = data_[:,1:]   
     
    images     = Trajectory('structures.backup.traj')
    # images_  = Trajectory('structures_mlp.traj')
    feature = np.array([e[0], e[1], e[5], e[8], e[10], e_cho[11], e_chn[11], e_chc[11],e[12], density])

    res  = np.sum(np.square(d - feature),axis=1)
    ind  = np.where(res<tolerance)
    if inde is None:
       ind_ = ind[0]
    else:
       ind_ = [inde]

    if len(ind_)>0:
       his  = TrajectoryWriter('structures.traj',mode='w')
       fd    = open('feature.csv','w') 
       print(', etot, eang, etor, evdw, ehb, ecoul, density',file=fd)
       for i,d in enumerate(d_):
           if i in ind_:
              print(i,data_[i][1],energy,data_[i][7],density)
              print(i,',',energy,',',d[1],',',d[2],',',d[3],',',d[4],',',d[5],',',d[4],',',d[7],',',d[8],',',density,file=fd)  
              his.write(atoms=atoms_dft)
           else:  
              print(i,',',d[0],',',d[1],',',d[2],',',d[3],',',d[4],',',d[5],',',d[6],',',d[7],',',d[8],',',d[9],file=fd)  
              his.write(atoms=images[i])
       fd.close()
       his.close()
    else:
       print(f'Specified structure not found in database!') 
       print(f'energy: {energy}')
    

# ──────────────────────────────────────────────
#  traj — POSCAR to trajectory
# ──────────────────────────────────────────────

def traj(fposcar="gatheredPOSCARS"):
    """Convert gatheredPOSCARS to ASE trajectory file.

    安全机制: gatheredPOSCARS 的结构块数可能与 Individuals 的能量条数
    不一致 (中途崩溃/手动增删结构)。能量缺失时用 0 占位并告警, 不再越界中断。
    """
    from .utils import Stack

    with open(fposcar) as fbp:
        lines = fbp.readlines()

    traj_w = TrajectoryWriter("Individuals.traj", mode="w")
    k = 0
    s = 0
    energies = []

    with open("Individuals") as f:
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
                energies.append(float(l[3]))
        st.close()

    def get_energy(idx):
        if idx < len(energies):
            return energies[idx], False
        return 0.0, True

    n_missing = 0
    for line in lines:
        if "EA" in line:
            if k > 0:
                fpos.close()
                atoms = read("POSCAR")
                energy, missing = get_energy(s)
                if missing:
                    n_missing += 1
                    print(f"警告: gatheredPOSCARS 第 {k} 个结构在 Individuals 中"
                          f"无对应能量 (仅 {len(energies)} 条), 能量置 0")
                atoms.calc = SinglePointCalculator(atoms, energy=energy)
                traj_w.write(atoms=atoms)
                s += 1
            fpos = open("POSCAR", "w")
            print(line[:-1], file=fpos)
            k += 1
        else:
            print(line[:-1], file=fpos)

    fpos.close()
    atoms = read("POSCAR")
    energy, missing = get_energy(s)
    if missing:
        n_missing += 1
        print(f"警告: gatheredPOSCARS 第 {k} 个结构在 Individuals 中"
              f"无对应能量 (仅 {len(energies)} 条), 能量置 0")
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    traj_w.write(atoms=atoms)
    traj_w.close()

    print(f"✓ 转换完成: {k} 个结构, {len(energies)} 条能量"
          + (f", 其中 {n_missing} 个能量缺失已置 0" if n_missing else ", 数量一致"))

# ──────────────────────────────────────────────
#  zmat — 内坐标
# ──────────────────────────────────────────────

def zmat(geo="POSCAR", i=-1, ffield=None):
    """Convert structure to USPEX Z-matrix format."""
    from irff.AtomDance import AtomDance
    atoms = read(geo, index=i)
    if ffield is None:
        ffield = _resolve_ffield()
    ad = AtomDance(atoms=atoms, rcut={"H-O": 2.7, "O-H": 2.7}, ffield=ffield)
    zmat_data = ad.InitZmat
    ad.write_zmat(zmat_data, uspex=True)
    ad.close()

# ──────────────────────────────────────────────
#  fdf — 写 SIESTA 输入
# ──────────────────────────────────────────────

def fdf(gen="poscar.gen", xcf="gga", i=-1):
    """Generate SIESTA input files."""
    from irff.dft.siesta import write_siesta_in
    A = read(gen, index=i)
    print("\n-  writing siesta input ...")
    if xcf == "gga":
        write_siesta_in(A, coord="cart", md=False, opt="CG",
                        VariableCell="true", xcf="GGA", xca="PBE", basistype="split")
    elif xcf == "vdw":
        write_siesta_in(A, coord="cart", md=False, opt="CG",
                        VariableCell="true", xcf="VDW", xca="DRSLL", basistype="split")
    else:
        print("Not supported yet!")

# ──────────────────────────────────────────────
#  info - 打印能量与晶格常数信息
# ──────────────────────────────────────────────

def info(gen=None, traj=None, i=-1, symmetry=True, symprec=0.1):
    """Print energy, lattice, and symmetry information of a structure.

    Args:
        gen: geometry file name (e.g. POSCAR, gulp.cif)
        traj: trajectory file name
        i: frame index (default: -1, last frame)
        symmetry: if True, perform pymatgen symmetry analysis (default: True)
        symprec: symmetry tolerance for pymatgen (default: 0.1)
    """
    if gen is not None:
        atoms = read(gen)
    elif traj is not None:
        images = Trajectory(traj)
        atoms = images[i]
    else:
        print("Error: please specify a structure file (--gen) or trajectory (--traj)")
        return

    cell = atoms.get_cell()
    lengths = cell.lengths()
    angles = cell.angles()
    volume = atoms.get_volume()
    masses = np.sum(atoms.get_masses())
    density = masses / volume / 0.602214129
    natoms = len(atoms)
    formula = atoms.get_chemical_formula()

    print(f"\n─ Structure Info ─────────────────────────────")
    print(f"  Formula:          {formula}")
    print(f"  Number of atoms:  {natoms}")
    print(f"\n─ Lattice ────────────────────────────────────")
    print(f"  a = {lengths[0]:12.6f} Å")
    print(f"  b = {lengths[1]:12.6f} Å")
    print(f"  c = {lengths[2]:12.6f} Å")
    print(f"  α = {angles[0]:12.6f}°")
    print(f"  β = {angles[1]:12.6f}°")
    print(f"  γ = {angles[2]:12.6f}°")
    print(f"  Volume  = {volume:12.4f} ų")
    print(f"  Density = {density:12.6f} g/cm³")

    # ── Oxygen Balance (CHNO explosives) ──
    _print_oxygen_balance(atoms)

    try:
        energy = atoms.get_potential_energy()
        print(f"\n─ Energy ─────────────────────────────────────")
        print(f"  Total energy = {energy:16.8f} eV")
        if natoms > 0:
            print(f"  Energy/atom  = {energy / natoms:16.8f} eV/atom")
    except Exception:
        print(f"\n─ Energy ─────────────────────────────────────")
        print("  (no energy data available)")

    # ── Symmetry Analysis (pymatgen) ──
    if symmetry:
        _print_symmetry_info(atoms, symprec)

    print(f"──────────────────────────────────────────────\n")


def oxygen_balance(counts, products="CO2"):
    """Oxygen balance of a CHNO explosive.

    OB = -1600/M * (2*n_C + n_H/2 - n_O)   [%]

    where n_C, n_H, n_O are atom counts in the molecular formula and M is
    the molar mass (g/mol).  Assumes complete oxidation to CO2 + H2O;
    positive OB means oxygen excess, negative means oxygen deficiency.

    Args:
        counts: dict mapping element symbol -> atom count (e.g. {'C': 6, 'H': 2, 'N': 2, 'O': 6})
        products: 'CO2' (default, CO2 + H2O products) or 'CO' (CO + H2O products).
            For the CO convention the carbon term is n_C instead of 2*n_C.

    Returns:
        OB percentage (float), or None if no C/H/O atoms present.
    """
    nC = counts.get("C", 0)
    nH = counts.get("H", 0)
    nO = counts.get("O", 0)
    if nC + nH + nO == 0:
        return None

    atomic_mass = {"H": 1.00794, "C": 12.011, "N": 14.007, "O": 15.999}
    molar_mass = sum(atomic_mass.get(el, _mass_of(el)) * n
                     for el, n in counts.items())

    c_need = nC if products == "CO" else 2 * nC
    ob = -1600.0 / molar_mass * (c_need + nH / 2.0 - nO)
    return ob


def _mass_of(symbol):
    """Fallback atomic mass lookup for elements beyond CHNO."""
    from ase.data import atomic_masses, atomic_numbers
    return atomic_masses[atomic_numbers[symbol]]


def _print_oxygen_balance(atoms):
    """Print oxygen balance for a structure (reduced to molecular formula)."""
    symbols = atoms.get_chemical_symbols()
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1

    # reduce to the smallest integer formula unit (gcd of all counts)
    from math import gcd
    from functools import reduce
    g = reduce(gcd, counts.values())
    reduced = {el: n // g for el, n in counts.items()} if g > 1 else counts

    ob_co2 = oxygen_balance(reduced, products="CO2")
    ob_co = oxygen_balance(reduced, products="CO")
    if ob_co2 is None:
        return

    formula_str = "".join(f"{el}{(n if n > 1 else '')}"
                          for el, n in sorted(reduced.items()))
    print(f"\n─ Oxygen Balance ─────────────────────────────")
    print(f"  Molecular formula: {formula_str}  (unit-cell formula / {g})")
    print(f"  OB (CO2 + H2O products): {ob_co2:8.2f} %")
    print(f"  OB (CO  + H2O products): {ob_co:8.2f} %")


def _print_symmetry_info(atoms, symprec):
    """Print symmetry analysis using pymatgen SpacegroupAnalyzer.

    Args:
        atoms: ASE Atoms object
        symprec: symmetry tolerance for pymatgen
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        print(f"\n─ Symmetry ───────────────────────────────────")
        print("  (pymatgen not available — install pymatgen)")
        return

    try:
        # Convert ASE Atoms → pymatgen Structure directly (no temp file)
        struct = Structure.from_ase_atoms(atoms)

        analyzer = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=5.0)

        spg_symbol = analyzer.get_space_group_symbol()
        spg_number = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system()
        point_group = analyzer.get_point_group_symbol()
        hall = analyzer.get_hall()
        lattice_type = analyzer.get_lattice_type()
        has_inversion = analyzer.is_laue()
        num_sym_ops = len(analyzer.get_symmetry_operations())
        sym_struct = analyzer.get_symmetrized_structure()

        print(f"\n─ Symmetry (pymatgen, symprec={symprec}) ──────")
        print(f"  Space Group (HM):    {spg_symbol}")
        print(f"  Space Group Number:  {spg_number}")
        print(f"  Hall Symbol:         {hall}")
        print(f"  Crystal System:      {crystal_system}")
        print(f"  Point Group:         {point_group}")
        print(f"  Lattice Type:        {lattice_type}")
        print(f"  Laue (inversion):    {'yes' if has_inversion else 'no'}")
        print(f"  Symmetry Operations: {num_sym_ops}")
        print(f"  Wyckoff Sites:       {len(sym_struct.wyckoff_symbols)}")

    except Exception as e:
        print(f"\n─ Symmetry ───────────────────────────────────")
        print(f"  (analysis failed: {e})")



# ──────────────────────────────────────────────
#  sample - 采样结构
# ──────────────────────────────────────────────

def sample(ind="", t=None):
    """Sample structures by index to samples.traj."""
    traj_w = TrajectoryWriter("samples.traj", mode="w")

    if ind:
        ids = [int(i) for i in ind.split()]
        if t is not None:
            images = Trajectory(t)
            for i in ids:
                traj_w.write(atoms=images[i])
        else:
            for i in ids:
                atoms = read(f"{i}/POSCAR.{i}_opt")
                atoms.calc = SinglePointCalculator(atoms, energy=0.0)
                traj_w.write(atoms=atoms)

    traj_w.close()


# ──────────────────────────────────────────────
#  lib — 将 ffield.json 转换为 reaxff_nn.lib
# ──────────────────────────────────────────────

def lib(jsonfile="ffield.json", libfile="reaxff_nn.lib"):
    """
    Convert ffield.json to reaxff_nn.lib format.

    Args:
        jsonfile: path to ffield.json (default: ffield.json)
        libfile:  output lib file name (default: reaxff_nn.lib)
    """
    import json as js
    from irff.reaxfflib import write_lib

    with open(jsonfile, "r") as lf:
        j = js.load(lf)
    p_ = j["p"]
    m_ = j["m"]
    vdw_layer = j.get("vdw_layer", None)
    mf_layer = j.get("mf_layer", None)
    be_layer = j.get("be_layer", None)

    spec, bonds, offd, angs, torp, hbs = _init_bonds(p_)
    write_lib(p_, spec, bonds, offd, angs, torp, hbs,
              m=m_, mf_layer=mf_layer, be_layer=be_layer,
              vdw_layer=vdw_layer, libfile=libfile)
    print(f"✓  Converted {jsonfile} → {libfile}")


def _init_bonds(p_):
    """Categorize force-field parameters from JSON keys."""
    spec, bonds, offd, angs, torp, hbs = [], [], [], [], [], []
    for key in p_:
        k = key.split("_")
        if k[0] == "bo1":
            bonds.append(k[1])
        elif k[0] == "rosi":
            kk = k[1].split("-")
            if len(kk) == 2:
                if kk[0] != kk[1]:
                    offd.append(k[1])
            elif len(kk) == 1:
                spec.append(k[1])
        elif k[0] == "theta0":
            angs.append(k[1])
        elif k[0] == "tor1":
            torp.append(k[1])
        elif k[0] == "rohb":
            hbs.append(k[1])
    return spec, bonds, offd, angs, torp, hbs


# ──────────────────────────────────────────────
#  ffield — 将 ffield.json 转换为 ffield
# ──────────────────────────────────────────────

def ffield(jsonfile="ffield.json", ffieldfile="ffield"):
    """
    Convert ffield.json to ReaxFF ffield format.

    Args:
        jsonfile:   path to ffield.json (default: ffield.json)
        ffieldfile: output ffield file name (default: ffield)
    """
    import json as js
    from irff.reaxfflib import write_ffield

    with open(jsonfile, "r") as lf:
        j = js.load(lf)
    p_ = j["p"]
    m_ = j["m"]
    mf_layer = j.get("mf_layer", None)
    be_layer = j.get("be_layer", None)

    spec, bonds, offd, angs, torp, hbs = _init_bonds(p_)
    write_ffield(p_, spec, bonds, offd, angs, torp, hbs,
                 m=m_, mf_layer=mf_layer, be_layer=be_layer,
                 libfile=ffieldfile)
    print(f"✓  Converted {jsonfile} → {ffieldfile}")


# ──────────────────────────────────────────────
#  molinfo — 分子识别与原子分组信息
# ──────────────────────────────────────────────

def molinfo(gen="data.traj", i=-1, jsonfile=None, verbose=True):
    """
    Print molecule atom indices for each molecule in the structure.

    Uses the IRFF Molecules module to identify molecules based on bond
    cutoffs from ffield.json (or defaults for CHON systems).

    Args:
        gen:      geometry file (ASE-readable: traj, POSCAR, xyz, lammps-data, etc.)
        i:        frame index in trajectory (default: -1 = last frame)
        jsonfile: path to ffield.json for rcutBond parameters (optional)
        verbose:  print molecule boundary info (default: True)
    """
    import json as js
    from ase.io import read
    from irff.molecule import Molecules, press_mol

    # ── Load structure ──
    if gen.endswith('.lammps') or gen.endswith('.lammps-data'):
        atoms = read(gen, format='lammps-data', atom_style='charge')
    else:
        atoms = read(gen, index=i)

    atoms = press_mol(atoms)

    # ── Build rcut dict from ffield.json or use defaults ──
    rcut = {}
    if jsonfile is not None:
        try:
            with open(jsonfile, 'r') as lf:
                j = js.load(lf)
            rcut = j.get('rcutBond', {})
            if verbose:
                print(f"# Loaded bond cutoffs from {jsonfile}")
        except FileNotFoundError:
            if verbose:
                print(f"# Warning: {jsonfile} not found, using defaults")
        except Exception as e:
            if verbose:
                print(f"# Warning: failed to read {jsonfile}: {e}, using defaults")

    if not rcut:
        # Default rcut for CHON systems (ReaxFF-style)
        rcut = {
            "H-O": 1.2, "H-C": 1.2, "H-H": 0.8, "H-N": 1.2,
            "O-N": 1.45, "O-C": 1.45, "O-O": 1.4,
            "N-C": 1.68, "N-N": 1.68,
            "C-C": 1.68,
            "others": 1.8,
        }
        if verbose:
            print("# Using default bond cutoffs for CHON systems")

    if verbose:
        print(f"# rcut: {rcut}")
        print(f"# Total atoms: {len(atoms)}")
        print(f"# Species: {sorted(set(atoms.get_chemical_symbols()))}")

    # ── Identify molecules ──
    mols = Molecules(atoms, rcut=rcut, check=True)
    nmol = len(mols)

    if verbose:
        print(f"# Number of molecules: {nmol}")

    # ── Print molecule atom indices (1-based for LAMMPS compatibility) ──
    for i_mol, mol in enumerate(mols):
        idx = np.array(mol.mol_index) + 1  # 1-based
        nat = len(idx)
        if verbose:
            print(f"# Mol {i_mol+1}: {nat} atoms")
        print(' '.join(str(x) for x in idx))

    return mols

def _resolve_ffield():
    if os.path.isfile("ffield.json"):
        return "ffield.json"
    bundled = os.path.join(os.path.dirname(__file__), "data", "ffield.json")
    if os.path.isfile(bundled):
        return bundled
    return "ffield.json"
