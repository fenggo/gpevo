#!/usr/bin/env python
import subprocess
import numpy as np
from ase.io.trajectory import Trajectory
import random
from pymatgen.io.ase import AseAtomsAdaptor


def select_random_structure(density=1.82,num_to_select=100):
    """删除指定目录中随机一半的文件"""
    # 获取当前目录下的所有文件（不包括子目录）
    images = Trajectory('structures.traj')
    data   = np.loadtxt('feature.csv',delimiter=',',skiprows=1)  ## get crystal feature data
    imgs   = [i for i in range(len(images)) if data[i][-1]>=density or data[i][5]<-0.285]
    print('{:d} structures have been chosen as candidate.'.format(len(imgs)))

    if not imgs:
        print("当前目录没有可删除的文件。")
        return

    # 随机选择结构
    if len(imgs)>num_to_select:
       atoms_to_select = random.sample(imgs, num_to_select)
    else:
       atoms_to_select = imgs
    print('{:d} structures have been selected.'.format(len(atoms_to_select)))
    # 将结构写入文件
    fposcars = open('POSCARS','w')
    for i,atoms in enumerate(images):
        if i in atoms_to_select:
            structure = AseAtomsAdaptor.get_structure(atoms)
            structure.to(filename="POSCAR")
            cell = atoms.get_cell()
            angles = cell.angles()
            lengths = cell.lengths()
            with open('POSCAR','r') as f:
                 lines = f.readlines()

            card = False
            for i_,line in enumerate(lines):
                if line.find('direct')>=0:
                   card = True
                if card and line.find('direct')<0:
                   print(line[:-3],file=fposcars)
                elif i_==0:
                   print('EA{:d} {:.6f} {:.6f} {:.6f} {:.3f} {:.3f} {:.3f} Sym.group: 1'.format(i,
                           lengths[0],lengths[1],lengths[2],
                           angles[0],angles[1],angles[2]),file=fposcars)
                else:
                   print(line[:-1],file=fposcars)

    fposcars.close()

if __name__ == "__main__":
    ''' select random structure to use as seeds '''
    select_random_structure(density=1.83,num_to_select=119)

