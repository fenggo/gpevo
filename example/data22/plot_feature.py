#!/usr/bin/env python
import subprocess
import sys
import argparse
import numpy as np

import matplotlib.pyplot as plt
# from irff.molecule import press_mol


parser = argparse.ArgumentParser(description='options')
parser.add_argument('--n', default=8,type=int, help='ncpu')
parser.add_argument('--p', default=1,type=int, help='plot which energy. 1: total 2:eangle 3: etor 4: evdw 5: ehb 6:ecoul') 
args = parser.parse_args(sys.argv[1:])
p    = args.p

data = np.loadtxt('feature.csv',delimiter=',',skiprows=1) 
E = data[:,p]
D = data[:,-1]
label = ['none','Etotal','Eangle', 'Etorsion','Evdw','Ehbond','Ecoul']

E_,D_ = [],[]
for d_,e_ in zip(D,E):
    if d_>1.75 and abs(e_-E[0])<1.5:
       E_.append(e_)
       D_.append(d_)
       
plt.figure()   
plt.rcParams['ytick.labelsize'] = 9 
plt.rcParams['xtick.labelsize'] = 9  
plt.xlabel(r'$Density$ ($g/cm^3$)')
# plt.xlim(1.75,1.875)
plt.ylabel(r'${:s}$ ($eV$)'.format(label[p]))

plt.scatter(D_,E_,alpha=0.8,
            edgecolor='#035f37', s=35,marker='s',
            color='none',
            label=label[p])
# x = np.linspace(0.59,0.77)
# y = x*0.67 + 1.41
# plt.plot(x,y,color='k',linestyle='-.')
plt.legend(loc='best',edgecolor='yellowgreen',ncol=3,fontsize=9) # loc = lower left upper right best
plt.savefig('{:s}.pdf'.format(label[p]),transparent=True) 
plt.close()

d   = data[:,1:]    # 去掉索引
cry = d[1,:]
d   = d[2:,:]
# print(cry)

res = np.sum(np.square(d - cry),axis=1)
ind = np.where(res<0.01)
# if len(ind[0])>0:
#    print(res[ind]) 
#    print(E[ind])
plt.figure()   
plt.rcParams['ytick.labelsize'] = 10  
plt.xlabel(r'$Crystal$ $ID$')
plt.ylabel(r'$Residual$')
 
plt.plot(res,label='Residual')

plt.legend(loc='best',edgecolor='yellowgreen',ncol=3,fontsize=9) # loc = lower left upper right best
plt.savefig('residual.pdf',transparent=True) 
plt.close()

