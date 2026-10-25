#!/usr/bin/env python
from os import getcwd, listdir
from os.path import isdir
import sys
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from ase.io import read 
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol,Molecules,enlarge # , moltoatoms
from irff.md.gulp import opt
from irff.deb.compare_energies import deb_gulp_energy

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='Individuals.traj',type=str, help='trajectory file')
parser.add_argument('--o', default=0,type=int, help='whether optimize the crystal structure')
parser.add_argument('--p', default=200,type=int, help='step of optimization')
parser.add_argument('--n', default=8,type=int, help='ncpu')
args = parser.parse_args(sys.argv[1:])

def num_mol(A,species=None):
    ff = [1.0,5.0] #,1.9 ,2.0,2.5,3.0,3.5,4.0
    cell = A.get_cell()

    m_  = Molecules(A,rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,
                            "O-O":1.4,"others": 1.68},
                    species=species,check=True)
    nmol = {}
    for m in m_:
        if m.label in nmol:
           nmol[m.label] += 1
        else:
           nmol[m.label]  = 1
    for f in ff:
        m = copy.deepcopy(m_)
        m,A = enlarge(m,cell=cell,fac=f,supercell=[1,1,1])
    # print('NM: ',nmol,'\nDhb',ehb[0],ehb[-1])
    return nmol

gens  = listdir(getcwd())
gens  = ['{:s}/POSCAR.{:s}_opt'.format(gen,gen) 
         for gen in gens if isdir(gen) and not gen.startswith('.') 
                                       # and not gen.startswith('tf') 
                                       and not gen.startswith('gens')]
# gens  = ['{:s}/POSCAR.{:s}_opt'.format(gen,gen) 
#          for gen in gens if isdir(gen) and not gen.startswith('.') and not gen.startswith('gens')]
atoms = read(gens[0])

E,Ehb,D = {},{},{}
I,Ebind = {},{}
Eb      = {}
eb,eb_per_mol,emol = 0.0, 0.0, 0.0

with open('ebind.dat','r') as f:
   for line in f:
       if line.startswith('#') or len(line)<1:
          continue
       l       = line.split()
       lab     = l[0] 
       Ebind[lab] = -float(l[3]) 

with open('hbond.dat','w') as fd:
     print('# Crystal_id hbond_energy binding_energy density',file=fd)

hide   = ['tf433']
show   = ['d3','k1350','k1705','l1671','g1942','g1337','i937','196','197','a960','198','15',
          'tnt','fox7','cl20','exp24','hmx']

for i,gen in enumerate(gens):
    # print(gen)
    s_ = gen.split('.')[1]
    s  = s_.split('_')[0]

    if s in ['b1792']:
       continue
    if show:
       if s not in show:
          continue
    atoms = read(gen)
    if args.o:
       atoms = opt(atoms=atoms,step=args.p,l=1,t=0.0000001,n=args.n, x=1,y=1,z=1)

    atoms  = press_mol(atoms)
    x      = atoms.get_positions()
    m      = np.min(x,axis=0)
    x_     = x - m
    atoms.set_positions(x_)
    atoms.write('gens/{:s}.gen'.format(s))
    
    mols   = num_mol(atoms)
    e      = Ebind[s]
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density= masses/volume/0.602214129

    nmol = 0
    for mol in mols:
        nmol += mols[mol]
    
    if 'C2N4H4O4' in mols and 'C6N12O12H6' in mols:
       label = '{:d}:{:d}'.format(mols['C6N12O12H6'],mols['C2N4H4O4'])
       if mols['C6N12O12H6'] == mols['C2N4H4O4']:
          label = '1:1'
    elif 'O4N4C2H4' in mols:
       label = 'FOX-7'
    elif 'C4H8N8O8' in mols and 'C6N12O12H6' in mols:
       label = '2CL-20/1HMX(exp.)'
    elif 'C6N12O12H6' in mols and 'C7H5N3O6' in mols:
       label = 'CL-20/TNT'
    elif 'C7H5N3O6' in mols or 'O6N3C7H5' in mols:
       label = 'TNT'
    elif 'C6H6N12O12' in mols:
       label = 'CL-20'
    elif 'C6N12O12H6' in mols:
       label = 'CL-20/HMX'
    elif 'C4H8N8O8' in mols:
       label = 'HMX'
    else:
       label = 'Others'
    # print(mols,label)

    if label in I:
       I[label].append(s)
       # Ehb[label].append(ehb/nmol)
       Eb[label].append(e/nmol)
       D[label].append(density)
    else:
       I[label]   = [s]
       # Ehb[label] = [ehb/nmol]
       Eb[label]  = [e/nmol]
       D[label]   = [density]
    # eb = e-ehb
    print('*{:6s}* NM: {:2d}, ebind: {:8.4f},'
          ' Density: {:9.6}'.format(s,nmol,e,density))
    with open('bind.dat','a') as fd:
         print(s,e,e/nmol,density,file=fd) 

# ── Publication-quality matplotlib configuration ──────────────────────────
plt.rcParams.update({
    'font.family':         'serif',
    'font.serif':          ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size':           11,
    'mathtext.fontset':    'stix',          # math font matching Times
    'axes.linewidth':      0.8,
    'axes.labelsize':      12,
    'axes.titlesize':      12,
    'xtick.labelsize':     10,
    'ytick.labelsize':     10,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.major.size':    4,
    'ytick.major.size':    4,
    'xtick.minor.size':    2.5,
    'ytick.minor.size':    2.5,
    'xtick.major.width':   0.8,
    'ytick.major.width':   0.8,
    'xtick.top':           True,
    'ytick.right':         True,
    'legend.fontsize':     9,
    'legend.frameon':      True,
    'legend.edgecolor':    'black',
    'legend.fancybox':     False,
    'legend.borderpad':    0.5,
    'legend.handletextpad': 0.4,
    'legend.columnspacing': 1.0,
    'figure.dpi':          300,
    'savefig.dpi':         600,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.02,
})

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 4.0))   # single-column friendly

markers= {'1:3':'^','1:4':'v','1:2':'>','1:1':'s','1:5':'<','1:6':'p','2:2':'D',
           '2:1':'d','3:1':'P','4:1':'h', 'CL-20/HMX':'*',
           '2CL-20/1HMX(exp.)':'8','FOX-7':'P','Others':'X','CL-20':'X',
           'HMX':'P','CL-20/TNT':'h',
           'TNT':'P'}
full   = ['HMX','FOX-7','CL-20','TNT']

# Professional colour palette (colourblind-friendly, distinguishable in print)
colors = {'1:3':'#1f77b4','1:4':'#aec7e8','1:2':'#ff7f0e','1:1':'#2ca02c',
          '1:5':'#98df8a','1:6':'#d62728','2:2':'#9467bd',
          '2:1':'#8c564b','3:1':'#e377c2','4:1':'#7f7f7f','FOX-7':'#17becf',
          'CL-20/HMX':'#bcbd22',
          '2CL-20/1HMX(exp.)':'#393b79','Others':'#8c8c8c','CL-20':'#ad494a',
          'CL-20/TNT':'#e7969c','HMX':'#31a354',
          'TNT':'#3182bd'}

left   = ['h1218','g1942','c641','g909','tf456','m968']
right  = ['f240','g1446','g1255','198']
labels = ['1:1','1:2','1:3','CL-20/TNT','FOX-7','CL-20/HMX','CL-20','HMX','TNT']

for label in labels:
    eb   = Eb[label]
    d    = D[label]
    if not d:
       continue
    mark = markers[label]
    if label=='1:1':
       lab_ = '1:1(CL-20/FOX-7)'
    elif label=='1:2':
       lab_ = '1:2(CL-20/FOX-7)'
    elif label=='1:3':
       lab_ = '1:3(CL-20/FOX-7)'
    else:
       lab_ = label
    if label in full:
       ax.scatter(eb, d, alpha=0.85,
                  color=colors[label], s=40, marker=mark,
                  edgecolors='black', linewidths=0.4,
                  label=lab_, zorder=5)
    else:
       ax.scatter(eb, d, alpha=0.85,
                  edgecolors=colors[label], s=40, marker=mark,
                  facecolors='none', linewidths=1.0,
                  label=lab_, zorder=5)

    for i, lb in enumerate(I[label]):
        if lb == 'fox7':
           lb_ = r'$\alpha$-FOX-7'
        elif lb == 'cl20':
           lb_ = r'$\varepsilon$-CL-20'
        elif lb == 'exp24':
           lb_ = '2CL-20/1HMX\n(Bolton)'
        elif lb == 'hmx':
           lb_ = r'$\beta$-HMX'
        elif lb == 'tnt':
           lb_ = 'TNT'
        else:
           lb_ = lb.upper()
        if lb in hide:
           continue
        if show:
           if lb not in show:
              continue
        if lb in left:
           ax.text(eb[i] - 0.008, d[i] + 0.003, lb_, ha='center', va='top',
                   fontsize=5.5, color='dimgray')
        elif lb in right:
           ax.text(eb[i] + 0.008, d[i] + 0.008, lb_, ha='center', va='top',
                   fontsize=5.5, color='dimgray')
        else:
           ax.text(eb[i], d[i] + 0.003, lb_, ha='center', va='bottom',
                   fontsize=5.5, color='dimgray')

# ── Axis labels & formatting ──────────────────────────────────────────────
ax.set_xlabel(r'$Binding\ Energy\ per\ Molecule\ (eV)$', fontsize=12)
ax.set_ylabel(r'$Density\ (g\ cm^{-3}$)', fontsize=12)

# Remove top and right spines for a cleaner look
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# Keep inward ticks only on bottom/left after removing top/right spines
ax.tick_params(axis='both', which='both', direction='in', top=False, right=False)

# ── Legend ────────────────────────────────────────────────────────────────
leg = ax.legend(loc='lower right', ncol=2, fontsize=8.5,
                frameon=True, edgecolor='black', framealpha=0.9,
                borderpad=0.4, handletextpad=0.3, columnspacing=0.8,
                handlelength=1.2)

# ── Save ──────────────────────────────────────────────────────────────────
plt.tight_layout()
# fig.savefig('ebind.pdf')    # vector — preferred for journal submission
fig.savefig('ebind.svg')    # vector — editable
# fig.savefig('ebind.png', dpi=600)  # high-res raster for preview
plt.close()
