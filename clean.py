#!/usr/bin/env python
from os.path import isfile
from os import system

if __name__ == '__main__':
   system('rm -r AntiSeeds')
   system('rm -r CalcFold1')
   system('rm -r CalcFoldTemp')
   #system('rm -r Seeds')
   system('rm *.mat')
   system('rm *.backup')
   # system('rm -r results1')
   if isfile('still_running'):
      system('rm still_running')
   if isfile('USPEX_IS_DONE'):
      system('rm USPEX_IS_DONE')
   if isfile('octave-workspace'):
      system('rm octave-workspace')
   if isfile('Warnings'):
      system('rm Warnings')
