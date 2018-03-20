# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:23:03 2018

@author: apruchon2016
"""

import os
folder = "Z:/Projet_Kite/Code"
os.chdir(folder)
import numpy as np
from scipy import interpolate
#from scipy import *
#import matplotlib.pyplot as plt

filename        = '../Code/DU21_A17.dat'
coefsaero       =np.genfromtxt(filename, dtype=float, skip_header=14)


AoAaero_index, CLaero_index, CDaero_index  = range(3)
coefsaero[:,AoAaero_index]=coefsaero[:,AoAaero_index]*np.pi/180

filename        = '../Code/NACA0015.txt'
coefshydro      =np.genfromtxt(filename, dtype=float,     skip_header=3)


AoAhydro_index, CLhydro_index, CDhydro_index, CMhydro_index  = range(4)
coefshydro[:,AoAhydro_index]=coefshydro[:,AoAhydro_index]*np.pi/180 #On convertit les angles en radians
tckCLderive = interpolate.splrep(coefshydro[:, AoAhydro_index], coefshydro[:, CLhydro_index]) #Meme notation que dans la doc scipy
tckCDderive = interpolate.splrep(coefshydro[:, AoAhydro_index], coefshydro[:, CDhydro_index]) #Meme notation que dans la doc scipy
