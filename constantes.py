# -*- coding: utf-8 -*-
"""
On définit les variables globales du problème
Caractéristiques du bateau, du vent, etc...
"""
import numpy as np

M = 2000 # Masse du navire
g = 9.81
uzeros = 100 # Vitesse initiale
#uvent = 30;
h = 3
l = 5
Cx = 0.01;
Skite = 10; 
Cz = 1;
rho = 1;
rhoeau = 1000;
L = 15; #On considère un bateau de 15m
dirvent = 0
psi = np.pi + 5*np.pi/180#angle de cap
vitessevent = 30
Sbateau = 5
Sderive = 1
Sgouv   = 0.5
