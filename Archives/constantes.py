#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
On définit les variables globales du problème
Caractéristiques du bateau, du vent, etc...
"""
import numpy as np

M = 2000 # Masse du navire
g = 9.81
uzeros = 100 # Vitesse initiale
I44=5; I46=0; I55=5; I64=0; I66=5; #On initialise les inerties
uvent = 30;
h = 3
l = 5
Cx = 0.01;
Cz = 1;
rho = 1;
rhoeau = 1025
L = 15; # On considère un bateau de 15m
dirvent = np.pi/2
