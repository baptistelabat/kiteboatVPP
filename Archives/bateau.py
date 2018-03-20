#!/usr/bin/env python
# -*- coding: utf-8 -*-

from vecteur import vecteur
import numpy as np

class classbateau:
    """
    Dans cette classe on va définir toutes les caractéristiques du bateau
    """
    def __init__(self):
        self.vitesseabs = 0 # Vitesse de déplacement du navire dans le repère terrestre
        self.omega 	= 0 # Angle de déplacement du navire par rapport à l'axe x =tan(Vy/Vx)
        self.pos = vecteur()
        self.vit = vecteur()
        self.acc = vecteur()
        self.hder = 1 # Taille de la dérive
        self.lder = 2
        self.M = 200 # Masse du navire
        # On définit les inerties du bateau
        self.I44 = 5
        self.I46 = 0
        self.I55 = 5
        self.I64 = 0
        self.I66 = 500
        self.A = np.array([[self.M,      0, 	   0, 	     0, 		   0, 		   0],
                          [      0, self.M,	     0,	       0,		     0,		     0],
                          [      0,	     0,	self.M,	       0,		     0,		     0],
                          [      0,	     0,	     0,	self.I44,	       0, self.I46],
                          [      0,	     0,	     0,	       0, self.I55,	       0],
                          [      0,	     0,	     0,	self.I64,	       0, self.I66]])
        self.L = 15
        self.uzeros = 0
