#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class classvent:
    """
    On va récupérer ici toute les caractéristiques propres au vent
    """
    def __init__(self):
        self.vitesse 		= 15*1.8
        self.direction 		= np.pi/12
        self.vitesseapparente 	= 0
        self.directionapparente = 0
        self.rho 		= 1
        
