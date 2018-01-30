#!/usr/bin/env python
# -*- coding: utf-8 -*-

from vecteur import vecteur

class classKite:
    """
    Dans cette classe, on va définir toutes les caractéristiques du Kite et du cable
    """
    def __init__(self):
        self.lcable = 50 # Longueur du câble
        self.S = 18  # Surface du kite
        self.alpha = 0 # Angle d'attaque du kite
        self.pos = vecteur()
        self.Tmax = 2700 # Tension maximale supportable du câble, équivalent à tirer deux bateaux
        self.elevation = 0 # Angle du câble avec la surface
        
