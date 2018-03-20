#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def reynolds(u, rho, L):
    visc = 1.07*10**(-3)
    Re = u*rho*L/visc
    return Re
    
def Cfbateau(Re):
    if Re == 0:
        return 0
    else:        
        Cf = 0.0075/(np.log10(Re) -2)**2
        return Cf
    
def b(g, F, bateau):
    uzeros = bateau.uzeros
    m 	  = bateau.M
    theta = bateau.pos.ry
    phi	  = bateau.pos.rx
    p	  = bateau.vit.rx
    q	  = bateau.vit.ry
    B = np.zeros(6)
    B[0] = F.tx -m*g*theta 
    B[1] = F.ty +m*g*phi -m*uzeros*p 
    B[2] = F.tz +m*uzeros*q 
    B[3] = F.rx 
    B[4] = F.ry 
    B[5] = F.rz
    return B 
    
def modif(pos_angle):
    psi = pos_angle[2]
    theta = pos_angle[1]
    phi = pos_angle[0]
    # On fait l'hypothése que tous les angles sont petits
    A = np.eye(3)
    A[0][1] = -psi
    A[1][0] =  psi
    
    B = np.eye(3)
    B[0][2] =  theta
    B[2][0] = -theta
    
    C = np.eye(3)
    C[1][2] = -phi
    C[2][1] =  phi
    abc = A*B*C
    return abc


