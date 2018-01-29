
import os
folder = "C:/Users/Jean/Desktop/Océan/Projet/Projet bateau"
os.chdir(folder)
import numpy as np
from constantes import *

def dirdep(Vx,Vy):
    omega=np.arctan2(Vy,Vx)
    return omega

def Ventapparent(uvent, dirvent, Ubateau, omega):
    """
    Uvent: vitesse réelle du vent
    dirvent: direction du vent dans le repére terrestre (0 deg vent arrière, 180 vent de face)
    Ubateau: vitesse du bateau
    psi: rotation du bateau autour de z
    """
    Uapp=np.hypot( uvent*np.sin(dirvent)-Ubateau*np.sin(omega), uvent*np.cos(dirvent)-Ubateau*np.cos(omega))
    #print(Uapp)
    Uappdir=np.arctan2(uvent*np.sin(dirvent)-Ubateau*np.sin(omega), uvent*np.cos(dirvent)-Ubateau*np.cos(omega))
    #print(Uappdir)
    return (Uapp, Uappdir)
    
def ForcesAero(Uapp, Cx, Cz):
    Portance = 0.5*rho*h*l*Uapp**2*Cz
    Trainee = 0.5*rho*h*l*Uapp**2*Cx
    return (Portance, Trainee)
    
def Forcesreelles(uvent, dirvent, Ubateau, psi, Cx, Cz,delta):
    [Uapp, Uappdir]=Ventapparent(uvent, dirvent, Ubateau, psi)
    [Portance, Trainee]=ForcesAero(Uapp, Cx, Cz)

    Fx=Trainee*np.cos(delta+Uappdir)
    Fy=Trainee*np.sin(delta+Uappdir)
    Fz=Portance
    return (Fx,Fy,Fz,Uappdir)
    
def Frottements(Ubateau, rho,L,Vx, Vy):
    Re=reynolds(Ubateau,rho,L)
    Cf=Cfbateau(Re);
    frotx=-Cf*Vx
    froty=-Cf*Vy
    return (frotx, froty)
"""
N=1000
Ubateau=np.zeros(N)
frotx=np.zeros(N)
froty=np.zeros(N)
F_x=np.zeros(N)
F_y=np.zeros(N)
F_z=np.zeros(N)
Uapp_dir=np.zeros(N)
"""