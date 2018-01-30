import os
folder = "."
os.chdir(folder)
import numpy as np
from constantes import *
from scipy import *

# Modifier avec les données de Xfoil

filename        = './Coefhydro/NACA0015.txt'
coefshydro      = np.genfromtxt(filename, dtype = float, skip_header = 3)


AoAhydro_index, CLhydro_index, CDhydro_index, CMhydro_index  = range(4)
coefshydro[:, AoAhydro_index] = coefshydro[:, AoAhydro_index]*np.pi/180 # On convertit les angles en radians

# Coefficient de portance du safran
def CLhydro(dirbateau):
    return interp(dirbateau, coefshydro[:, AoAhydro_index], coefshydro[:, CLhydro_index])
    
# Coefficient de trainée du safran
def CDhydro(dirbateau):
    return interp(dirbateau, coefshydro[:, AoAhydro_index], coefshydro[:, CDhydro_index])
    
# Coefficient de moment du safran
def CMhydro(dirbateau):
    if dirbateau<0:
        return -interp(-dirbateau, coefshydro[:, AoAhydro_index], coefshydro[:, CMhydro_index])
    else:
        return interp(dirbateau, coefshydro[:, AoAhydro_index], coefshydro[:, CMhydro_index])


def dirdep(bateau):
    Vx = bateau.vit.tx
    Vy = bateau.vit.ty
    omega = np.arctan2(Vy, Vx)
    return omega


def ForcesHydroDerive(bateau): # dirbateau est l'angle entre la direction du bateau et la vitesse de celui-ci
    Ubateau = bateau.vitesseabs
    dirbateau = bateau.pos.rz
    Portance = 0.5*rhoeau*bateau.hder*bateau.lder*Ubateau**2*0.001
    Trainee  = 0.5*rhoeau*bateau.hder*bateau.lder*Ubateau**2*0.001
    return (Portance, Trainee)
    
    
def ForcesreelleshydroDerive(bateau): # Il faudra peut être modifier en fonction du roulis
    Ubateau = bateau.vitesseabs
    omega = bateau.omega
    dirbateau = bateau.pos.rz
    [Portance, Trainee] = ForcesHydroDerive(bateau)
    Fx =  Portance*np.sin(dirbateau) -Trainee*np.cos(dirbateau)
    Fy = -Portance*np.cos(dirbateau) -Trainee*np.sin(dirbateau)
    Fz = 0 #A modifier peut être plus tard
    Mz = 0.5*rhoeau*bateau.hder*bateau.lder*Ubateau**2*CMhydro((omega -dirbateau)*np.pi/180)
    return (Fx, Fy, Fz, Mz)

  
def Frottements(rho, bateau):
    Vx = bateau.vit.tx
    Vy = bateau.vit.ty
    Ubateau = bateau.vitesseabs
    Re = reynolds(Ubateau, rho, bateau.L)
    Cf = Cfbateau(Re);
    frotx = -Cf*Vx
    froty = -Cf*Vy
    return (frotx, froty)
    
    
    
"""
#A réutiliser plus tard
def EquilibreKite(uvent, dirvent, Ubateau, omega, Cx, Cz, delta):

   # Cette fonction renvoie la position d'équilibre du kite en fonction de la direction et de la vitesse du vent

    (Fx, Fy, Fz, Uappdir) = Forcesreellesaero(uvent, dirvent, Ubateau, omega, Cx, Cz, delta)

    err = np.abs(Fz -mkite*g)
    while err>0.1: # Précision recherchée
        # On fait l'hypothèse d'un câble sans masse        
        Z = lcable*np.sin(np.arctan2(Fz, np.hypot(Fx, Fy)))
        # On fait aussi l'hypothèse d'un profil de vent logarithmique
        ucalc = constantes.uvent*np.log(Z/Z0)/np.log(Zref/Z0)

        (Fx, Fy, Fz, Uappdir) = Forcesreellesaero(ucalc, dirvent, Ubateau, omega, Cx, Cz, delta)
        err = np.abs(Fz -mkite*g)
        print(Z)

    X = lcable*np.sin(np.arctan2(Fx, np.hypot(Fz, Fy)))
    Y = lcable*np.sin(np.arctan2(Fy, np.hypot(Fx, Fz)))
    Z = lcable*np.sin(np.arctan2(Fz, np.hypot(Fx, Fy)))
    return (Fx, Fy, Fz, X, Y, Z, Uappdir)
"""


Alpha = np.linspace(-15*np.pi/180, 15*np.pi/180, 1000)

plt.figure()
plt.plot(Alpha, CLhydro(Alpha))
plt.plot(Alpha, CDhydro(Alpha))
#plt.plot(Alpha, CMhydro(Alpha))
