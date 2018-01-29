import os
folder = "C:/Users/Jean/Desktop/Océan/Projet/Projet bateau"
os.chdir(folder)
import numpy as np
from constantes import *
from scipy import *
import matplotlib.pyplot as plt

filename        = '../Projet bateau/Coefaero/DU21_A17.dat'
coefsaero       =np.genfromtxt(filename, dtype=float, skip_header=14)


AoAaero_index, CLaero_index, CDaero_index  = range(3)
coefsaero[:,AoAaero_index]=coefsaero[:,AoAaero_index]*np.pi/180

#Coefficient de portance de la voile
def CLaero(alpha):
    return interp(alpha, coefsaero[:, AoAaero_index], coefsaero[:, CLaero_index])
    
#Coefficient de trainée de la voile
def CDaero(alpha):
    return interp(alpha, coefsaero[:, AoAaero_index], coefsaero[:, CDaero_index])

def CDoverCL(alpha):
    return CDaero(alpha)/CLaero(alpha)
    
def CLoverCD(alpha):
    return CLaero(alpha)/CDaero(alpha)
    
def Ventapparent(vent,bateau):
    """
    Uvent: vitesse réelle du vent
    dirvent: direction du vent dans le repére terrestre (0 deg vent arrière, 180 vent de face)
    Ubateau: vitesse du bateau
    omega: angle de rotation du bateau autour de z calculé avec les vitesses
    """
    Ubateau=bateau.vitesseabs
    omega=bateau.omega
    Uapp=np.hypot(vent.vitesse*np.sin(vent.direction)-Ubateau*np.sin(omega), vent.vitesse*np.cos(vent.direction)-Ubateau*np.cos(omega))
    Uappdir=np.arctan2(vent.vitesse*np.sin(vent.direction)-Ubateau*np.sin(omega), vent.vitesse*np.cos(vent.direction)-Ubateau*np.cos(omega))
    return (Uapp, Uappdir)

    
def ForcesAero(alpha,vent):
    Portance = 0.5*vent.rho*h*l*vent.vitesseapparente**2*CLaero(alpha)
    Trainee = 0.5*vent.rho*h*l*vent.vitesseapparente**2*CDaero(alpha)
    return (Portance, Trainee)
    
def AlphaOptim(kite, vent):
    angle=linspace(1,pi,179) 
    T=kite.Tmax
    aopt=angle[0]
    (Portance, Trainee)= ForcesAero(aopt,vent)
    i=1
    while (np.sqrt(Portance**2+Trainee**2)<T and Portance>0 and CLaero(angle[i]) !=0):
        if CDoverCL(angle[i])>aopt:
            aopt=angle[i]
            (Portance, Trainee)= ForcesAero(angle[i],vent)
        i+=1
    elevation=np.arctan2(Portance, Trainee)
    return (aopt, elevation)

def Forcesreellesaero(vent,bateau, alpha,delta):
    Ubateau=bateau.vitesseabs
    [Uapp, Uappdir]=Ventapparent(vent, bateau)
    [Portance, Trainee]=ForcesAero(alpha,vent)
    Fx=Trainee*np.cos(Uappdir)
    Fy=Trainee*np.sin(Uappdir)
    Fz=Portance
    return (Fx,Fy,Fz)
"""
Alpha=np.linspace(-180,180,1000)

plt.figure()
plt.plot(Alpha, CLaero(Alpha))
plt.plot(Alpha, CDaero(Alpha))

uvent=30
dirvent=np.pi/6
Ubateau=10
omega=np.pi/4-0.0001
delta=0

plt.figure()
plt.plot(Alpha, Forcesreellesaero(uvent, dirvent, Ubateau, omega, Alpha,delta)[0],label="Fx")
plt.plot(Alpha, Forcesreellesaero(uvent, dirvent, Ubateau, omega, Alpha,delta)[1],label="Fy")
plt.plot(Alpha, Forcesreellesaero(uvent, dirvent, Ubateau, omega, Alpha,delta)[2],label="Fz")
plt.legend()

plt.figure()
plt.plot(Alpha, Forcesreellesaero(uvent, dirvent, Ubateau, omega, Alpha,delta)[3],label="Direction du vent apparent")
plt.legend()
"""
Alpha=np.linspace(-np.pi,np.pi,1000)

plt.figure()
plt.plot(Alpha, CLaero(Alpha))
plt.plot(Alpha, CDaero(Alpha))
