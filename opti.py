# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:03:38 2018

@author: apruchon2016
"""

#Projet de traction d'un navire par kite
#Réalisé par Abel Pruchon, Jiakan Zhou et Jean Cresp

#Optimisation de la vitesse projetée

from scipy.optimize import minimize
import os
folder = "Z:/Projet_Kite/Code"
os.chdir(folder)
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import constantes
import extractionccoefs as extrac
import sys
import math
runfile('Z:/Projet_Kite/Code/constantes.py', wdir='Z:/Projet_Kite/Code')
runfile('Z:/Projet_Kite/Code/extractionccoefs.py', wdir='Z:/Projet_Kite/Code')


#Definition des differences forces
'''
Beta = angle de Ubateau
delta  = angle de barre
psi = angle de cap
dirvent = angle de vent réel
gamma = angle du vent apparent
'''
#Calcul du nombre de Reynolds
def reynolds(u,rho,L):
    visc=1.07*10**(-3)
    Re=np.abs(u)*rho*L/visc
    return Re

#Calcul du coefficient de frottement du bateau selon une loi empirique    
def Cfbateau(Re):
    #print(Re)
    if Re==0:
        return 0
    else:        
        Cf=0.075/(np.log10(Re)-2)**2
        return Cf

#Détermination des frottements sur la coque du navire    
def Frottements(Ubateau,rho,L,u):
    Re=reynolds(Ubateau,rho,L)
    #print(Re)
    Cf=Cfbateau(Re);
    #print(Cf)
    frot=-0.5*Cf*rho*constantes.Sbateau*u**2
    return frot    

#Calcul de la norme et de la direction du vent apparent
def Ventapparent(uvent, Ubateau, beta):
    """
    Uvent: vitesse réelle du vent
    Ubateau: vitesse du bateau
    beta: angle de la vitesse du bateau
    """
    Uapp=np.hypot(Ubateau*np.sin(beta),uvent+Ubateau*np.cos(beta))
    #print(Uapp)
    Uappdir=-np.arctan2(Ubateau*np.sin(beta),uvent+Ubateau*np.cos(beta))
    #print(Uappdir)
    return (Uapp, Uappdir)



#Somme des forces suivant l'axe x
def sumFx(D):  #D=(Ubateau, beta, delta)
    #Interpolation des coefficients
    CLder = interpolate.splev(D[1]-constantes.psi, extrac.tckCLderive, der=0)
    CDder = interpolate.splev(D[1]-constantes.psi, extrac.tckCDderive, der=0)
    CLgouv = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCLderive, der=0)
    CDgouv = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCDderive, der=0)
    
    #Definition des forces
    Fderive_x = 0.5*constantes.rhoeau*constantes.Sderive*D[0]**2*np.sqrt(CLder**2+CDder**2)*(np.sin(constantes.psi))
    Fgouv_x = 0.5*constantes.rhoeau*constantes.Sgouv*D[0]**2*np.sqrt(CLgouv**2+CDgouv**2)*(np.sin(constantes.psi+D[2]))
    [Uapp, diruapp] = Ventapparent(constantes.vitessevent,D[0],D[1])
    Fkite_x = 0.5*constantes.rho*constantes.Skite*Uapp**2*np.cos(diruapp)
    Ffrott_x = Frottements(D[0], constantes.rhoeau, constantes.L, D[0])*np.cos(D[1])**2
    #print(Fkite_x)
    #print(Ffrott_x)
    #print(D[0])
    #if math.isnan(D[0]):
    #    sys.exit()
    return Fderive_x + Fgouv_x + Fkite_x + Ffrott_x


#Somme des forces suivant l'axe y
def sumFy(D):  #D=(V, beta, delta)
    #Interpolation des coefficients
    CLder = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCLderive, der=0)
    CDder = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCDderive, der=0)
    #Definition des forces
    Fderive_y = 0.5*constantes.rhoeau*constantes.Sderive*D[0]**2*np.sqrt(CLder**2+CDder**2)*(-np.cos(constantes.psi))
    Fgouv_y = 0.5*constantes.rhoeau*constantes.Sgouv*D[0]**2*np.sqrt(CLder**2+CDder**2)*(-np.cos(constantes.psi+D[2]))
    [Uapp, diruapp] = Ventapparent(constantes.vitessevent,D[0],D[1])
    Fkite_y = 0.5*constantes.rho*constantes.Skite*Uapp**2*(np.sin(diruapp))
    Ffrott_y = Frottements(D[0], constantes.rhoeau, constantes.L, D[0])*np.sin(D[1])**2
    #print(Fkite_y)
    return Fderive_y + Fgouv_y + Ffrott_y + Fkite_y




def sumdFx(D):#D=(Ubateau, beta, delta)
    [Uapp, diruapp] = Ventapparent(constantes.vitessevent,D[0],D[1])
    #Interpolation des coefficients
    CLder = interpolate.splev(D[1]-constantes.psi, extrac.tckCLderive, der=0)
    CDder = interpolate.splev(D[1]-constantes.psi, extrac.tckCDderive, der=0)
    CLgouv = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCLderive, der=0)
    CDgouv = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCDderive, der=0)
    dCLder = interpolate.splev(D[1]-constantes.psi, extrac.tckCLderive, der=1)
    dCDder = interpolate.splev(D[1]-constantes.psi, extrac.tckCDderive, der=1)
    dCLgouv = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCLderive, der=1)
    dCDgouv = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCDderive, der=1)
    
    #Definition des dérivées des forces
    d0Fderive_x = constantes.rhoeau*constantes.Sderive*D[0]*np.sqrt(CLder**2+CDder**2)*(np.sin(constantes.psi))
    d0Fgouv_x = constantes.rhoeau*constantes.Sgouv*D[0]*np.sqrt(CLgouv**2+CDgouv**2)*(np.sin(constantes.psi+D[2]))
    """
    d0Uapp = (-np.cos(D[1])*(constantes.vitessevent*np.cos(constantes.dirvent)-D[0]*np.cos(D[1]))
             -(np.sin(D[1])*(constantes.vitessevent*np.sin(constantes.dirvent)-D[0]*np.sin(D[1]))))/Uapp
    d0diruapp = d0Uapp*(1/(1+Uapp**2))
    """
    d0Uapp = (constantes.vitessevent*np.cos(D[1])+D[0])/Uapp
    d0diruapp=-(constantes.vitessevent*np.sin(D[1]))/(Uapp**2)
    d0Fkite_x = (d0Uapp*constantes.rho*constantes.Skite*Uapp*np.cos(diruapp)
                -0.5*d0diruapp*constantes.rho*constantes.Skite*Uapp**2*np.sin(diruapp)) 
    d0Ffrott_x = (0.5*constantes.rho*constantes.Sbateau)*((-0.075*np.cos(D[1])**2*D[0]*(2*np.log10(reynolds(D[0],constantes.rho,constantes.L))**2-8*np.log10(reynolds(D[0],constantes.rho,constantes.L))
                 + 8 - np.log10(reynolds(D[0],constantes.rho,constantes.L))/np.log(10) + 4/np.log(10))/(np.log10(reynolds(D[0],constantes.rho,constantes.L))-2)**4))

    d1Fderive_x = 0.5*constantes.rhoeau*constantes.Sderive*D[0]**2*((dCLder*CLder + dCDder*CDder)/np.sqrt(CLder**2+CDder**2))*(np.sin(constantes.psi))
    d1Fgouv_x = 0.5*constantes.rhoeau*constantes.Sgouv*D[0]**2*((dCLgouv*CLgouv + dCDgouv*CDgouv)/np.sqrt(CLgouv**2+CDgouv**2))*(np.sin(constantes.psi+D[2]))
    
    d1Uapp = -D[0]*constantes.vitessevent*np.sin(D[1])/Uapp
    d1diruapp = -(D[0]*constantes.vitessevent*np.cos(D[1])+D[0]**2)/Uapp**2   
                 
    d1Fkite_x = (d1Uapp*constantes.rho*constantes.Skite*Uapp*np.cos(diruapp)
                -0.5*d1diruapp*constantes.rho*constantes.Skite*Uapp**2*np.sin(diruapp)) 
    
    d1Ffrott_x = Frottements(D[0], constantes.rhoeau, constantes.L, D[0])*(-2*np.cos(D[1]*np.sin(D[1])))

  
    d2Fderive_x = 0
    d2Fgouv_x = (0.5*constantes.rhoeau*constantes.Sgouv*D[0]**2*(-(dCLgouv*CLgouv + dCDgouv*CDgouv)/np.sqrt(CLgouv**2+CDgouv**2))*(np.sin(constantes.psi+D[2]))
                + 0.5*constantes.rhoeau*constantes.Sgouv*D[0]**2*np.sqrt(CLgouv**2+CDgouv**2)*(np.cos(constantes.psi+D[2])))
    d2Fkite_x = 0
    d2Ffrott_x = 0
    

    return np.array([d0Fderive_x + d0Fgouv_x + d0Fkite_x + d0Ffrott_x,
                     d1Fderive_x + d1Fgouv_x + d1Fkite_x + d1Ffrott_x,
                     d2Fderive_x + d2Fgouv_x + d2Fkite_x + d2Ffrott_x])

def sumdFy(D):#D=(V, beta, delta)   
    [Uapp, diruapp] = Ventapparent(constantes.vitessevent,D[0],D[1])
    #Interpolation des coefficients
    CLder = interpolate.splev(D[1]-constantes.psi, extrac.tckCLderive, der=0)
    CDder = interpolate.splev(D[1]-constantes.psi, extrac.tckCDderive, der=0)
    CLgouv = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCLderive, der=0)
    CDgouv = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCDderive, der=0)
    dCLder = interpolate.splev(D[1]-constantes.psi, extrac.tckCLderive, der=1)
    dCDder = interpolate.splev(D[1]-constantes.psi, extrac.tckCDderive, der=1)
    dCLgouv = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCLderive, der=1)
    dCDgouv = interpolate.splev(D[1]-constantes.psi-D[2], extrac.tckCDderive, der=1)
    
    #Definition des dérivées des forces
    d0Fderive_y = constantes.rhoeau*constantes.Sderive*D[0]*np.sqrt(CLder**2+CDder**2)*(-np.cos(constantes.psi))
    d0Fgouv_y = constantes.rhoeau*constantes.Sgouv*D[0]*np.sqrt(CLgouv**2+CDgouv**2)*(-np.cos(constantes.psi+D[2]))

    d0Uapp = (constantes.vitessevent*np.cos(D[1])+D[0])/Uapp
    d0diruapp=-(constantes.vitessevent*np.sin(D[1]))/(Uapp**2)
    d0Fkite_y = (d0Uapp*constantes.rho*constantes.Skite*Uapp*np.sin(diruapp)
                +0.5*d0diruapp*constantes.rho*constantes.Skite*Uapp**2*np.cos(diruapp)) 
    #print(np.log10(reynolds(D[0],constantes.rho,constantes.L)))
    #print((reynolds(D[0],constantes.rho,constantes.L)))
    d0Ffrott_y = (0.5*constantes.rho*constantes.Sbateau)*((-0.075*np.sin(D[1])**2*D[0]*(2*np.log10(reynolds(D[0],constantes.rho,constantes.L))**2-8*np.log10(reynolds(D[0],constantes.rho,constantes.L))
                 + 8 - np.log10(reynolds(D[0],constantes.rho,constantes.L))/np.log(10) + 4/np.log(10))/(np.log10(reynolds(D[0],constantes.rho,constantes.L))-2)**4))

    d1Fderive_y = 0.5*constantes.rhoeau*constantes.Sderive*D[0]**2*((dCLder*CLder + dCDder*CDder)/np.sqrt(CLder**2+CDder**2))*(-np.cos(constantes.psi))
    d1Fgouv_y = 0.5*constantes.rhoeau*constantes.Sgouv*D[0]**2*((dCLgouv*CLgouv + dCDgouv*CDgouv)/np.sqrt(CLgouv**2+CDgouv**2))*(-np.cos(constantes.psi+D[2]))
   
    d1Uapp = -D[0]*constantes.vitessevent*np.sin(D[1])/Uapp
    d1diruapp = -(D[0]*constantes.vitessevent*np.cos(D[1])+D[0]**2)/Uapp**2        
    d1Fkite_y = (d1Uapp*constantes.rho*constantes.Skite*Uapp*np.sin(diruapp)
                +0.5*d1diruapp*constantes.rho*constantes.Skite*Uapp**2*np.cos(diruapp)) 

    d1Ffrott_y = Frottements(D[0], constantes.rhoeau, constantes.L, D[0])*(2*np.cos(D[1])*np.sin(D[1]))

  
    d2Fderive_y = 0
    d2Fgouv_y = (0.5*constantes.rhoeau*constantes.Sgouv*D[0]**2*(-(dCLgouv*CLgouv + dCDgouv*CDgouv)/np.sqrt(CLgouv**2+CDgouv**2))*(-np.cos(constantes.psi+D[2]))
                + 0.5*constantes.rhoeau*constantes.Sgouv*D[0]**2*np.sqrt(CLgouv**2+CDgouv**2)*(np.sin(constantes.psi+D[2])))
    d2Fkite_y = 0
    d2Ffrott_y = 0


    
    return np.array([d0Fderive_y + d0Fgouv_y + d0Fkite_y + d0Ffrott_y,
                     d1Fderive_y + d1Fgouv_y + d1Fkite_y + d1Ffrott_y,
                     d2Fderive_y + d2Fgouv_y + d2Fkite_y + d2Ffrott_y])
  
def Fopti(D):
    return -D[0]*np.cos(D[1]-constantes.psi)

def dFopti(D):
    dfdD0 = -np.cos(D[1]-constantes.psi)
    dfdD1 = D[0]*np.sin(D[1]-constantes.psi)
    dfdD2 = 0
    return np.array([dfdD0, dfdD1, dfdD2])






'''
AFFICHAGE 3D
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.linspace(-15*np.pi/180, 15*np.pi/180, 100)
Y = np.linspace(-15*np.pi/180, 15*np.pi/180, 100)
Z = np.zeros((len(X),len(X)))

for i in range(len(X)):
    for j in range(len(Y)):
        Z[i,j] = sumFx([10,X[i],Y[j]])
    
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot') #Affichage 3D
#ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap='hot') #tracé contour

ax.set_zlim(-100000,100000)


# savefig('../figures/plot3d_ex.png',dpi=48)
plt.show()

'''

'''

X = np.linspace(-15*np.pi/180, 15*np.pi/180, 100)
Z= np.zeros(len(X))
dZ= np.zeros(len(X))
for i in range(len(X)):
    Z[i] = sumFx([10,X[i],0.01])
    dZ[i] = sumdFx([10,X[i],0.01])[2]
plt.ion()
plt.plot(X, Z, X, dZ)    
'''


'''
        {'type':'eq',
         'fun' :lambda x: sumFy(x),
         'jac' :lambda x: sumdFy(x)},
         
        {'type':'eq',
         'fun' :lambda x: sumFx(x),
         'jac' :lambda x: sumdFx(x)}
        {'type':'ineq',
         'fun' :lambda x: -sumFx(x)+10**(-2),
         'jac' :lambda x: -sumdFx(x)},
        
        {'type':'ineq',
         'fun' :lambda x: sumFx(x)+10**(-2),
         'jac' :lambda x: sumdFx(x)},
    
        {'type':'ineq',
         'fun' :lambda x: -sumFy(x)+10**(-2),
         'jac' :lambda x: -sumdFy(x)},
         
         {'type':'ineq',
         'fun' :lambda x: sumFy(x)+10**(-2),
         'jac' :lambda x: sumdFy(x)},
'''
cons = ( {'type':'eq',
         'fun' :lambda x: sumFy(x),
         'jac' :lambda x: sumdFy(x)},
         
        {'type':'eq',
         'fun' :lambda x: sumFx(x),
         'jac' :lambda x: sumdFx(x)},
         
        {'type':'ineq',
         'fun' :lambda x: np.array([x[0]-0.1]),
         'jac' :lambda x: np.array([1.0, 0.0, 0.0])},
         
        {'type':'ineq',
         'fun' :lambda x: np.array([-x[0]+100]),
         'jac' :lambda x: np.array([-1.0, 0.0, 0.0])},

        {'type':'ineq',
         'fun' :lambda x: np.array([-x[1]+constantes.psi+14.0*np.pi/180]),
         'jac' :lambda x: np.array([0.0, -1.0, 0.0])},
        
        {'type':'ineq',
         'fun' :lambda x: np.array([x[1]-constantes.psi+14.0*np.pi/180]),
         'jac' :lambda x: np.array([0.0, 1.0, 0.0])},
  
        {'type':'ineq',
         'fun' :lambda x: np.array([x[2]+14.0*np.pi/180]),
         'jac' :lambda x: np.array([0.0, 0.0, 1.0])},
        
        {'type':'ineq',
         'fun' :lambda x: np.array([-x[2]+14.0*np.pi/180]),
         'jac' :lambda x: np.array([0.0, 0.0, -1.0])})

#res = minimize(Fopti, np.array([29,np.pi,0]), jac=dFopti,
#                constraints = cons, method='SLSQP', options={'disp':True,'maxiter': 10000})

 
N =1000
Vini = np.linspace(29,30.7,N)       
U = np.zeros(N)
beta = np.zeros(N)
delta = np.zeros(N)

for i in range(N):         
    res = minimize(Fopti, np.array([Vini[i],np.pi,0]), jac=dFopti,
                constraints = cons, method='SLSQP', options={'maxiter': 1000})
    U[i]=res.x[0]
    beta[i]=res.x[1]
    delta[i]=res.x[2]

plt.figure(1)    
plt.plot(Vini,U)    
plt.figure(2)    
plt.plot(Vini,beta) 
plt.figure(3)    
plt.plot(Vini,delta) 
