# Projet de traction d'un navire par kite
# Réalisé par Abel Pruchon, Jiakan Zhou et Jean Cresp

import os
import subprocess
folder = "."
os.chdir(folder)

# os.startfile('vecteur.py')
# os.startfile('vent.py')
# os.startfile('bateau.py')
# os.startfile('fonction_forces_aero.py')
# os.startfile('fonction_forces_hydro.py')
# os.startfile('fonctions.py')

#subprocess.call('test.py')



import numpy as np
import matplotlib.pyplot as plt
from scipy import *
import constantes
from mpl_toolkits.mplot3d import Axes3D
from pylab import *


tps = 20 
N = 1000 # Nombre d'itération
dt = tps/N # Pas de temps 
temps = np.linspace(1, tps, N)
Alpha = np.linspace(-180, 180, 50)
P = len(Alpha)
P = 1

        # On va initialiser les inconnues

# On définit les vecteurs position,vitesse et accélération du bateau
Bateau = classbateau()
listeBat = []
Bateau.vit.tx = uzeros
Bateau.vitesseabs = Bateau.uzeros
Bateau.omega = dirdep(Bateau)
A = Bateau.A

# On définit maintenant les caractéristiques du vent

vent = classvent()
listevent = []
(Uapp, Uappdir) = Ventapparent(vent, Bateau)
vent.directionapparente = Uappdir
vent.vitesseapparente = Uapp

# On initialise aussi le kite

kite = classKite()
listekite = []

# On initilise les différentes forces en jeu
Ffrot = vecteur()
listeffrot = []

Fkite = vecteur()
listefkite = []

Fderive = vecteur()
listefderive = []

Ftot = vecteur()
listeftot = []


j = 0

# alpha = 0
delta = 0
# On donne l'expression des forces

"""Forces de frottement sur la carène"""
Re = reynolds(Bateau.vit.tx, rhoeau, L)
Cf = Cfbateau(Re)
frot = -Cf*Bateau.vit.tx
Ffrot.tx = frot

"""Forces dues au kite"""
(aopt, elevation) = AlphaOptim(kite, vent)
kite.alpha = aopt
kite.elevation = elevation
(Fx,Fy,Fz) = Forcesreellesaero(vent, Bateau, aopt, delta)
Fkite = egaltableau([Fx, Fy, Fz, 0, 0, 0])

"""Forces dues à la dérive"""
(Fxhydro, Fyhydro, Fzhydro, Mzhydro) = ForcesreelleshydroDerive(Bateau)
Fderive = egaltableau([Fxhydro, Fyhydro, Fzhydro, 0, 0, Mzhydro])


Ftot = sommevecteur(Ffrot, Fkite, Fderive)

B = zeros((6, N, P))
B[:, 0, j] = b(g, Ftot, Bateau)
Bateau.acc = egaltableau(np.linalg.solve(A, B[:, 0, j]))

listevent  	.append(vent)
listekite	.append(kite)
listeffrot	.append(Ffrot)
listefkite	.append(Fkite)
listefderive	.append(Fderive)
listeftot	.append(Ftot)
listeBat	.append(Bateau)

for i in range(1,N):
    Bateau = classbateau()
    
    # On calcule les nouvelles vitesses
    
    Bateau.vit = sommevecteur(listeBat[i-1].vit, produitcste(listeBat[i-1].acc, dt))

    # On calcule les nouvelles coordonnées
    
    pos_angle = rot(sommevecteur(listeBat[i-1].pos, produitcste(listeBat[i-1].vit, dt)))
    abc = modif(pos_angle)
    pos = trans(listeBat[i-1].pos) +dot(abc, trans(listeBat[i-1].vit))*dt
    position = egaltableau(np.concatenate((pos, pos_angle)))
    Bateau.pos = position

    # On exprime les nouvelles forces
    
    Bateau.vitesseabs = np.hypot(Bateau.vit.tx, Bateau.vit.ty)
    Bateau.omega = dirdep(Bateau)
       
        # La trainée
    Ffrot = vecteur()
    (frotx, froty) = Frottements(rhoeau, Bateau)
    Ffrot.tx = frotx
    Ffrot.ty = froty
    
        # Les forces dues au kite    
    vent = classvent()
    kite = classKite()
    Fkite = vecteur()
    (Uapp, Uappdir) = Ventapparent(vent, Bateau)
    vent.directionapparente = Uappdir
    vent.vitesseapparente = Uapp
    (aopt, elevation) = AlphaOptim(kite, vent)
    kite.alpha = aopt
    kite.elevation = elevation
    (Fxaero, Fyaero, Fzaero) = Forcesreellesaero(vent, Bateau, aopt, delta)
    Fkite = egaltableau([Fxaero, Fyaero, Fzaero, 0 , 0, 0])
    
        # Les forces dues à la dérive        
    (Fxhydro, Fyhydro, Fzhydro, Mzhydro) = ForcesreelleshydroDerive(Bateau)
    Fderive = egaltableau([Fxhydro, Fyhydro, Fzhydro, 0, 0, Mzhydro])
    """
    pos_kite[0, i, j] = Xkite +pos[0,i,j]
    pos_kite[1, i, j] = Ykite +pos[1,i,j]
    pos_kite[2, i, j] = Zkite +pos[2,i,j]

    #F_xderive[i, j] = Fxhydro
    #F_yderive[i, j] = Fyhydro
    #F_zderive[i, j] = Fzhydro
    # Uapp_dir[i, j] = Uappdir
    """
    
        # Tous les efforts

    Ftot = sommevecteur(Ffrot, Fkite, Fderive)
    
    B[:, i, j] = b(g, Ftot, Bateau)
    Bateau.acc = egaltableau(np.linalg.solve(A, B[:, i, j]))
    listevent	.append(vent)
    listekite	.append(kite)
    listeffrot	.append(Ffrot)
    listefkite	.append(Fkite)
    listefderive.append(Fderive)
    listeftot	.append(Ftot)
    listeBat	.append(Bateau)


plt.figure(1)
plt.subplot(211)
plt.title("Vitesse")
plt.plot(temps, recupvaleur(recupvaleur(listeBat, 'vit'), 'tx'), label = "Vx")
plt.plot(temps, recupvaleur(recupvaleur(listeBat, 'vit'), 'ty'), label = "Vy")
plt.legend()
plt.xlabel("t(s)")
plt.subplot(212)
plt.title("Trajectoire")
plt.plot(recupvaleur(recupvaleur(listeBat, 'pos'), 'tx'), recupvaleur(recupvaleur(listeBat, 'pos'), 'ty'))
plt.show()

plt.figure(2)
plt.title('Force')
plt.plot(temps, recupvaleur(listefkite, 'tx'), label = "F_xkite") # Passer en echelle log pour voir le début, mais il faut une meilleure précision
plt.plot(temps, recupvaleur(listefkite, 'ty'),label = "F_ykite")
plt.plot(temps, recupvaleur(listefkite, 'tz'),label = "F_zkite")
plt.legend()
plt.show()

plt.figure(3)
plt.plot(temps, recupvaleur(listevent, 'directionapparente'), label = "Angle du vent apparent")
plot(temps, recupvaleur(listeBat, 'omega'), label = "Angle de la trajectoire du bateau")
plot(temps, recupvaleur(recupvaleur(listeBat, 'pos'),'rz'), label = "Angle de la dérive")
plt.legend()
plt.show()

plt.figure(4)
plt.plot(temps, recupvaleur(listeBat, 'vitesseabs'))
plt.show()
"""
plt.figure()
plt.title("Portance")
plt.plot(temps, F_zkite -mkite*g)
plt.show()



plt.figure()
plt.title("Trajectoire du kite")
plt.plot(np.hypot(pos_kite[0, :, 0], pos_kite[1, :, 0]), pos_kite[2, :, 0])
plt.show()

ax = Axes3D(plt.figure())
ax.plot(pos_kite[0, :, 0], pos_kite[1, :, 0], pos_kite[2, :, 0])
plt.show()
"""

"""
Quand on fait varier la direction du vent
"""
"""

plt.figure()
plt.plot(Dirvent, Puissance[N-1, :])
plt.show()
plt.figure()
plt.plot(Dirvent, Ubateau[N-1, :])
plt.show()

plt.figure()
plt.plot(Dirvent, F_xkite[N-1, :])
plt.plot(Dirvent, F_ykite[N-1, :])
plt.show()

distance = np.sqrt(pos[0, N-1, :]**2+pos[1, N-1, :]**2)
posx = zeros(P)
posy = zeros(P)
for k in range(P):
    posx[k] = distance[k]*np.cos(Dirvent[k])
    posy[k] = distance[k]*np.sin(Dirvent[k])
    
plt.figure()
plt.title("Position finale en fonction de la direction du vent")
plt.grid()
plt.plot(posx, posy, "bs")
plt.xlabel("x(m)")
plt.ylabel("y(m)")
plt.show()

plt.figure()
plt.plot(Dirvent, distance, "bs")
plt.xlabel("Direction du vent (rad)")
plt.ylabel("Distance parcourue(m)")
plt.show()
"""
