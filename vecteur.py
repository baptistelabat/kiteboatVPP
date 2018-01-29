class vecteur:
    def __init__(self):
        self.tx=0
        self.ty=0
        self.tz=0
        self.rx=0
        self.ry=0
        self.rz=0
        
def sommevecteur(*args):
    v3=vecteur()
    for arg in args:
        v3.tx+=arg.tx
        v3.ty+=arg.ty
        v3.tz+=arg.tz
        v3.rx+=arg.rx
        v3.ry+=arg.ry
        v3.rz+=arg.rz
    return v3
    
def egaltableau(tab):
    v=vecteur()
    v.tx=tab[0]
    v.ty=tab[1]
    v.tz=tab[2]
    v.rx=tab[3]
    v.ry=tab[4]
    v.rz=tab[5]
    return v
    
def produitcste(v1,cste):
    v=vecteur()
    v.tx=v1.tx*cste
    v.ty=v1.ty*cste
    v.tz=v1.tz*cste
    v.rx=v1.rx*cste
    v.ry=v1.ry*cste
    v.rz=v1.rz*cste
    return v
    
def recupvaleur(liste,valeur):
    v=[]
    for j in range(len(liste)):
        v.append(getattr(liste[j],valeur))
    return v

def trans(v):   #Renvoie une liste des coordonnées associées en translation
    v1=[]
    v1.append(v.tx)
    v1.append(v.ty)
    v1.append(v.tz)
    return v1
    
def rot(v):     #Renvoie une liste des coordonnées associées en rotation
    v1=[]
    v1.append(v.rx)
    v1.append(v.ry)
    v1.append(v.rz)
    return v1
    