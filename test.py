#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A=vitesse()
listevitesse = [A]
listevitesse.append(listevitesse[0].u +A.u)
"""
def Test1(*args):
    #print (Test1)
    i = 0
    #print(kwargs)
    print(args)
    for arg in args:
        i += arg    
    #for k, v in kwargs.items():
        #print( k+v)
        #i+=k
    print(i)
    

    
    
    
