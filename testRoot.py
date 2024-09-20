#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:11:53 2024

@author: elrohir
"""


import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def f(x):
    
    return (x-0.5*np.ones_like(x))

X,Y=np.meshgrid(np.arange(-1,1,.1),np.arange(-1,1,.1))
v=np.concatenate([X[:,:,None],Y[:,:,None]],axis=2)
F=np.apply_along_axis(lambda x: np.linalg.norm(f(x)), 2,v)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,F)

res=opt.root(f, x0=np.zeros(3))#, args=(paths, groupMethod))
print(res)

