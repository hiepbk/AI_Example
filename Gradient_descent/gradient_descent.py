# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt

#%%
def grad(x):
    return 2*x+5*np.cos(x)
def cost(x):
    return x**2+5*np.sin(x)
def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) <1e-3:
            break
        x.append(x_new)
    return (x, it)
#%%
(x1, it1) = myGD1(0.1,-5)
    

