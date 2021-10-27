#######################################
#                                     #
#  Scott Woody                        #
#  ECE 351-52                         #
#  Lab 8                              #
#  21OCT2021                          #
#                                     # 
#                                     # 
#######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

T = 8
#Part1 Task1
w0 = 2*np.pi/T
a0 = 1*T/2 - 1*T/2
print("a0 = ", a0)
a1 = 2/T * 1/(1*w0)*(np.sin(T*1*w0/2)-np.sin(0)+np.sin(0) - np.sin(-T*1*w0/2))
print("a1 =", a1)
k = 1
b1 = 2/(k*np.pi)*(1-np.cos(k*np.pi))
print("b1 = ", b1)
k = 2
b2 = 2/(k*np.pi)*(1-np.cos(k*np.pi))
k = 3
b3 = 2/(k*np.pi)*(1-np.cos(k*np.pi))
print("b2 = ", b2)
print("b3 = ", b3)

#Part1 Task2
steps = 1e-3
t = np.arange(0, 20 + steps, steps)

def x(k):
    b = 0
    y = 0
    for x in np.arange(1, k+1):
        b = 2/((x)*np.pi)*(1-np.cos((x)*np.pi))
        xt = b*np.sin(x*w0*t)
        y += xt
    return y
    

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x(1))
plt.grid ()
plt.ylabel ('k = 1')

plt.subplot (3 , 1 , 2)
plt.plot (t , x(3))
plt.grid ()
plt.ylabel ('k = 3')

plt.subplot (3 , 1 , 3)
plt.plot (t , x(15))
plt.grid ()
plt.ylabel ('k = 15')
plt.xlabel ('t')
plt.show ()

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x(50))
plt.grid ()
plt.ylabel ('k = 50')

plt.subplot (3 , 1 , 2)
plt.plot (t , x(150))
plt.grid ()
plt.ylabel ('k = 150')

plt.subplot (3 , 1 , 3)
plt.plot (t , x(1500))
plt.grid ()
plt.ylabel ('k = 1500')
plt.xlabel ('t')
plt.show ()
