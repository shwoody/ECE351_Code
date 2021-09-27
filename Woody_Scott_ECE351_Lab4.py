#######################################
#                                     #
#  Scott Woody                        #
#  ECE 351-52                         #
#  Lab 4                              #
#  30SEP2021                          #
#                                     # 
#                                     # 
#######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

#Part 1 Function declaration
steps = 1e-2
t = np.arange(-10, 10 + steps, steps)

def funcstep(t):  
    x = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i]< 0 :
            x[i] = 0
        else :
            x[i] = 1   
    return x
def h1(t):
    y = np.exp(-2 * t)*(funcstep(t)-funcstep(t - 3))
    return y

def h2(t):
    x = funcstep(t - 2) - funcstep(t - 6)
    return x

def h3(t):
    w = .25 * 2 * np.pi
    y = np.cos(w * t)*funcstep(t)
    return y

#Part 1 function plotting
plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , h1(t))
plt.grid ()
plt.ylabel ('h3(t)')


plt.subplot (3 , 1 , 2)
plt.plot (t , h2(t))
plt.grid ()
plt.ylabel ('h2(t)')


plt.subplot (3 , 1 , 3)
plt.plot (t , h3(t))
plt.grid ()
plt.ylabel ('h3(t)')
plt.xlabel ('t')
plt.show ()

def conv(f1,f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1, np.zeros((1, Nf2-1)))
    f2Extended = np.append(f2, np.zeros((1, Nf1-1)))
    result = np.zeros(f1Extended.shape)
    for i in range(Nf2 + Nf1 - 2):
        result[i] = 0
        for j in range(Nf1):
            if((i - j + 1) > 0):
                try:
                    result[i] += f1Extended[j] * f2Extended[i - j + 1]
                except:
                    print(i, j)
    return result

u = funcstep(t)
h11 = h1(t)
h22 = h2(t)
h33 = h3(t)

f1 = conv(u, h11)*steps
f2 = conv(u, h22)*steps
f3 = conv(u, h33)*steps

t = np.arange(-10, 10 + steps, steps)
NN = len(t)
tExtended = np.arange(2 * t[0] , 2 * t[NN-1] + steps, steps)


plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (tExtended , f1)
plt.grid ()
plt.ylabel ('h1(t)*u(t)')


plt.subplot (3 , 1 , 2)
plt.plot (tExtended , f2)
plt.grid ()
plt.ylabel ('h2(t)*u(t)')


plt.subplot (3 , 1 , 3)
plt.plot (tExtended , f3)
plt.grid ()
plt.ylabel ('h3(t)*u(t)')
plt.xlabel ('t')
plt.show ()

t = np.arange(-20, 20 + steps, steps)

def f1c(t):
    x = .5*(1 - np.exp(-2*t))*funcstep(t) - .5*(1 - np.exp(-2*(t-3)))*funcstep(t-3)
    return x
def f2c(t):
    x = (t-2)*funcstep(t-2) - (t-6) * funcstep(t-6)
    return x
def f3c(t):
    w = .25 * 2 * np.pi
    x = (1/w)*np.sin(w * t)*funcstep(t)
    return x

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , f1c(t))
plt.grid ()
plt.ylabel ('h1(t)*u(t)')


plt.subplot (3 , 1 , 2)
plt.plot (t , f2c(t))
plt.grid ()
plt.ylabel ('h2(t)*u(t)')


plt.subplot (3 , 1 , 3)
plt.plot (t , f3c(t))
plt.grid ()
plt.ylabel ('h3(t)*u(t)')
plt.xlabel ('t')
plt.show ()


