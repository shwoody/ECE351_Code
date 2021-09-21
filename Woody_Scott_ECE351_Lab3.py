#######################################
#                                     #
#  Scott Woody                        #
#  ECE 351-52                         #
#  Lab 3                              #
#  23SEP2021                          #
#                                     # 
#                                     # 
#######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

#Part 1
steps = 1e-2
t = np.arange(0, 20 + steps, steps)
# ramp and step from lab 2
def funcstep(t):  
    x = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i]< 0 :
            x[i] = 0
        else :
            x[i] = 1   
    return x
ystep = funcstep(-t)
def funcramp(t):
    z = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            z[i] = 0
        else :
            z[i] = t[i]    
    return z
yramp = funcramp(t)
# new functions
def f1(t):
    y1 = funcstep(t-2) - funcstep(t-9)
    return y1

def f2(t):
    y2 = funcstep(t) * np.exp(-t)
    return y2

def f3(t):
    y3 = funcramp(t-2) * (funcstep(t-2) - funcstep(t-3)) + funcramp(-t + 4) * (funcstep(t-3) - funcstep(t-4))
    return y3

y1 = f1(t)

#plotting all the functions
plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , y1 )
plt.grid ()
plt.ylabel ('y1(t)')
plt.xlabel ('t')
plt.title ('Part 1, Task 2, f1')

y2 = f2(t)

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t, y2 )
plt.grid ()
plt.ylabel ('y2(t)')
plt.xlabel ('t')
plt.title ('Part 1, task 2, f2')

y3 = f3(t)

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t, y3 )
plt.grid ()
plt.ylabel ('y3(t)')
plt.xlabel ('t')
plt.title ('Part 1, task 2, f3')
plt.show ()

#User defined Convolution

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

# Part 2 task 2
t = np.arange(0, 20 + steps, steps)
NN = len(t)
tExtended = np.arange(0, 2 * t[NN-1], steps)

f12 = conv(y1, y2) * steps

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (tExtended, f12 )
plt.grid ()
plt.ylabel ('f12(t)')
plt.xlabel ('t')
plt.title ('Part 2, task 2')
plt.show ()

f23 = conv(f2(t), f3(t)) * steps

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (tExtended, f23 )
plt.grid ()
plt.ylabel ('f23(t)')
plt.xlabel ('t')
plt.title ('Part 2, task 3')
plt.show ()

f13 = conv(f1(t), f3(t)) * steps

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (tExtended, f13 )
plt.grid ()
plt.ylabel ('f13(t)')
plt.xlabel ('t')
plt.title ('Part 2, task 4')
plt.show ()

y12 = scipy.signal.convolve(y2, y3) # done for task 2, 3, and 4

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (tExtended, y12 )
plt.grid ()
plt.ylabel ('f13(t)')
plt.xlabel ('t')
plt.title ('Part 2, task 4')
plt.show ()