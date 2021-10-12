#######################################
#                                     #
#  Scott Woody                        #
#  ECE 351-52                         #
#  Lab 6                              #
#  07October2021                      #
#                                     # 
#                                     # 
#######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#Part 1 Function declaration
steps = 1e-3
t = np.arange(0, 2 + steps, steps)

def funcstep(t):  
    x = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i]< 0 :
            x[i] = 0
        else :
            x[i] = 1   
    return x

def f1(t):
    x = ( .5 - .5 * np.exp(-4 * t) + np.exp(-6 * t)) * funcstep(t)
    return x

num = [1, 6, 12] # Creates a matrix for the numerator
den = [1, 10, 24] # Creates a matrix for the denominator
to, yo = sig.step((num, den), T=t)

#Part1 Task 2
plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , f1(t))
plt.grid ()
plt.ylabel ('hand solved')

plt.subplot (2 , 1 , 2)
plt.plot (to , yo)
plt.grid ()
plt.ylabel ('scipy solved')
plt.xlabel ('t')
plt.show ()

#Part1 Task 3
den = [1, 10, 24, 0]
print(sig.residue(num, den))

#Part2 Task1

num2 = [0, 0, 0, 0, 0, 25250]
den2 = [1, 18, 218, 2036, 9085, 25250, 0]
print(sig.residue(num2, den2))

#Part2 Task2
resn1 = [-0.48557692+0.72836538j,
       -0.48557692-0.72836538j, 0.09288674-0.04765193j,  0.09288674+0.04765193j]
resd1 = [  -3. +4.j, -3. -4.j, -1.+10.j,  -1.-10.j]

def cos(d, n):
    yt = 0
    for i in range(len(d)):
        k = np.abs(n[i])
        ka = np.angle(n[i])
        a = np.real(d[i])
        o = np.imag(d[i])
        yt += k*np.exp(a*t)*np.cos(o*t + ka)
    return yt


t = np.arange(0, 4.5 + steps, steps)

y= funcstep(t)*(cos(resd1, resn1) + 1 + -.21461963*np.exp(-10*t))

#Part2 Task 3
den22 = [1, 18, 218, 2036, 9085, 25250]
to, yo = sig.step((num2, den22), T=t)

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t, y)
plt.grid ()
plt.ylabel ('hand solved')

plt.subplot (2 , 1 , 2)
plt.plot (to , yo)
plt.grid ()
plt.ylabel ('scipy solved')
plt.xlabel ('t')
plt.show ()


    
    
    



