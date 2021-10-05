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
import scipy.signal as sig

#Part 1 Function declaration
steps = 1e-5
t = np.arange(0, 1.2e-3 + steps, steps)

def funcstep(t):  
    x = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i]< 0 :
            x[i] = 0
        else :
            x[i] = 1   
    return x

def f1(t):
    x = -1.0356e4 * np.exp(-5000*t)*np.sin(18584*t+105.06)*funcstep(t)
    return x

num = [0, 1e4, 0] # Creates a matrix for the numerator
den = [1, 1e4, 37037e4] # Creates a matrix for the denominator
tout , yout = sig. impulse (( num , den), T = t)


# Plot tout , yout

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , f1(t))
plt.grid ()
plt.ylabel ('hand solved')

plt.subplot (2 , 1 , 2)
plt.plot (tout , yout)
plt.grid ()
plt.ylabel ('scipy solved')
plt.xlabel ('t')
plt.show ()

to, yo = sig.step((num, den), T=t)

plt.figure ( figsize = (10 , 7) )
plt.subplot (1 , 1 , 1)
plt.plot (to , yo)
plt.grid ()
plt.ylabel ('step response')
plt.xlabel ('t')
plt.show ()
