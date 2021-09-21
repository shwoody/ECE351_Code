#######################################
#                                     #
#  Scott Woody                        #
#  ECE 351-52                         #
#  Lab 2                              #
#  16SEP2021                          #
#                                     # 
#                                     # 
#######################################

import numpy as np
import matplotlib.pyplot as plt

# Part 1 Code
steps = 1e-2
t = np.arange(0, 10 + steps, steps)

def func1(t):
    y=np.cos(t)
    return y
y = func1(t)


plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , y )
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title ('Part 1, task 2')
plt.show ()

# Part 2 Code

t = np.arange(-5, 10 + steps, steps)

def funcstep(t):  
    x = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i]< 0 :
            x[i] = 0
        else :
            x[i] = 1   
    return x
def funcramp(t):
    z = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            z[i] = 0
        else :
            z[i] = t[i]    
    return z
x = funcstep(t)


plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , x )
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title ('Part 2, task 2, step function')
plt.show ()

z = funcramp(t)

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , z )
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title ('Part 2, task 2, ramp function')
plt.show ()

def task3(t):
    u = np.zeros(t.shape)
    
    u = funcramp(t) - funcramp(t-3) + 5*funcstep(t-3) - 2*funcstep(t-6) - 2*funcramp(t-6)
    return u
u = task3(t)  

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , u )
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title ('Part 2, task 3')
plt.show ()      
    
# Part 3
# time reversal
t = np.arange(-10,5+steps,steps)
r = task3(-t)

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , r )
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title ('Part 3, task 1')
plt.show ()      

# time shift
t = np.arange(0, 15 + steps, steps)
ts = task3(t-4)

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , ts )
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title ('Part 3, task 2, t-4')
plt.show ()    

t = np.arange(-15, 0 + steps, steps)
ts = task3(-t-4)

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , ts )
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title ('Part 3, task 2, -t-4')
plt.show () 

# Time scale
t = np.arange(-5, 20 + steps, steps)
ts = task3(t/2)

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , ts )
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title ('Part 3, task 3, t/2')
plt.show () 

t = np.arange(-2, 5 + steps, steps)
ts = task3(t*2)

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot (t , ts )
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title ('Part 3, task 3, t*2')
plt.show () 
  
#   Derivatives

t = np.arange(-5, 10 + steps, steps)
ts = task3(t)
dt = np.diff(t)
dy = np.diff(task3(t))/dt

plt.figure ( figsize = (15 , 10) )
plt.subplot (2 , 1 , 1)
plt.ylim((-2,10))
plt.plot(t, u)
plt.plot (t[range(len(dy))] , dy )
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title ('Part 3, task 3, derivative plot')
plt.show () 
    
    