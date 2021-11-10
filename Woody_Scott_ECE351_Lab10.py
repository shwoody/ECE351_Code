#######################################
#                                     #
#  Scott Woody                        #
#  ECE 351-52                         #
#  Lab 10                             #
#  04NOV2021                          #
#                                     # 
#                                     # 
#######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

steps = 1
w = np.arange(1e3, 1e6 +steps, steps)

def mag(R, C, L, w):
    y = (w/(R*C))/np.sqrt(w**4+(1/(L*C))**2+(1/(R*C)**2-2/(L*C))*w**2)
    y = 20*np.log(y)
    return y

def ang(R, C, L, w):
    
    x = np.pi/2 - np.arctan((w/(R*C))/((1/(L*C))-w**2))
    x = np.degrees(x)
    for i in range(len(w)):  
        if x[i] > 90:
            x[i] = x[i]-180
    return x
R = 1000
C = 100e-9
L = 27e-3 

H1 =  mag(R, C, L, w)
H1a = ang(R, C, L, w)


#Task 1
plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.semilogx(w, H1)
plt.grid ()
plt.ylabel ('H magnitude')
plt.title('Task 1')

plt.subplot (2 , 1 , 2)
plt.semilogx(w, H1a)
plt.grid ()
plt.ylabel ('H angle')
plt.xlabel ('w')


#Task 2

hs = scipy.signal.TransferFunction([1/(R*C), 0], [1, 1/(R*C), 1/(L*C)])
w, mag, phase = scipy.signal.bode(hs, w)

plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.semilogx(w, mag)
plt.grid ()
plt.xlim(1e3, 1e6)
plt.ylabel ('H magnitude')
plt.title('Task 2')

plt.subplot (2 , 1 , 2)
plt.semilogx(w, phase)
plt.grid ()
plt.xlim(1e3, 1e6)
plt.ylabel ('H angle')
plt.xlabel ('w')

#Task 3
import control as con # this package is not included in the Anaconda
# distribution , but should have been installed in lab 0
num = [1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]
sys = con.TransferFunction ( num , den )
plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )
# use _ = ... to suppress the output

#Part2 Task 1

fs = 1000000
steps = 1/fs
t = np.arange(0, .01 + steps, steps)

x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

plt.figure ( figsize = (10 , 7) )
plt.subplot (1 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('x(t)')
plt.xlabel ('t')
plt.title('Part 2 Task 1')

#Part2 Task 2

numz, denz = scipy.signal.bilinear(num, den, fs)
y = scipy.signal.lfilter(numz, denz, x)



plt.figure ( figsize = (10 , 7) )
plt.subplot (1 , 1 , 1)
plt.plot (t , y)
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title('Part 2 Task 4')



