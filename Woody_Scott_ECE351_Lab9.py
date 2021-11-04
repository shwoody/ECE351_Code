#######################################
#                                     #
#  Scott Woody                        #
#  ECE 351-52                         #
#  Lab 9                              #
#  28OCT2021                          #
#                                     # 
#                                     # 
#######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.fftpack

steps = 1e-2
t = np.arange(0, 2, steps)


def ft(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return freq, X_mag, X_phi


x = np.cos(2*np.pi*t)
fs = 100
freq, X_mag, X_phi = ft(x, fs)

#Task 1 plot

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 1: cos(2*pi*t)')

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()


#Task 2
x = 5*np.sin(2*np.pi*t)
fs = 100
freq, X_mag, X_phi = ft(x, fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 2: 5*sin(2*pi*t)')

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

#Task 3
x = 2*np.cos((2*np.pi*t*2)-2) + np.sin((2*np.pi*6*t)+3)*np.sin((2*np.pi*6*t)+3)
fs = 100
freq, X_mag, X_phi = ft(x, fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 3')

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-15, 15)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-15, 15)
plt.xlabel ('f[Hz]')
plt.show ()



#Task 4

def ft1(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    for i in range(len(X_phi)):
        if abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0
    return freq, X_mag, X_phi

x1 = np.cos(2*np.pi*t)
freq1, X_mag1, X_phi1 = ft1(x1, fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x1)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 4 p1')

plt.subplot (3 , 2 , 3)
plt.stem(freq1, X_mag1)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq1, X_phi1)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq1, X_mag1)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq1, X_phi1)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

x2 = 5*np.sin(2*np.pi*t)
freq2, X_mag2, X_phi2 = ft1(x2, fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x2)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 4 p2')

plt.subplot (3 , 2 , 3)
plt.stem(freq2, X_mag2)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq2, X_phi2)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq2, X_mag2)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq2, X_phi2)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()


x3 = 2*np.cos((2*np.pi*t*2)-2) + np.sin((2*np.pi*6*t)+3)*np.sin((2*np.pi*6*t)+3)
freq3, X_mag3, X_phi3 = ft1(x3, fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x3)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 4 p3')

plt.subplot (3 , 2 , 3)
plt.stem(freq3, X_mag3)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq3, X_phi3)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq3, X_mag3)
plt.grid ()
plt.xlim(-15, 15)

plt.subplot (3 , 2 , 6)
plt.stem (freq3, X_phi3)
plt.grid ()
plt.xlim(-15, 15)
plt.xlabel ('f[Hz]')
plt.show ()


#Task5

T = 8
t = np.arange(0, 16, steps)
w0 = 2*np.pi/T

def x(k):
    b = 0
    y = 0
    for x in np.arange(1, k+1):
        b = 2/((x)*np.pi)*(1-np.cos((x)*np.pi))
        xt = b*np.sin(x*w0*t)
        y += xt
    return y
   

x3 = x(15)
freq3, X_mag3, X_phi3 = ft1(x3, fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x3)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 5')

plt.subplot (3 , 2 , 3)
plt.stem(freq3, X_mag3)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq3, X_phi3)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq3, X_mag3)
plt.grid ()
plt.xlim(-3, 3)

plt.subplot (3 , 2 , 6)
plt.stem (freq3, X_phi3)
plt.grid ()
plt.xlim(-3, 3)
plt.xlabel ('f[Hz]')
plt.show ()