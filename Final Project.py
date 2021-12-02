#######################################
#                                     #
#  Scott Woody                        #
#  ECE 351-52                         #
#  Lab 12                             #
#  28NOV2021                          #
#                                     # 
#                                     # 
#######################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import scipy.fftpack

# load input signal
df = pd.read_csv ('NoisySignal.csv ')

t = df['0'].values
sensor_sig = df['1'].values

plt . figure ( figsize = (10 , 7) )
plt . plot (t , sensor_sig )
plt . grid ()
plt . title ('Noisy Input Signal ')
plt . xlabel ('Time [s]')
plt . ylabel ('Amplitude [V]')
plt . show ()

def ft1(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    for i in range(len(X_phi)):
        if abs(X_mag[i]) < .05:
            X_phi[i] = 0
    return freq, X_mag, X_phi

fs = 1e6
freq, X_mag, X_phi = ft1(sensor_sig, fs)


def make_stem ( ax ,x ,y , color ='k', style ='solid', label ='', linewidths =2.5 ,** kwargs ) :
    ax . axhline ( x [0] , x [ -1] ,0 , color ='r')
    ax . vlines (x , 0 ,y , color = color , linestyles = style , label = label , linewidths = linewidths )
    ax . set_ylim ([1.05* y . min () , 1.05* y . max () ])

#single plot    
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid ()
plt . show ()
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_phi )
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')
plt.grid ()
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid ()
plt.xlim(0, 1800)
plt . show ()
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_phi )
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')
plt.xlim(0, 1800)
plt.grid ()
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid ()
plt.xlim(1800, 2000)
plt . show ()
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_phi )
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')
plt.xlim(1800, 2000)
plt.grid ()
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid ()
plt.xlim(2000, 1e5)
plt . show ()
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_phi )
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')
plt.xlim(2000, 1e5)
plt.grid ()
plt . show ()

# for subplots of the orientation plt. subplot (2 ,1 ,_)
#fig , ( ax1 , ax2 ) = plt.subplots (2 , 1 , figsize =(10 , 7) )
#plt . subplot ( ax1 )
#make_stem ( ax1 ,freq, X_mag)
#plt . subplot ( ax2 )
#make_stem ( ax2 ,freq, X_phi)
#plt . show ()


#Part 2
C = 35.26e-9
L = .199
R = 1000

num = [R/L, 0]
den = [1, R/L, 1/(L*C)]

steps = 1
f = np.arange(1, 1e6 +steps, steps)
w = f*2*np.pi

numz, denz = scipy.signal.bilinear(num, den, fs)
y = scipy.signal.lfilter(numz, denz, sensor_sig)

plt.figure ( figsize = (10 , 7) )
plt.subplot (1 , 1 , 1)
plt.plot (t , y)
plt.grid ()
plt.ylabel ('y(t)')
plt.xlabel ('t')
plt.title('Filtered Signal')

import control as con
# part 3
sys = con.TransferFunction ( num , den )
plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )

plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , np.arange(1, 1800 +steps, steps)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )

plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , np.arange(1800, 2000 +steps, steps)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )

plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , np.arange(2000, 1e6 +steps, steps)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )

#part 4
freq, X_mag, X_phi = ft1(y, fs)

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid ()
plt . show ()
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_phi )
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')
plt.grid ()
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid ()
plt.xlim(0, 1800)
plt . show ()
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_phi )
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')
plt.xlim(0, 1800)
plt.grid ()
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid ()
plt.xlim(1800, 2000)
plt . show ()
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_phi )
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')
plt.xlim(1800, 2000)
plt.grid ()
plt . show ()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.grid ()
plt.xlim(2000, 1e5)
plt . show ()
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_phi )
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')
plt.xlim(2000, 1e5)
plt.grid ()
plt . show ()