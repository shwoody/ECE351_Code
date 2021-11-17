#######################################
#                                     #
#  Scott Woody                        #
#  ECE 351-52                         #
#  Lab 11                             #
#  11NOV2021                          #
#                                     # 
#                                     # 
#######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


numz = [2, -40]
denz = [1, -10, 16]
print(scipy.signal.residuez(numz, denz))


def zplane (b , a , filename = None ) :

    import numpy as np
    import matplotlib . pyplot as plt
    from matplotlib import patches

# get a figure / plot
    ax = plt . subplot (1 , 1 , 1)

# create the unit circle
    uc = patches.Circle ((0 ,0) , radius =1 , fill = False , color ='black' , ls ='dashed')
    ax . add_patch ( uc )

# the coefficients are less than 1 , normalize the coefficients
    if np.max( b ) > 1:
        kn = np . max( b )
        b = np . array ( b ) / float ( kn )
    else :
        kn = 1

    if np . max( a ) > 1:
        kd = np . max( a )
        a = np . array ( a ) / float ( kd )
    else :
        kd = 1

# get the poles and zeros
    p = np . roots ( a )
    z = np . roots ( b )
    k = kn / float ( kd )

# plot the zeros and set marker properties
    t1 = plt . plot ( z . real , z . imag , 'o', ms =10 , label ='Zeros')
    plt.setp ( t1 , markersize =10.0 , markeredgewidth =1.0)

# plot the poles and set marker properties
    t2 = plt.plot ( p . real , p . imag , 'x', ms =10 , label ='Poles')
    plt.setp ( t2 , markersize =12.0 , markeredgewidth =3.0)
    
    ax.spines ['left']. set_position ('center')
    ax.spines ['bottom']. set_position ('center')
    ax.spines ['right']. set_visible ( False )
    ax.spines ['top']. set_visible ( False )
    
    plt . legend ()

# set the ticks

# r = 1.5; plt. axis ( ’ scaled ’); plt. axis ([ -r, r, -r, r])
# ticks = [ -1 , -.5 , .5 , 1]; plt. xticks ( ticks ); plt. yticks ( ticks )

    if filename is None :
        plt . show ()
    else :
        plt . savefig ( filename )
    
    return z , p , k

zplane(numz,denz, None)
 

#Task 5
w, h = scipy.signal.freqz(numz, denz, whole = True)


plt.figure ( figsize = (10 , 7) )
plt.subplot (2 , 1 , 1)
plt.plot(w/np.pi, 20 * np.log10(abs(h)))
plt.grid ()
plt.ylabel ('Amplitude [dB]')
plt.title('Task 5')

plt.subplot (2 , 1 , 2)
plt.plot(w/np.pi, np.angle(h))
plt.grid ()
plt.ylabel ('Angle (radians)')
plt.xlabel ('Frequency [rad/sample]')



