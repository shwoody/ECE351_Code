#######################################
#                                     #
#  Scott Woody                        #
#  ECE 351-52                         #
#  Lab 7                              #
#  14OCT2021                          #
#                                     # 
#                                     # 
#######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 1e-3
t = np.arange(0, 5 + steps, steps)

#Part1 Task1
#g(s)
num = [0, 1, 9]
den = [1, -2, -40, -64]
print(sig.residue(num,den))
#g(s)= -.35/(s+2)+.208/(s+4)+.1417/(s-8) zero at s=-9
#a(s)
num = [1, 4]
den = [1, 4, 3]
print(sig.residue(num, den))
#a(s) = 1.5/(s+1)-.5/(s+3) zero at s=-4
#b(s)
p1 = (-26+np.sqrt(26*26-4*168))/2
p2 = (-26-np.sqrt(26*26-4*168))/2
print(p1)
print(p2)
#b(s) zeros at s=-12 and -14

#Part1 Task 2
num = [0, 1, 9]
den = [1, -2, -40, -64]
print(sig.tf2zpk(num, den))
num = [1, 4]
den = [1, 4, 3]
print(sig.tf2zpk(num, den))
print(np.roots([1,26,168]))

#Part1 Task 3

#H_open = (s+9)/((s+2)(s-8)(s+1)(s+3))

#Part1 Task 4

#This is not response stable since one denominator
#pole is s-8 which results in an unstable response

#Part1 Task 5
num = sig . convolve ([1 , 9] , [1 , 4])
print ('Numerator = ', num )

den = sig . convolve ([1, -2 , -40 , -64] , [1, 4 , 3])
print (' Denominator = ', den )

to, yo = sig.step((num, den), T=t)

plt.figure ( figsize = (10 , 7) )
plt.plot (to , yo)
plt.grid ()
plt.ylabel ('scipy step')
plt.xlabel ('t')
plt.show ()

#Part1 Task 6

# The result does support the answer in Task 4
# This is because the response goes to infinity


#Part2 Task1
#H_close = [numA*numG]/(denA*denG+denA*numB*numG)

#Part2 Task2
numA = [1, 4]
numG = [1, 9]
numB = [1, 26, 168]
denA = [1, 4, 3]
denG = [1, -2, -40, -64]
numF = sig.convolve(numA, numG)
num2 = sig.convolve(numB, numG)
num1 = sig.convolve(denA, denG)
num2 = sig.convolve(num2, denA)
print(numF)
print(num1)
print(num2)
denF = num1 + num2
print(denF)

print(sig.tf2zpk(numF, denF))

#H_close = ((s+4)(s+9))/((s+3)(s+1)(s+6.175)(s+5.16-9.51j)(s+5.16+9.51j))

#Part2 Task 3
#The closed loop is stable since all of the denominator
#poles are positive real part. This will stabalize

#Part2 Task 4

to, yo = sig.step((numF, denF), T=t)

plt.figure ( figsize = (10 , 7) )
plt.plot (to , yo)
plt.grid ()
plt.ylabel ('scipy step')
plt.xlabel ('t')
plt.show ()

#Part2 Task 5

#This result supports my answer since the value
#is settling as time goes to infinity
