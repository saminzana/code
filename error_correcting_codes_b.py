# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:55:01 2019

@author: SAMESUNG
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import stats
import time
tic=time.time()


#N=1000032
N=1000000
bits = np.random.randint(2, size=(4, (int(N/4)))).astype(int)

phi=np.array([[0,1,1,1],[1,1,1,0],[1,1,0,1]]).astype(int)

paritybits= (np.dot(phi,bits)%2).astype(int)
codebits=np.vstack((bits,paritybits)).astype(int)
print("These are the message bits:\n", bits)
print("This is the parity bit matrix:\n", phi)
print("These are the parity bits:\n",paritybits)
print("These are the code bits:\n",codebits)

p=0.3
def bsc(txBits,p): #simulates a binary symmetric channel with transition probability p
    flips = np.zeros((7,(int((N/4)))),dtype='bool') #there are no flips at this point
    x = np.random.rand(7,(int((N/4))))
    flips[x<p] = True
    rxBits = np.logical_xor(txBits,flips)
    return rxBits
H=np.hstack((phi, np.array([[1,0,0],[0,1,0],[0,0,1]]))).astype(int)
print("This is H:\n", H)
recievedbits=(bsc(codebits,p)).astype(int)
print("These are the recieved bits:\n", recievedbits)
syndromebits=(np.dot(H,recievedbits)%2).astype(int)
print ("The Syndrome Bits are:\n", syndromebits)
errorbits=(np.logical_xor(codebits,recievedbits)).astype(int)
print("The bits in error are:\n", errorbits)

shape=(7,(int(N/4)))
biterror= (np.zeros(shape)).astype(int)

#for x in range (0,int(N/4)):
#    if syndromebits[:,x]==H[:,0]:
#        biterror[0,x]==1
#    elif syndromebits[:,x]==H[:,1]:
#        biterror[1,x]==1
#    elif syndromebits[:,x]==H[:,2]:
#        biterror[2,x]==1
#    elif syndromebits[:,x]==H[:,3]:
#        biterror[3,x]==1
#    elif syndromebits[:,x]==H[:,4]:
#        biterror[4,x]==1
#    elif syndromebits[:,x]==H[:,5]:
#        biterror[5,x]
#    elif syndromebits[:,x]==H[:,6]:
#        biterror[6,x]==1

#for each column in the syndrome bits compare to the columns in H. where they are the same (if they are not the same dont change the message)
#the H column will change the row of biterror[]  and the syndrome bits are the columns of biterror[]. put a 1 in the index[H,syndromebits]
# this new matrix biterror will not equal errorbits but they should be similar
for x in range (0,int(N/4)-1):
    if(syndromebits[0,x]==H[0,0] and syndromebits[1,x]==H[1,0] and syndromebits[2,x]==H[2,0]):
        biterror[0,x]=1
    
    elif(syndromebits[0,x]==H[0,1] and syndromebits[1,x]==H[1,1] and syndromebits[2,x]==H[2,1]):
        biterror[1,x]=1
        
    elif(syndromebits[0,x]==H[0,2] and syndromebits[1,x]==H[1,2] and syndromebits[2,x]==H[2,2]):
        biterror[2,x]=1
        
    elif(syndromebits[0,x]==H[0,3] and syndromebits[1,x]==H[1,3] and syndromebits[2,x]==H[2,3]):
        biterror[3,x]=1
        
    elif(syndromebits[0,x]==H[0,4] and syndromebits[1,x]==H[1,4] and syndromebits[2,x]==H[2,4]):
        biterror[4,x]=1
        
    elif(syndromebits[0,x]==H[0,5] and syndromebits[1,x]==H[1,5] and syndromebits[2,x]==H[2,5]):
        biterror[5,x]=1
        
    elif(syndromebits[0,x]==H[0,6] and syndromebits[1,x]==H[1,6] and syndromebits[2,x]==H[2,6]):
        biterror[6,x]=1
print ("The new bit error:\n", biterror)

correctedbits=(np.logical_xor(recievedbits,biterror)).astype(int)
numberoferrors=np.sum(np.logical_xor(correctedbits,codebits))
print("The corrected bits are:\n", correctedbits)
print("The number of errors are:\n", numberoferrors)

