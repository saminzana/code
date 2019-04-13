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
N=44
bits = np.random.randint(2, size=(4, (int(N/4))))

phi=np.array([[0,1,1,1],[1,1,1,0],[1,1,0,1]])

paritybits= (np.dot(phi,bits)%2)
codebits=np.vstack((bits,paritybits))
print("These are the message bits:\n", bits)
#print("This is the parity bit matrix:\n", phi)
print("These are the parity bits:\n",paritybits)
print("These are the code bits:\n",codebits)

p=0.3
def bsc(txBits,p): #simulates a binary symmetric channel with transition probability p
    flips = np.zeros((7,(int((N/4)))),dtype='bool') #there are no flips at this point
    x = np.random.rand(7,(int((N/4))))
    flips[x<p] = True
    rxBits = np.logical_xor(txBits,flips)
    return rxBits

recievedbits=(bsc(codebits,p)).astype(int)
print("These are the recieved bits:\n", recievedbits)

