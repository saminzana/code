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
bits = np.random.randint(2, size=(4, (int(N/4)))).astype(int)

phi=np.array([[0,1,1,1],[1,1,1,0],[1,1,0,1]]).astype(int)

paritybits= (np.dot(phi,bits)%2).astype(int)
codebits=np.vstack((bits,paritybits)).astype(int)
print("These are the message bits:\n", bits)
print("This is the parity bit matrix:\n", phi)
print("These are the parity bits:\n",paritybits)
print("These are the code bits:\n",codebits)

y2=np.array([])
y1=np.array([])
for p in np.arange(-3.0, -0.29, 0.01):
    def bsc(txBits,p): #simulates a binary symmetric channel with transition probability p
        flips = np.zeros((7,(int((N/4)))),dtype='bool') #there are no flips at this point
        x = np.random.rand(7,(int((N/4))))
        flips[x<p] = True
        rxBits = np.logical_xor(txBits,flips)
        return rxBits
    H=np.hstack((phi, np.array([[1,0,0],[0,1,0],[0,0,1]]))).astype(int)
    #print("This is H:\n", H)
    recievedbits=(bsc(codebits,10**p)).astype(int)
    #print("These are the recieved bits:\n", recievedbits)
    syndromebits=(np.dot(H,recievedbits)%2).astype(int)
    #print ("The Syndrome Bits are:\n", syndromebits)
    errorbits=(np.logical_xor(codebits,recievedbits)).astype(int)
    #print("The bits in error are:\n", errorbits)
    
    shape=(7,(int(N/4)))
    biterror= (np.zeros(shape)).astype(int)
    
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
       # print ("The new bit error:\n", biterror)
    
        correctedbits=(np.logical_xor(recievedbits,biterror)).astype(int)
        numberoferrors=np.sum(np.logical_xor(correctedbits,codebits))
        BER=numberoferrors/N
        y1=np.append(y1,BER)
        #print("The corrected bits are:\n", correctedbits)
       # print("The number of errors are:\n", numberoferrors)



################################################################################################################################################


bits1 = np.random.randint(2, size=(11, (int(N/11)))).astype(int)

phi1=np.array([[1,1,0,1,1,0,1,0,1,0,1],[1,0,1,1,0,1,1,0,0,1,1],[0,1,1,1,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,1]]).astype(int)

paritybits1= (np.dot(phi1,bits1)%2).astype(int)
codebits1=np.vstack((bits1,paritybits1)).astype(int)
   # print("These are the message bits:\n", bits1)
   # print("This is the parity bit matrix:\n", phi1)
#print("These are the parity bits:\n",paritybits1)
#print("These are the code bits:\n",codebits1)

for p in np.arange(-3.0, -0.29, 0.01):
    def bsc1(txBits,p): #simulates a binary symmetric channel with transition probability p
        flips = np.zeros((15,(int((N/11)))),dtype='bool') #there are no flips at this point
        x = np.random.rand(15,(int((N/11))))
        flips[x<p] = True
        rxBits = np.logical_xor(txBits,flips)
        return rxBits
    H1=(np.hstack((phi1, np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])))).astype(int)
    #print("This is H:\n", H1)
    recievedbits1=(bsc1(codebits1,10**p)).astype(int)
   # print("These are the recieved bits:\n", recievedbits1)
    syndromebits1=(np.dot(H1,recievedbits1)%2).astype(int)
    #print ("The Syndrome Bits are:\n", syndromebits1)
    errorbits1=(np.logical_xor(codebits1,recievedbits1)).astype(int)
   # print("The bits in error are:\n", errorbits1)
    
    shape1=(15,(int(N/11)))
    biterror1= (np.zeros(shape1)).astype(int)
    
    for x in range (0,int(N/11)-1):
        if(syndromebits1[0,x]==H1[0,0] and syndromebits1[1,x]==H1[1,0] and syndromebits1[2,x]==H1[2,0] and syndromebits1[3,x]==H1[3,0]):
            biterror1[0,x]=1
        
        elif(syndromebits1[0,x]==H1[0,1] and syndromebits1[1,x]==H1[1,1] and syndromebits1[2,x]==H1[2,1] and syndromebits1[3,x]==H1[3,0]):
            biterror1[1,x]=1
            
        elif(syndromebits1[0,x]==H1[0,2] and syndromebits1[1,x]==H1[1,2] and syndromebits1[2,x]==H1[2,2] and syndromebits1[3,x]==H1[3,0]):
            biterror1[2,x]=1
            
        elif(syndromebits1[0,x]==H1[0,3] and syndromebits1[1,x]==H1[1,3] and syndromebits1[2,x]==H1[2,3] and syndromebits1[3,x]==H1[3,0]):
            biterror1[3,x]=1
            
        elif(syndromebits1[0,x]==H1[0,4] and syndromebits1[1,x]==H1[1,4] and syndromebits1[2,x]==H1[2,4] and syndromebits1[3,x]==H1[3,0]):
            biterror1[4,x]=1
            
        elif(syndromebits1[0,x]==H1[0,5] and syndromebits1[1,x]==H1[1,5] and syndromebits1[2,x]==H1[2,5] and syndromebits1[3,x]==H1[3,0]):
            biterror1[5,x]=1
            
        elif(syndromebits1[0,x]==H1[0,6] and syndromebits1[1,x]==H1[1,6] and syndromebits1[2,x]==H1[2,6] and syndromebits1[3,x]==H1[3,0]):
            biterror1[6,x]=1
        
        elif(syndromebits1[0,x]==H1[0,7] and syndromebits1[1,x]==H1[1,7] and syndromebits1[2,x]==H1[2,7] and syndromebits1[3,x]==H1[3,0]):
            biterror1[7,x]=1
            
        elif(syndromebits1[0,x]==H1[0,8] and syndromebits1[1,x]==H1[1,8] and syndromebits1[2,x]==H1[2,8] and syndromebits1[3,x]==H1[3,0]):
            biterror1[8,x]=1
            
        elif(syndromebits1[0,x]==H1[0,9] and syndromebits1[1,x]==H1[1,9] and syndromebits1[2,x]==H1[2,9] and syndromebits1[3,x]==H1[3,0]):
            biterror1[9,x]=1
            
        elif(syndromebits1[0,x]==H1[0,10] and syndromebits1[1,x]==H1[1,10] and syndromebits1[2,x]==H1[2,10] and syndromebits1[3,x]==H1[3,0]):
            biterror1[10,x]=1
            
        elif(syndromebits1[0,x]==H1[0,11] and syndromebits1[1,x]==H1[1,11] and syndromebits1[2,x]==H1[2,11] and syndromebits1[3,x]==H1[3,0]):
            biterror1[10,x]=1
            
        elif(syndromebits1[0,x]==H1[0,12] and syndromebits1[1,x]==H1[1,12] and syndromebits1[2,x]==H1[2,12] and syndromebits1[3,x]==H1[3,0]):
            biterror1[12,x]=1
            
        elif(syndromebits1[0,x]==H1[0,13] and syndromebits1[1,x]==H1[1,13] and syndromebits1[2,x]==H1[2,13] and syndromebits1[3,x]==H1[3,0]):
            biterror1[13,x]=1
            
        elif(syndromebits1[0,x]==H1[0,14] and syndromebits1[1,x]==H1[1,14] and syndromebits1[2,x]==H1[2,14] and syndromebits1[3,x]==H1[3,0]):
            biterror1[14,x]=1
  
        #print ("The new bit error:\n", biterror1)
        
        correctedbits1=(np.logical_xor(recievedbits1,biterror1)).astype(int)
        numberoferrors1=np.sum(np.logical_xor(correctedbits1,codebits1))
        BER1=numberoferrors1/N
        y2=np.append(y2,BER1)
        #print("The corrected bits are:\n", correctedbits1)
        #print("The number of errors are:\n", numberoferrors1)
        
toc=time.time()
r=np.arange(-3.0, -0.29, 0.01)
y1=np.log10(y1)
y2=np.log10(y2)

fig1, ax= plt.subplots()
ax.plot(r,y1, label= "Error rate of r=3")
ax.plot(r,y2, label= "Error rate of r=5")
plt.xlabel("Transition Probability")
plt.ylabel("log_10 (Error Rate)")
ax.legend()
ax.grid()
plt.show()
print ("The total run time is:", (toc-tic))











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