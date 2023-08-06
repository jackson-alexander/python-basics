# Monty Hall Simulation
# Jackson Alexander

#### Settings ####
# import packages:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# general:
print('\n'*100) # pseudo-clear the console
np.random.seed(0) # set rng seed for reproducability

#### Code ####
num_doors = 3 # number of doors
samples = 1000 # number of samples
repititions = 1000 # number of repititions in each sample

def MH(num_doors,samples,repititions):
    prize = np.random.randint(num_doors,size=(repititions,samples)) # repititions-by-samples
    # chooses number [0,num_doors-2] (picks door with prize) for every possible sample-repitition pair
    # now we do the same but for the participant choosing a door:
    select = np.random.randint(num_doors,size=(repititions,samples)) # repititions-by-samples
    
    swap = np.zeros((repititions,samples))
    for i in range(0,samples): # loop over index of samples
        for j in range(0,repititions): # loop over index of repititions
            # we don't need to store the reveal variable:
            reveal = np.arange(0,num_doors) # row vector of all doors
            # note: prize and select match the column index for reveal
            reveal = np.delete(reveal,[prize[j,i],select[j,i]]) # remove price and select doors
            # Now, Monty will always reveal the first num_doors-2 doors (WLOG)
            # this really only matters for prize=select case
            reveal = reveal[list(range(0,num_doors-2))]
            # for example: prize=select=0, num_doors=100, reveal doors index 1-98
            # now we can swap to the door that's leftover:
            swap_temp = np.arange(0,num_doors)
            swap_temp = np.delete(swap_temp,[[select[j,i]],reveal.tolist()])

            # now we store the door we swap to:
            swap[j,i] = swap_temp[0] # scalar
    win = (prize==swap).astype(int)
    win_percent = np.mean(win,axis=0) # win % for each sample

    # now finding the cumulative win % at each sample (treating each repitition-sample pair as a trial)
    win_percent2 = np.zeros((1,samples))
    for i in range(0,samples): # loop over index of samples
        win_percent2[:,i] = np.mean(win[:,:i+1]) # default is mean of flattened array
    return win_percent, win_percent2

win_percent,win_percent2=MH(num_doors,samples,repititions)

# figures
fig,(ax1,ax2)=plt.subplots(2,1)
ax1.plot(np.arange(0,samples),win_percent2[0])
ax2.hist(win_percent)
plt.show()
