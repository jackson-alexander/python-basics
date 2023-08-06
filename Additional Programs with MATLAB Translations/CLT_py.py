# CLT Simulation
# Jacksom Alexander

#### Settings ####
# import packages:
import numpy as np # numpy is an array package
import matplotlib.pyplot as plt # for figures
import matplotlib.animation as animation # for animated figures
# general:
print('\n'*100) # pseudo-clear the console
np.random.seed(0) # set rng seed for reproducability

#### Code ####
# Parameters
n = [1,2,10,100] # sample size(s), must be list (in brackets)
samples = 1000 # number of samples
a = 0; b = 1 # uniform distribution parameters

def CLT(n,samples,a,b):
    mu = (b-a)/2 # E[X]
    Xn = np.random.uniform(0,1,(max(n),samples)) # Xn ~ U(0,1) n-by-samples array
    CLT=np.empty((len(n),samples)) # preallocate
    for i in range(0,len(n)): # note: indexing begins at 0, range(a,b) begins at 'a' but does not include 'b'.
        Xn_temp = Xn[:n[i],:] # note: numpy indexes x[i:j:k] where k is the step (different than MATLAB)
        Xn_temp_bar = np.mean(Xn_temp,axis=0) # axis=0 results in row vector (mean of columns)
        CLT[i,:] = np.sqrt(n[i])*(Xn_temp_bar-mu) # store row vector in array
    return CLT

CLT = CLT(n,samples,a,b)

#### Plots ####
# animated figure
if (len(n) % 2)==0:
    sz1 = np.size(np.reshape(n,(2,-1)),axis=0)
    sz2 = np.size(np.reshape(n,(2,-1)),axis=1)
elif (len(n) % 3)==0:
    sz1 = np.size(np.reshape(n,(3,-1)),axis=0)
    sz2 = np.size(np.reshape(n,(3,-1)),axis=1)
else:
    sz1=1
    sz2=len(n)

fig,axs = plt.subplots(sz1,sz2,figsize=(8,8),sharey=True,sharex=True)
# set titles and y limits

def update_hist(num):
    #num=10*num
    #if num>samples: return
    for i,ax in enumerate(axs.flatten()):
        ax.cla()
        ax.hist(CLT[i,:num*20],bins=30,range=[-b,b])
        ax.set_title('n=' + str(n[i]))
        ax.set_ylim((0,100))
    return axs

animation = animation.FuncAnimation(fig,update_hist,samples//20,repeat=False)
plt.show()


