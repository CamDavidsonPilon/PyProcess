import math
import pyprocess as SP
import scipy.stats as stats 

#lets create a Wiener process conditioned at an end position (Brownian Bridge) 
#Initialize the initial values and parameters
wienerParams = {"mu":0, "sigma":1}
wienerInitial = {"startTime":0, "startPosition":0, "endTime":12, "endPosition": 1 }
Wiener = SP.Wiener_process(wienerParams, wienerInitial)

#Get the mean and variance at time=10
t=10

print Wiener.get_mean_at(t)
print Wiener.get_variance_at(t)
print
#Generate a sample position at time=10
print Wiener.generate_position_at(t)
print
#Generate a sample path at set times in an array:
times = range(10) # = [0,1,2,3,4,5,6,7,8,9]
print Wiener.generate_sample_path(times)
# will print an array of points [(t, W_t) for t in times]



# Lets create a custom diffusion. We specify the drift and diffusion functions in the SDE:
# The functions must have time and space parameters, though they don't need to depend on them.
def drift(x,t):
    return t*x

#the diffusion function must be non-negative in it's domain.
def diffusion(x,t):
    return math.sqrt(abs(x))

customParams={"a":drift, "b":diffusion}
customInitial = {"startTime":0, "startPosition":1}
Custom = SP.Custom_diffusion(customParams, customInitial)

#get the mean and poisiton at t=10.
#As this diffusion is untractable, we use MC methods. You can specify the accuracy in the library.
print Custom.generate_position_at(10)



#Lets create a process which can be seperated into the sum of a CEV diffusion and Compound Poisson jump process.
#First the CEV Process
CEVParams = {"r":0.002, "delta":0.5, "beta":-0.3}
CEVInitial = {"startTime":0, "startPosition":100}
CEV = SP.CEV_process(CEVParams, CEVInitial)

#Next the Compound Process. The jumps are Normal random variables, so lets create a "frozen" scipy.stats random variable. 
#Frozen just means that the parameters are fixed upon construction of the random variable.

Nor = stats.norm(-0.2,0.05) #The parameters represent the mean and standard deviation.
rate = 0.1 #This is the jump rate, often denoted lambda in literature.
CPoiParams = {"J":Nor, "rate":rate}
CPoiInitial = {"startTime":0, "startPosition":0}
CPoi = SP.Compound_poisson_process(CPoiParams, CPoiInitial)

#create the custom process:
CustomProcess = SP.Custom_process(GBM, CPoi)

#play around with it?
print CustomProcess.get_mean_at(10)
print CustomProcess.generate_sample_path(range(100))


#Any question please email me at cam.davidson.pilon@gmail.com
