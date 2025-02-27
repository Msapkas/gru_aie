import numpy as np

def sigmoid_lut(x,N):
    interval = np.linspace(-x,x,N)
    return 1/(1 + np.exp(-interval))
    
def tanh_lut(x, N):
    interval = np.linspace(-x,x,N)
    return np.tanh(interval)
    
np.savetxt(sigmoid_lut_4096, sigmoid_lut(-6,6,4096), fmt='%f')
np.savetxt(tanh_lut_4096, tanh_lut(-6,6,4096), fmt='%f')

