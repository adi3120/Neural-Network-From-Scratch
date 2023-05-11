from Activation import Activation
import numpy as np

def tanh(x):
	return np.tanh(x)
def tanh_prime(x):
	return 1-np.tanh(x)**2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh,tanh_prime)

            
