from Dense import Dense
from Tanh import Tanh
from MSE import mse,mse_prime

import numpy as np

X=np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))

Y=np.reshape([[0],[1],[1],[0]],(4,1,1))

network=[
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

epochs=10000
learning_rate=0.1

for e in range(epochs):
    error=0
    for i in range(0,4):
        x=X[i]
        y=Y[i]
        
        output=x
        for layer in network:
            output=layer.forward(output)
        error+=mse(y,output)
        
        error_gradient=mse_prime(y,output)
        
        for layer in reversed(network):
            error_gradient=layer.backward(error_gradient,learning_rate)
            
    error=error/len(X)

    print(f'Epochs = {e+1}/{epochs}, error = {error}')
    

for i in range(0,4):
	x=X[i]
	output=x
	for layer in network:
		output=layer.forward(output)
		
	print(np.round(output[0]))