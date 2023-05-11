from Layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self,input_size,output_size):
        self.weights=np.random.randn(output_size,input_size)
        self.bias=np.random.randn(output_size,1)
        
    def forward(self,input):
        self.input=input
        output=np.dot(self.weights,self.input)+self.bias
        return output
    
    def backward(self,output_gradient,learning_rate):
        weight_gradient=np.dot(output_gradient,self.input.T)
        self.weights=self.weights-learning_rate*weight_gradient
        self.bias=self.bias-learning_rate*output_gradient
        input_gradient=np.dot(self.weights.T,output_gradient)
        
        return input_gradient
        
        