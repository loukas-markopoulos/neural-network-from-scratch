import numpy as np
from NeuralNetwork import Activation



def relu(x):
    x = x.astype('float64')
    return np.maximum(0,x)

def relu_deriv_base(x):
    if x > 0:
       return 1
    else:
        return 0 

relu_deriv_vectorized = np.vectorize(relu_deriv_base)

def relu_deriv(x):
    return relu_deriv_vectorized(x)



def sigmoid(x):
    x = x.astype('float64')
    return 1 / (1 + np.exp(-1 * x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))




def tanh(x):
    x = x.astype('float64')
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - tanh(x) ** 2

