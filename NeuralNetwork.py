import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_pass(self, input):
        pass

    def backward_pass(self, output_gradient, learning_rate):
        pass



class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)
    
    def forward_pass(self, input):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias
    
    def backward_pass(self, output_gradient, learning_rate):
        input_error = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)
        #print(type(learning_rate))
        #print(weights_gradient.shape)
        #print(output_gradient.shape)
        #print(self.weights.shape)
        #print(self.bias.shape)
        #print(self.input.shape)
        self.weights = self.weights - np.multiply(weights_gradient, learning_rate)
        self.bias = self.bias - np.multiply(output_gradient, learning_rate)
        return input_error



class Activation(Layer):
    def __init__(self, activation, activation_deriv):
        self.activation = activation
        self.activation_deriv = activation_deriv
    
    def forward_pass(self, input):
        self.input = input
        #print(self.activation(self.input))
        return self.activation(self.input)
    
    def backward_pass(self, output_gradient, learning_rate):
        #print(output_gradient)
        #print(self.activation_deriv(self.input))
        return np.multiply(output_gradient, self.activation_deriv(self.input))



class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_deriv = None
    
    def add(self, layer):
        self.layers.append(layer)
    
    def use(self, loss, loss_deriv):
        self.loss = loss
        self.loss_deriv = loss_deriv
    
    def predict(self, input_data):
        features, samples = input_data.shape
        result = []

        for i in range(samples):
            output = input_data[:, i]
            output= np.reshape(output, (output.shape[0], 1))
            output = output.T
                 
            for layer in self.layers:
                output = layer.forward_pass(output)
            result.append(output)
        
        #return result
        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        features, samples = x_train.shape
    
        # loop over all the training samples
        for i in range(epochs):
            error = 0
            for j in range(samples):
                # forward propogation
                output = x_train[:, j]
                output = np.reshape(output, (output.shape[0], 1))
                output = output.T
                
                for layer in self.layers:
                    output = layer.forward_pass(output)

                # compute loss (for display purposes only)
                error += self.loss(y_train[j], output)

                # backward propogation to update the parameters w and b
                error = self.loss_deriv(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_pass(error, learning_rate)
                
                # calculate the average error on all of the samples
                error /= samples
                print(f'epoch {i + 1}{epochs}, error = {error}')