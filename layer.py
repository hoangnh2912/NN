import numpy as np
from activation import ReLU, Sigmoid


class Layer:
    def __init__(self):
        pass


class Dense(Layer):
    def __init__(self, units, activation='relu'):
        self.units = units
        if activation == 'relu':
            self.activation = ReLU()
        if activation == 'sigmoid':
            self.activation = Sigmoid()

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.bias
        self.outputs = self.activation.forward(self.outputs)
        return self.outputs

    def backward(self, grads):
        self.grads = grads
        self.dweights = np.dot(self.inputs.T, grads)
        self.dbias = np.sum(grads, axis=0)
        self.dinputs = np.dot(grads, self.weights.T)
        self.dinputs = self.activation.backward(self.dinputs)
        return self.dinputs

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias

    def init_weights(self, shape):
        self.weights = np.random.randn(*shape) / np.sqrt(shape[0])
        self.bias = np.zeros((1, shape[1]))
