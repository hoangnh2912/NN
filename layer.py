import numpy as np
from activation import ReLU, Sigmoid
from optimizer import Optimizer


class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, grads):
        pass

    def update(self, layer, optimizer, gradients):
        pass


class InputLayer(Layer):
    def __init__(self, input_shape):
        if type(input_shape) == int:
            self.input_shape = (1, input_shape)
        else:
            self.input_shape = input_shape
        self.output_shape = self.input_shape

    def forward(self, x):
        self.x = x
        return x

    def backward(self, x):
        return x


class Dense(Layer):
    def __init__(self, units, activation='relu', name=None):
        self.name = name
        self.units = units
        if activation == 'relu':
            self.activation = ReLU()
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        self.output_shape = (1, units)

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.init_weights(input_shape)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation.forward(
            np.dot(inputs, self.weights) + self.bias)
        return self.outputs

    def backward(self, grads):
        grads = self.activation.backward(grads)
        return np.dot(grads, self.weights.T)

    def init_weights(self, input_shape):
        self.weights = np.random.randn(input_shape[1], self.units)
        self.bias = np.random.randn(1, self.units)

    def update(self, layer, optimizer: Optimizer, gradients):
        optimizer.update(layer, gradients)
