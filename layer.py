import numpy as np
from activation import ReLU, Sigmoid


class Layer:
    weights = None
    bias = None
    units = None
    input_shape = None
    inputs = None
    outputs = None
    grads = None
    weights_grad = None
    bias_grad = None
    m_weights = None
    v_weights = None
    m_bias = None
    v_bias = None

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, grads):
        pass

    def set_input_shape(self, input_shape):
        pass


class Dense(Layer):
    weights = None
    bias = None
    units = None
    input_shape = None
    inputs = None
    outputs = None
    grads = None
    weights_grad = None
    bias_grad = None
    m_weights = None
    v_weights = None
    m_bias = None
    v_bias = None

    def __init__(self, units, activation='relu', name=None, input_shape=None):
        self.name = name
        self.units = units
        if activation == 'relu':
            self.activation = ReLU()
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        self.output_shape = (1, units)
        if input_shape is not None:
            self.set_input_shape(input_shape)

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.init_weights(input_shape)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation.forward(
            np.dot(inputs, self.weights) + self.bias)
        return self.outputs

    def backward(self, grads):
        self.grads = self.activation.backward(self.outputs) * grads
        self.weights_grad = np.dot(self.inputs.T, self.grads)
        self.bias_grad = np.sum(self.grads, axis=0, keepdims=True)

        if self.m_weights is None:
            # self.m_weights = np.random.rand(*self.weights_grad.shape)
            # self.v_weights = np.random.rand(*self.weights_grad.shape)
            # self.m_bias = np.random.rand(*self.bias_grad.shape)
            # self.v_bias = np.random.rand(*self.bias_grad.shape)
            self.m_weights = np.zeros(self.weights_grad.shape)
            self.v_weights = np.zeros(self.weights_grad.shape)
            self.m_bias = np.zeros(self.bias_grad.shape)
            self.v_bias = np.zeros(self.bias_grad.shape)
        return np.dot(self.grads, self.weights.T)

    def init_weights(self, input_shape):
        self.weights = np.random.randn(input_shape[1], self.units)
        self.bias = np.random.randn(1, self.units)
