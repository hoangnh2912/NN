import numpy as np


class Activation:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self, input, grad_output):
        pass


class ReLU(Activation):

    def forward(self, input):
        self.input = input
        self.output = input.copy()
        self.output[self.output < 0] = 0
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input < 0] = 0
        return grad_input


class Sigmoid(Activation):

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output * (1 - self.output) * self.output
        return grad_input
