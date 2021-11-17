from typing import List
import numpy as np

from layer import Layer


class Optimizer:
    """
    Optimizer class.
    """

    def __init__(self):
        pass

    def update(self, layer):
        """
        Update weights of layer.
        """
        raise NotImplementedError


class Adam(Optimizer):
    """
    Adam optimizer.
    """

    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-07):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, layers: List[Layer], iter_num: int):
        for layer in layers:
            layer.m_weights = (self.beta1/(1 - self.beta1**iter_num)) * \
                layer.m_weights + \
                ((1-self.beta1)/(1 - self.beta1**iter_num)) * layer.weights_grad

            layer.v_weights = (self.beta2/(1 - self.beta2**iter_num)) * \
                layer.v_weights + \
                ((1-self.beta2)/(1 - self.beta2**iter_num)) * \
                layer.weights_grad * layer.weights_grad

            layer.weights -= self.learning_rate * \
                (layer.m_weights/(layer.v_weights**0.5 + self.epsilon))

            layer.m_bias = (self.beta1/(1 - self.beta1**iter_num)) * \
                layer.m_bias + \
                ((1-self.beta1)/(1 - self.beta1**iter_num)) * layer.bias_grad

            layer.v_bias = (self.beta2/(1 - self.beta2**iter_num)) * \
                layer.v_bias + \
                ((1-self.beta2)/(1 - self.beta2**iter_num)) * \
                layer.bias_grad * layer.bias_grad

            layer.bias -= self.learning_rate * \
                (layer.m_bias/(layer.v_bias**0.5 + self.epsilon))
