import numpy as np


class Optimizer:
    """
    Optimizer class.
    """

    def __init__(self):
        pass

    def update(self, layer, gradients):
        """
        Update weights of layer.
        """
        raise NotImplementedError


class Adam(Optimizer):
    """
    Adam optimizer.
    """

    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-07):
        """
        Initialize Adam optimizer.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, layer, gradient):
        """
        Update weights of layer.
        """

        if self.m is None:
            # self.m = [0] * len(layer.weights.T)
            self.m = np.random.rand(len(layer.weights))
            self.v = np.random.rand(len(layer.weights))
            # self.v = [0] * len(layer.weights.T)
        print(self.m)
        for idx, grad in enumerate(gradient):
            pass
            # print(gradient.shape)
            # print(self.m[0])
            # print("="*10)
            # for i in range(len(layer.weights)):
            #     self.m[i] = self.beta1 * self.m[i] + \
            #         (1 - self.beta1) * gradient[i]
            #     self.v[i] = self.beta2 * self.v[i] + \
            #         (1 - self.beta2) * gradient[i]**2

            # m_hat = [self.m[i] / (1 - self.beta1)
            #          for i in range(len(layer.weights))]
            # v_hat = [self.v[i] / (1 - self.beta2)
            #          for i in range(len(layer.weights))]
            # layer.weights = np.array([layer.weights[i] - self.learning_rate * m_hat[i] /
            #                           (v_hat[i]**0.5 + self.epsilon) for i in range(len(layer.weights))])
