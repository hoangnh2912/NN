import numpy as np


class Loss:
    def loss(self, y, y_pred):
        raise NotImplementedError

    def grad(self, y, y_pred):
        raise NotImplementedError


class BinaryCrossEntropy(Loss):
    def loss(self, y, y_pred):
        return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def grad(self, y, y_pred):
        return -(y / y_pred - (1 - y) / (1 - y_pred))
