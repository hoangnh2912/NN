from typing import List
import numpy as np
from layer import Layer
from loss import binary_crossentropy
from optimizer import Adam


class Model:
    def __init__(self):
        pass

    def predict(self, X):
        pass

    def fit(self, X, y, epochs, batch_size):
        pass

    def evaluate(self, X, y):
        pass

    def compile(self,  loss, optimizer):
        pass


class Sequential(Model):
    def __init__(self):
        self.layers: List[Layer] = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        if len(self.layers) > 0:
            layer.set_input_shape(self.layers[-1].output_shape)
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        if loss == 'binary_crossentropy':
            self.loss = binary_crossentropy
        if optimizer == 'adam':
            self.optimizer = Adam()

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def fit(self, X, y, epochs, batch_size, validation_data):

        list_layer_trainable = self.layers[1:]
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                # Forward pass
                m_X = np.array([X[i]])
                for layer in list_layer_trainable:
                    m_X = layer.forward(m_X)

                # Backward pass
                gradients = [np.zeros(layer.weights.shape)
                             for layer in reversed(list_layer_trainable)]
                for gradient, layer in zip(gradients, reversed(list_layer_trainable)):
                    gradient = layer.backward(gradient)

                # Update weights
                for gradient, layer in zip(reversed(gradients), list_layer_trainable):
                    layer.update(layer, self.optimizer, gradient)

    def accuracy(y, y_pred):
        return np.mean(np.equal(y, np.round(y_pred)))

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        # loss = self.loss(y, y_pred)
        accuracy = self.accuracy(y, y_pred)
        return accuracy
