import numpy as np


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
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def fit(self, X, y, epochs, batch_size, validation_data):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                # Forward pass
                for layer in self.layers:
                    X = layer.forward(X)

                # Backward pass
                gradients = [np.zeros(layer.weights.shape)
                             for layer in self.layers]
                for layer in reversed(self.layers):
                    gradients = layer.backward(gradients)

                # Update weights
                for layer in self.layers:
                    layer.update(self.optimizer)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
