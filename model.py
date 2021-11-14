from typing import List
import numpy as np
from layer import Layer
from loss import BinaryCrossEntropy
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

    def add(self, layer: Layer):
        if layer.input_shape is None:
            layer.set_input_shape(self.layers[-1].output_shape)
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        if loss == 'binary_crossentropy':
            self.loss = BinaryCrossEntropy()
        if optimizer == 'adam':
            self.optimizer = Adam()

    def predict(self, X):
        total_predict = []
        for value in X:
            m_val = value.copy()
            for layer in self.layers:
                m_val = layer.forward(m_val)
            total_predict.append(m_val[0])
        return np.array(total_predict)

    def fit(self, X, y, epochs, batch_size, validation_data):

        for epoch in range(epochs):
            loss_values = []
            for i in range(0, len(X), batch_size):
                # Forward pass
                iter_value = np.array([X[i]])
                for layer in self.layers:
                    iter_value = layer.forward(iter_value)

                loss_value = self.loss.loss(y[i], iter_value)
                loss_values.append(loss_value)
                # Backward pass
                reversed_layers = self.layers[::-1]

                grad_loss = self.loss.grad(y[i], iter_value)
                print(grad_loss)
                for layer in reversed_layers:
                    grad_loss = layer.backward(grad_loss)

            loss_values = np.array(loss_values).mean()
            print(f'Epoch {epoch}: loss: {loss_values}')

    def accuracy(self, y, y_pred):
        return np.mean(np.equal(y, np.round(y_pred)))

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        # print(y.shape)
        # loss = self.loss(y, y_pred)
        accuracy = self.accuracy(y, y_pred)
        return accuracy
