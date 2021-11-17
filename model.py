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

    def loss_(self, y, y_pred):
        return self.loss.loss(y, y_pred).mean()

    def fit(self, X, y, epochs, batch_size, validation_data):
        iter_num = 0
        X_val, Y_val = validation_data
        for epoch in range(epochs):

            for i in range(0, len(X), batch_size):
                # Forward pass
                iter_value = np.array([X[i]])
                for layer in self.layers:
                    iter_value = layer.forward(iter_value)

                # Backward pass
                reversed_layers = self.layers[::-1]

                grad_loss = self.loss.grad(y[i], iter_value)
                for layer in reversed_layers:
                    grad_loss = layer.backward(grad_loss)
                # update weights
                iter_num += 40
                self.optimizer.update(reversed_layers, iter_num)
            # loss_trains = np.array(loss_trains).mean()
            acc, loss = self.evaluate(X, y)

            acc_val, loss_val = self.evaluate(X_val, Y_val)
            print(
                f'Epoch {epoch}: loss: {np.round(loss,3)}, acc: {np.round(acc,3)}, acc_val: {np.round(acc_val,3)}, loss_val: {np.round(loss_val,3)}')

    def accuracy(self, y, y_pred):
        return np.mean(np.equal(y, np.round(y_pred)))

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        loss = self.loss_(y, y_pred)
        accuracy = self.accuracy(y, y_pred)
        return accuracy, loss
