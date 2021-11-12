from layer import Layer, Dense, ReLU
from loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
import numpy as np
from preprocess import X_val, X_train, Y_train, Y_val
from model import Sequential
network = Sequential()
network.add(Dense(8, activation='relu'))
network.add(Dense(8, activation='relu'))
network.add(Dense(8, activation='relu'))
network.add(Dense(1, activation='sigmoid'))


network.compile(loss='binary_crossentropy', optimizer='adam')
network.fit(X_train, Y_train, epochs=100, batch_size=16,
            validation_data=(X_val, Y_val))
