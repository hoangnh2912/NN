import numpy as np
from layer import Layer, Dense
from preprocess import X_val, X_train, Y_train, Y_val
from model import Sequential


network = Sequential()
network.add(Dense(8, activation='relu', name='layer_1', input_shape=(1, 7)))
network.add(Dense(8, activation='relu', name='layer_2'))
network.add(Dense(8, activation='relu', name='layer_3'))
network.add(Dense(1, activation='sigmoid', name='output_layer'))


network.compile(loss='binary_crossentropy', optimizer='adam')
network.fit(X_train, Y_train, epochs=9999, batch_size=16,
            validation_data=(X_val, Y_val))
