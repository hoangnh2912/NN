import numpy as np
from layer import Layer, Dense, ReLU, InputLayer
from preprocess import X_val, X_train, Y_train, Y_val
from model import Sequential

network = Sequential()
network.add(InputLayer(7))
network.add(Dense(8, activation='relu', name='h_layer_1'))
network.add(Dense(8, activation='relu', name='h_layer_2'))
network.add(Dense(8, activation='relu', name='h_layer_3'))
network.add(Dense(1, activation='sigmoid', name='output_layer'))

# out = network.predict(np.array([[1, 2, 3, 4, 5, 6, 7]]))

network.compile(loss='binary_crossentropy', optimizer='adam')
network.fit(X_train, Y_train, epochs=100, batch_size=16,
            validation_data=(X_val, Y_val))
