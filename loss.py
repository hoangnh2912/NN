import numpy as np


def binary_crossentropy(y_true, y_pred):
    '''
    y_true: ground truth
    y_pred: prediction
    '''
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
