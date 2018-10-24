def gen_data(p, data, batch_size=1):
    # generate data for the model
    # y in train data is a matrix (batch_size, seq_length)
    # y in test data is an array
    x = data['x'][p:p + batch_size]
    y = data['y'][p:p + batch_size]
    batch_data = {'x': x, 'y': y}
    if data.has_key('t'):
        batch_data['t'] = data['t'][p:p + batch_size]


import numpy as np
import theano.tensor as T
n = np.array([[1, 2], [3, 4]])
T.matrix(n)