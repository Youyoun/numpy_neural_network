import numpy as np

def shuffle(x, y):
    randomize = np.arange(x.shape[0])
    np.random.shuffle(randomize)
    return x[randomize, :], y[randomize, :]
