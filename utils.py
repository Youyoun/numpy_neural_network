import numpy as np

def shuffle(x, y):
    randomize = np.arange(x.shape[0])
    np.random.shuffle(randomize)
    return x[randomize, :], y[randomize, :]

def confusion_matrix(mask_next, mask_prev):
    rows = []
    for i in mask_prev:
        col = []
        for j in mask_next:
            col.append(j*i)
        rows.append(col)
    return np.array(rows)