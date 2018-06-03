import numpy as np


def shuffle(x, y):
    """
    Shuffle arrays simultaneously (to keep order)
    :param x: Feature vector (n_samples, n_features)
    :param y: Labels
    :return: Feature vectors and labels shuffled
    """
    randomize = np.arange(x.shape[0])
    np.random.shuffle(randomize)
    return x[randomize, :], y[randomize, :]


def confusion_matrix(mask_next, mask_prev):
    """
    Compute confusion matrix from " arrays
    :param mask_next: Next Mask
    :param mask_prev: Previous mask
    :return: Matrix of same shape as (mask_next, mask_prev) containing 1 for indexes that have been kept and 0 for those removed
    """
    rows = []
    for i in mask_prev:
        col = []
        for j in mask_next:
            col.append(j * i)
        rows.append(col)
    return np.array(rows)
