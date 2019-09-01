import numpy as np


def dropout(X, keep_prob):
    assert 0 < keep_prob < 1
    if keep_prob == 0:
        return np.zeros(X.shape)
    mask = np.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob