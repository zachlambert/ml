import numpy as np


def make_transform_X_linear():
    def transform_X(X):
        return np.concatenate([np.ones((1, X.shape[1])), X])
    return transform_X


def make_transform_X_polynomial(order):
    def transform_X(X):
        return np.concatenate(
            [np.ones((1, X.shape[1]))]
            + [
                np.reshape(X[i, :] ** k, (1, X.shape[1]))
                for k in range(1, order + 1)
                for i in range(X.shape[0])
            ]
        )
    return transform_X


def make_transform_X_radial_basis(centres, widths):
    c = np.transpose(np.array(centres))
    w = np.array(widths)
    w = w.reshape(w.size, 1)
    assert c.shape[1] == w.size

    def transform_X(X):
        assert c.shape[0] == X.shape[0]
        return np.concatenate(
            [np.ones((1, X.shape[1]))]
            + [
                np.exp(
                    -0.5
                    * np.sum((X - c[:, k]) ** 2, axis=0)
                    / widths[k]
                )
                for k in range(w.size)
            ]
        )
    return transform_X
