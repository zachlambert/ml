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


def make_transform_X_radial_basis(C, widths):
    w = np.array(widths)
    w = w.reshape(w.size, 1)
    assert C.shape[1] == w.size

    def transform_X(X):
        assert C.shape[0] == X.shape[0]
        return np.concatenate(
            [np.ones((1, X.shape[1]))]
            + [
                np.exp(
                    -0.5
                    * np.sum((X - C[:, k].reshape((C.shape[0], 1))) ** 2,
                             axis=0)
                    / widths[k]
                ).reshape((1, X.shape[1]))
                for k in range(w.size)
            ], axis=0
        )
    return transform_X
