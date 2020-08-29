import numpy as np


def make_transform_X_linear():
    def transform_X(X):
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    return transform_X


def make_transform_X_polynomial(order):
    def transform_X(X):
        return np.concatenate(
            [np.ones((X.shape[0], 1))]
            + [
                np.reshape(X[:, i] ** k, (X.shape[0], 1))
                for k in range(1, order + 1)
                for i in range(X.shape[1])
            ],
            axis=1,
        )

    return transform_X


def make_transform_X_radial_basis(centres, widths):
    c = np.array(centres)
    if len(c.shape) == 1:
        c = np.reshape(c, (c.size, 1))
    w = np.array(widths)
    w = w.reshape(w.size, 1)
    assert c.shape[0] == w.size

    def transform_X(X):
        assert c.shape[1] == X.shape[1]
        return np.concatenate(
            [np.ones((X.shape[0], 1))]
            + [
                np.exp(
                    -0.5
                    * np.sum((X - c[k]) ** 2, axis=1).reshape(X.shape[0], 1)
                    / widths[k]
                )
                for k in range(w.size)
            ],
            axis=1,
        )

    return transform_X
