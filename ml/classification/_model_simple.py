import numpy as np
from ml.common.transform import (
    make_transform_X_linear,
    make_transform_X_polynomial,
    make_transform_X_radial_basis,
)
from ml.common.optimisation import (
    gradient_descent_matrix
)


def _P_matrix(X_tilde, W):
    P = np.exp(np.matmul(np.transpose(W), X_tilde))
    P = np.divide(P, np.sum(P, axis=0))
    return P


def _Y_matrix(y, C):
    Y = np.zeros((C, y.size))
    for i, c in enumerate(y):
        Y[c, i] = 1
    return Y


class ModelSimple:
    def __init__(self, transform="linear", learning_rate=0.1, var_w=0.1,
                 order=3, basis_width=1):
        self.transform = transform
        self.learning_rate = learning_rate
        self.var_w = var_w
        self.order = order
        self.basis_width = basis_width

        if self.transform == "linear":
            self._transform_X = make_transform_X_linear()
        elif self.transform == "polynomial":
            self._transform_X = make_transform_X_polynomial(order)
        elif self.transform == "gaussian_radial_basis":
            pass
        else:
            raise TypeError("Invalid transform name passed.")

        self.W = None

    def fit(self, C, X, y):
        self.C = C
        if self.transform == "gaussian_radial_basis":
            self._transform_X = make_transform_X_radial_basis(
                X, np.full(X.shape[1], self.basis_width)
            )
        X_tilde = self._transform_X(X)
        self.D_tilde = X_tilde.shape[0]
        Y = _Y_matrix(y, self.C)

        def cost_gradient_T(W):
            grad = np.matmul(X_tilde,
                             np.transpose(_P_matrix(X_tilde, W) - Y)) \
                + (1/self.var_w**2)*W
            return grad

        self.W = gradient_descent_matrix(
            cost_gradient_T, (self.D_tilde, self.C), self.learning_rate
        )

    def predict(self, X):
        X_tilde = self._transform_X(X)
        P_matrix_pred = _P_matrix(X_tilde, self.W)
        y_pred = np.argmax(P_matrix_pred, axis=0)
        y_prob = P_matrix_pred[y_pred]
        return y_pred, y_prob

    def __repr__(self):
        return "Simple, transform={}, learning_rate={}, var_w={}, weights={}" \
            .format(self.transform, self.learning_rate, self.var_w, self.W)
