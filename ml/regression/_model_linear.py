import numpy as np
from ml.common.transform import (
    make_transform_X_linear,
    make_transform_X_polynomial,
    make_transform_X_radial_basis,
)
from ml.common.maths import (
    dot_products
)


class ModelLinear:
    def __init__(self, transform="linear", var_e=1, var_ratio=0.1,
                 order=3, basis_width=1):
        self.transform = transform
        self.var_e = var_e
        self.var_ratio = var_ratio
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

        self.w_mean = None
        self.w_covar = None

    def fit(self, X, y):
        if self.transform == "gaussian_radial_basis":
            self._transform_X = make_transform_X_radial_basis(
                X, np.full(X.shape[0], self.basis_width)
            )
        X_tilde = self._transform_X(X)
        A = np.matmul(np.transpose(X_tilde), X_tilde) \
            + self.var_ratio*np.eye(X_tilde.shape[1])
        A_inv = np.linalg.inv(A)
        # If computing the full inverse is numerically unstable, consider
        # trying a low rank approximation of the inverse
        # A_inv = approximate_inverse(A)
        self.w_mean = np.matmul(np.matmul(A_inv, np.transpose(X_tilde)), y)
        self.w_covar = self.var_e * A

    def predict(self, X):
        X_tilde = self._transform_X(X)
        y_pred = np.matmul(X_tilde, self.w_mean)
        y_var = dot_products(X_tilde, np.matmul(X_tilde, self.w_covar))
        return y_pred, y_var

    def __repr__(self):
        return "Linear, transform={}, var_e={}, var_ratio={}, weights={}" \
            .format(self.transform, self.var_e, self.var_ratio, self.w_mean)
