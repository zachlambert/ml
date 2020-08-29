import numpy as np
from ml.common.transform import (
    make_transform_X_linear,
    make_transform_X_polynomial,
    make_transform_X_radial_basis,
)
from ml.common.maths import (
    dot_products
)
from ml.common.optimisation import (
    gradient_descent
)


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

        self.params = None

    def fit(self, X, y):
        if self.transform == "gaussian_radial_basis":
            self._transform_X = make_transform_X_radial_basis(
                X, np.full(X.shape[0], self.basis_width)
            )
        X_tilde = self._transform_X(X)

        def cost_gradient(params):
            pass

        return gradient_descent(
            cost_gradient, self.D*self.C, self.learning_rate
        )

    def predict(self, X):
        X_tilde = self._transform_X(X)
        return y_pred, y_prob

    def __repr__(self):
        return "Linear, transform={}, var_e={}, var_ratio={}, weights={}" \
            .format(self.transform, self.var_e, self.var_ratio, self.w_mean)
