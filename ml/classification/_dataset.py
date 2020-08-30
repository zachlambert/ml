import numpy as np
from ml.common.transform import (
    make_transform_X_linear,
    make_transform_X_polynomial,
)


class ArtificialDataset:
    def __init__(self):
        self.X = None
        self.y = None

    def generate_X_gaussian(self, D, N, mean=None, cov=None):
        mean = mean if mean is not None else np.zeros(D)
        cov = cov if cov is not None else np.eye(D)
        self.X = np.transpose(np.random.multivariate_normal(mean, cov, N))

    def generate_X_uniform(self, D, N, low=None, high=None):
        low = low if low is not None else np.zeros(D)
        high = high if high is not None else np.ones(D)
        self.X = np.transpose(np.random.uniform(low, high, (N, D)))

    def generate_y(self, transform_X):
        X_tilde = transform_X(self.X)

        W = np.zeros(self.D, self.C)
        for c in range(self.C):
            W[:, c] = np.random.normal(0, 1, X_tilde.shape[0])

        P = np.exp(np.matmul(np.transpose(W), X_tilde))
        P = np.divide(P, np.sum(P, axis=0))

        # Get the cumulative distribution
        F = np.cumsum(P, axis=0)

        # Want to generate a set of labels y, according to the pmf of y
        # Therefore, generate a uniform random variable for eac data point
        # and lookup on the cumulative distribution to get the label to use
        samples = np.random.uniform(0, 1, self.N)

        # np.less(fy, samples.reshape(...)) gives true for each point
        # on the cmf below the sample value. The label to select is equal
        # to the index of the first cdf value above the sample, which
        # is equivalent to the total number of cdf values below the sample
        self.y = np.sum(np.less(F, samples), axis=0)
        return W

    def generate_y_linear(self):
        return self.generate_y(make_transform_X_linear())

    def generate_y_polynomial(self, order=3):
        return self.generate_y(make_transform_X_polynomial(order))
