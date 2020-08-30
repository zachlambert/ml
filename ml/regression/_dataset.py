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

    def generate_y(self, transform_X, snr=10):
        X_tilde = transform_X(self.X)

        w = np.random.normal(0, 1, X_tilde.shape[0])
        w = w.reshape((w.size, 1))
        self.y = np.matmul(np.transpose(X_tilde), w)

        # Add on noise, with a given SNR
        # SNR = signal power / noise power
        #     = E[y^2] / E[n^2]
        y_power = np.mean(self.y ** 2)
        noise_var = y_power / snr
        self.y[:, 0] += np.random.normal(
            0, np.sqrt(noise_var), self.y.shape[0]
        )
        return w, noise_var

    def generate_y_linear(self, snr=10):
        return self.generate_y(make_transform_X_linear(), snr)

    def generate_y_polynomial(self, order=3, snr=10):
        return self.generate_y(make_transform_X_polynomial(order), snr)
