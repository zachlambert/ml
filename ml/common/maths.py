import numpy as np


def approximate_inverse(A, eigenvalue_threshold=0.1):
    w, v = np.linalg.eig(A)
    max_w = np.max(w)
    A_inv = np.zeros(A.shape)
    for i, wi in enumerate(w):
        if wi > eigenvalue_threshold * max_w:
            A_inv += (1/wi) * np.outer(v[i], v[i])
    return A_inv


dot_products = np.vectorize(lambda a, b: np.dot(a, b), signature="(m),(n)->()")
