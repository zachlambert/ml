import numpy as np


def _gradient_descent(gradient_T_func, norm_func, theta_shape,
                      learning_rate=0.1, norm_threshold=1e-3):
    theta = np.zeros(theta_shape)
    gradient_T = gradient_T_func(theta)
    while norm_func(gradient_T) >= norm_threshold:
        theta -= learning_rate * gradient_T
        gradient_T = gradient_T_func(theta)
    return theta


def gradient_descent_vector(gradient_T_func, num_params, learning_rate=0.1,
                            norm_threshold=1e-3):
    return _gradient_descent(
        gradient_T_func,
        lambda grad: np.abs(grad),  # Default is L2 norm for vectors
        (num_params, 1),
        learning_rate,
        norm_threshold
    )


def gradient_descent_matrix(gradient_T_func, theta_shape, learning_rate=0.1,
                            norm_threshold=1e-3):
    return _gradient_descent(
        gradient_T_func,
        lambda grad: np.abs(grad),  # Default is Frobenius norm for matrices
        theta_shape,
        learning_rate,
        norm_threshold
    )
