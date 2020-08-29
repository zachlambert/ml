import numpy as np


def gradient_descent(gradient_func, num_params, learning_rate=0.1,
                     gradient_threshold=0.1):
    params = np.zeros(num_params)
    gradient = gradient_func(params)
    while np.abs(gradient) >= gradient_threshold:
        params -= learning_rate * gradient
        gradient = gradient_func(params)
    return params
