import numpy as np


def describe_data(manager):
    string = ""
    string += "{} features\n".format(manager.D)
    string += "Feature names: {}\n".format(manager.feature_names)
    string += "Output name: {}\n".format(manager.output_name)
    string += "{} data points\n".format(manager.N)
    string += "y mean = {}, var = {}".format(
        np.mean(manager.y), np.var(manager.y)
    )
    return string


def describe_results(manager):
    string = ""
    for i, model in enumerate(manager.models):
        string += "Model {}".format(i)
        if i == manager.models_best:
            string += " [BEST]\n"
        else:
            string += "\n"
        string += str(model) + "\n"
        string += "MSE: {}\n".format(manager.models_mse[i])
    return string[:-1]  # Omit final \n
