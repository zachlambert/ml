import numpy as np


def describe_data(manager):
    string = ""
    string += "{} features\n".format(manager.D)
    string += "Feature names: {}\n".format(manager.feature_names)
    string += "{} classes\n".format(manager.C)
    string += "Class names: {}\n".format(manager.class_names)
    string += "Output name: {}\n".format(manager.output_name)
    string += "{} data points\n".format(manager.N)
    for c in range(1, manager.C + 1):
        string += "\tNum {} = {}\n".format(c, np.sum(manager.y == c))
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
        string += "Accuracy: {}%\n".format(manager.models_accuracy[i]*100)
    return string[:-1]  # Omit final \n
