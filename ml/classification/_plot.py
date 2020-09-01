import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


def plot_data_two_features(ax, manager, index1, index2):
    for c in range(manager.C):
        X_c = manager.X[:, manager.y == c]
        plt.scatter(
            X_c[index1, :],
            X_c[index2, :],
            label="{}".format(manager.class_names[c]),
            marker="x"
        )
    plt.xlabel(manager.feature_names[index1])
    plt.ylabel(manager.feature_names[index2])
    plt.title(manager.output_name, fontsize="14")
    plt.legend()


def plot_results_two_features(ax, manager, index1, index2):
    for c in range(manager.C):
        actual_X_c = manager.X_test[:, manager.y_test == c]
        plt.scatter(
            actual_X_c[index1, :],
            actual_X_c[index2, :],
            label=manager.class_names[c],
            marker="x"
        )
    plt.xlabel(manager.feature_names[index1])
    plt.xlabel(manager.feature_names[index2])
    plt.title("Actual {}".format(manager.output_name), fontsize=12)
    plt.legend()
    plt.show()

    y_pred, _ = manager.predict(manager.X_test)
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for c in range(manager.C):
        color = next(colors)
        pred_X_c = manager.X_test[
            :, np.logical_and(y_pred == c, manager.y_test == c)]
        plt.scatter(
            pred_X_c[index1, :],
            pred_X_c[index2, :],
            label="Correct {}".format(manager.class_names[c]),
            color=color,
            marker="x"
        )
        pred_X_c = manager.X_test[
            :, np.logical_and(y_pred == c, manager.y_test != c)]
        plt.scatter(
            pred_X_c[index1, :],
            pred_X_c[index2, :],
            label="Incorrect {}".format(manager.class_names[c]),
            color=color,
            marker="o"
        )

    plt.xlabel(manager.feature_names[index1])
    plt.xlabel(manager.feature_names[index2])
    plt.title("Predicted {}".format(manager.output_name), fontsize=12)
    plt.legend()
    plt.show()
