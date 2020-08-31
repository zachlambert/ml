import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


def plot_data_one_feature(ax, manager, index):
    for c in manager.C:
        X_c = manager.X[manager.y == c]
        plt.scatter(
            X_c[index, :],
            np.full(X_c.shape[0], c),
            label="{}".format(manager.class_names[c]),
            marker="x"
        )
    plt.xlabel(manager.feature_names[index])
    plt.ylabel(manager.output_name)
    plt.legend()


def plot_data_two_features(ax, manager, index1, index2):
    for c in manager.C:
        X_c = manager.X[manager.y == c]
        plt.scatter(
            X_c[index1, :],
            X_c[index2, :],
            label="{}".format(manager.class_names[c])
        )
    plt.xlabel(manager.feature_names[index1])
    plt.ylabel(manager.feature_names[index2])
    plt.title(manager.output_name, fontsize="14")
    plt.legend()


def plot_result_one_feature(ax, manager, index):
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for c in manager.C:
        color = next(colors)
        actual_X_c = manager.X_test[manager.y_test == c]
        plt.scatter(
            actual_X_c[index, :],
            np.full(actual_X_c.shape[0], c),
            label="Actual {}".format(manager.class_names[c]),
            marker="x",
            color=color
        )
        y_pred, _ = manager.predict(manager.X_test)
        pred_X_c = manager.X_test[y_pred == c]
        plt.scatter(
            pred_X_c[index, :],
            np.full(pred_X_c.shape[0], c),
            label="Predicted {}".format(manager.class_names[c]),
            marker="o",
            color=color
        )
    plt.xlabel(manager.feature_names[index])
    plt.ylabel(manager.output_name)
    plt.legend()


def plot_result_two_features(ax, manager, index1, index2):
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for c in manager.C:
        color = next(colors)
        actual_X_c = manager.X_test[manager.y_test == c]
        plt.scatter(
            actual_X_c[index1, :],
            actual_X_c[index2, :],
            label="Actual {}".format(manager.class_names[c]),
            marker="x",
            color=color
        )
        y_pred, _ = manager.predict(manager.X_test)
        pred_X_c = manager.X_test[y_pred == c]
        plt.scatter(
            pred_X_c[index1, :],
            pred_X_c[index2, :],
            label="Predicted {}".format(manager.class_names[c]),
            marker="o",
            color=color
        )
    plt.xlabel(manager.feature_names[index1])
    plt.xlabel(manager.feature_names[index2])
    plt.title(manager.output_name, fontsize=14)
    plt.legend()
