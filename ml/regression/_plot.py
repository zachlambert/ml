import numpy as np
import matplotlib.pyplot as plt
from ml.common.plot import (
    plot_contours
)


def plot_data_one_feature(ax, manager, index):
    ax.scatter(
        manager.X[index, :],
        manager.y,
        marker="x"
    )
    ax.set_xlabel(manager.feature_names[index])
    ax.set_ylabel(manager.output_name)


def plot_data_two_features(ax, manager, index1, index2):
    plot_contours(
        ax,
        manager.X[index1, :],
        manager.X[index2, :],
        manager.y
    )
    ax.set_xlabel(manager.feature_names[index1])
    ax.set_ylabel(manager.feature_names[index2])
    ax.set_title(manager.output_name, fontsize="12")


def plot_results_one_feature(ax, manager, index):
    ax.scatter(
        manager.X_test[index, :],
        manager.y_test,
        marker="x", label="actual"
    )
    y_pred, _ = manager.predict(manager.X_test)
    ax.scatter(
        manager.X_test[index, :],
        y_pred,
        marker="x", label="prediction"
    )
    ax.set_xlabel(manager.feature_names[index])
    ax.set_ylabel(manager.output_name)
    ax.legend()


def plot_results_two_features(ax1, ax2, ax3, manager, index1, index2):
    plot_contours(
        ax1,
        manager.X_test[index1, :],
        manager.X_test[index2, :],
        manager.y_test
    )
    ax1.set_xlabel(manager.feature_names[index1])
    ax1.set_ylabel(manager.feature_names[index2])
    ax1.set_title(
        "Actual {}".format(manager.output_name),
        fontsize="12"
    )

    y_pred, y_var = manager.predict(manager.X_test)

    plot_contours(
        ax2,
        manager.X_test[index1, :],
        manager.X_test[index2, :],
        y_pred
    )
    ax2.set_xlabel(manager.feature_names[index1])
    ax2.set_ylabel(manager.feature_names[index2])
    ax2.set_title(
        "Predicted {}".format(manager.output_name),
        fontsize="12"
    )

    plot_contours(
        ax3,
        manager.X_test[index1, :],
        manager.X_test[index2, :],
        np.sqrt(y_var.reshape((y_var.size, 1))),
        cmap="cool"
    )
    ax3.set_xlabel(manager.feature_names[index1])
    ax3.set_ylabel(manager.feature_names[index2])
    ax3.set_title(
        "Prediction standard deviation",
        fontsize="12"
    )
