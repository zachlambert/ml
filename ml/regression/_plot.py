import matplotlib.pyplot as plt
from ml.common.plot import (
    plot_contours
)


def plot_data_one_feature(ax, manager, index):
    plt.scatter(
        manager.X[:, index],
        manager.y,
        marker="x"
    )
    plt.xlabel(manager.feature_names[index])
    plt.ylabel(manager.output_name)


def plot_data_two_features(ax, manager, index1, index2):
    plot_contours(
        ax,
        manager.X[:, index1],
        manager.X[:, index2],
        manager.y
    )
    plt.xlabel(manager.feature_names[index1])
    plt.ylabel(manager.feature_names[index2])
    plt.title(manager.output_name, fontsize="14")


def plot_result_one_feature(ax, manager, index):
    plt.scatter(
        manager.X_test[:, index],
        manager.y_test,
        marker="x", label="actual"
    )
    y_pred, _ = manager.predict(manager.X_test)
    plt.scatter(
        manager.X_test[:, index],
        y_pred,
        marker="x", label="prediction"
    )
    plt.xlabel(manager.feature_names[index])
    plt.ylabel(manager.output_name)
    plt.legend()
