import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_contours(ax, x, y, z, n_levels=10, cmap="viridis"):
    z = z[:, 0]
    X = np.linspace(np.min(x), np.max(x), 1000)
    Y = np.linspace(np.min(y), np.max(y), 1000)
    X, Y = np.meshgrid(X, Y)

    points = np.concatenate(
        [x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)], axis=1
    )
    Z = griddata(points, z, (X, Y), "linear")
    mappable = ax.contourf(X, Y, Z, levels=n_levels, cmap=cmap)
    plt.colorbar(mappable, ax=ax)
