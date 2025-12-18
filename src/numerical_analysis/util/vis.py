from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plot1d(filename: Path | str, x: ArrayLike, numerical_: ArrayLike, analytical_: ArrayLike) -> None:
    """_summary_

    Args:
        filename (Path | str): name of output image file.
        x (ArrayLike): array of positions of nodes.
        numerical_ (ArrayLike): array of numerical solution.
        analytical_ (ArrayLike): array of analytical solution.
    """
    numerical = np.array(numerical_).flatten()
    analytical = np.array(analytical_).flatten()
    error = np.abs(numerical - analytical)
    fig = plt.figure(figsize=(7.5, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, numerical, label="Analytical", color="red")
    ax1.scatter(x, analytical, label="Numerical", color="blue", s=16)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x)")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, error)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Absolute Error, Err(x)")
    fig.tight_layout()
    fig.savefig(filename)


def plot2d(
    filename: Path | str, x_: ArrayLike, y_: ArrayLike, numerical_: ArrayLike, analytical_: ArrayLike, n_x: int, n_y: int
) -> None:
    """_summary_

    Args:
        filename (Path | str): name of output image file.
        x (ArrayLike): array of positions of nodes.
        y (ArrayLike): array of positions of nodes.
        numerical_ (ArrayLike): array of numerical solution.
        analytical_ (ArrayLike): array of analytical solution.
    """
    x = np.reshape(x_, (n_y, n_x))
    y = np.reshape(y_, (n_y, n_x))
    numerical = np.reshape(numerical_, (n_y, n_x))
    analytical = np.reshape(analytical_, (n_y, n_x))
    error = np.abs(numerical - analytical)
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(1, 3, 1)
    cmap1 = ax1.pcolormesh(x, y, analytical)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Analytical")
    ax2 = fig.add_subplot(1, 3, 2)
    cmap2 = ax2.pcolormesh(x, y, numerical)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Numerical")
    ax3 = fig.add_subplot(1, 3, 3)
    cmap3 = ax3.pcolormesh(x, y, error)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("Absolute Error")
    fig.colorbar(cmap1, ax=ax1)
    fig.colorbar(cmap2, ax=ax2)
    fig.colorbar(cmap3, ax=ax3)
    fig.tight_layout()
    fig.savefig(filename)
