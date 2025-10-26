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
