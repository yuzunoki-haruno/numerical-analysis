import numpy as np
from numpy.typing import ArrayLike


def absolute_error(numerical_: ArrayLike, analytical_: ArrayLike) -> float:
    """Calculate the absolute error.

    Args:
        numerical_ (ArrayLike): array of numerical solution.
        analytical_ (ArrayLike): array of analytical solution.

    Returns:
        float: _description_
    """
    numerical = np.array(numerical_).flatten()
    analytical = np.array(analytical_).flatten()
    error = np.abs(numerical - analytical).max()
    return float(error)


def relative_error(numerical_: ArrayLike, analytical_: ArrayLike) -> float:
    """Calculate the relative error.

    Args:
        numerical_ (ArrayLike): array of numerical solution.
        analytical_ (ArrayLike): array of analytical solution.

    Returns:
        float: _description_
    """
    numerical = np.array(numerical_).flatten()
    analytical = np.array(analytical_).flatten()
    rerror = np.abs(numerical - analytical).max()
    rerror /= np.abs(analytical).max()
    return float(rerror)
