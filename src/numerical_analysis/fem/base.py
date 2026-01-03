from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix


class FemBase(metaclass=ABCMeta):
    def __init__(self, n_nodes: int, dim: int) -> None:
        self.laplacian_matrix = lil_matrix((n_nodes, n_nodes))
        self.differential_matrices = [lil_matrix((n_nodes, n_nodes))] * dim
        self.term_matrix = lil_matrix((n_nodes, n_nodes))

    def laplacian(self, vector: NDArray) -> NDArray:
        return np.array(self.laplacian_matrix.dot(vector))

    def differential(self, vector: NDArray) -> tuple[NDArray, ...]:
        return tuple(np.array(mat.dot(vector)) for mat in self.differential_matrices)

    def term(self, vector: NDArray) -> NDArray:
        return np.array(self.term_matrix.dot(vector))

    @abstractmethod
    def implement_dirichlet(self, coefficient: lil_matrix, rhs: NDArray, values: NDArray) -> None: ...

    @abstractmethod
    def implement_neumann(self, rhs: NDArray, values: NDArray) -> None: ...


def update_global_matrix(matrix: lil_matrix, indexes: tuple[int, ...], local_matrix: NDArray) -> None:
    row_idx = np.vstack((indexes,) * len(indexes)).T.flatten()
    col_idx = np.hstack((indexes,) * len(indexes))
    matrix[row_idx, col_idx] += local_matrix


def update_global_matrices(matrices: list[lil_matrix], indexes: tuple[int, ...], local_matrices: tuple[NDArray, ...]) -> None:
    row_idx = np.vstack((indexes,) * len(indexes)).T.flatten()
    col_idx = np.hstack((indexes,) * len(indexes))
    for i, local_matrix in enumerate(local_matrices):
        matrices[i][row_idx, col_idx] += local_matrix


def implement_dirichlet(coefficient: lil_matrix, rhs: NDArray, values: NDArray, indexes: NDArray) -> None:
    d = np.zeros_like(rhs)
    d[indexes] = values[indexes]
    rhs -= coefficient.dot(d)
    rhs[indexes] = values[indexes]
    coefficient[indexes, :] = 0.0
    coefficient[:, indexes] = 0.0
    coefficient[indexes, indexes] = 1.0
