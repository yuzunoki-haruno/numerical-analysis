import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from numerical_analysis.discretization import Condition, PolygonMesh


def _area(x: NDArray, y: NDArray) -> np.float64:
    area = np.abs(_det(x, y)) / 2
    return np.float64(area)


def _det(x: NDArray, y: NDArray) -> np.float64:
    x1, x2, x3 = x
    y1, y2, y3 = y
    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    return np.float64(det)


def LAPLACIAN_MATRIX_3X3(x: NDArray, y: NDArray) -> NDArray:
    b = np.roll(y, -1) - np.roll(y, -2)
    c = np.roll(x, -1) - np.roll(x, -2)
    matrix = b[:, np.newaxis] * b[np.newaxis, :]
    matrix += c[:, np.newaxis] * c[np.newaxis, :]
    matrix /= 4.0 * _area(x, y)
    return matrix.T.flatten()


def DIFFERENTIAL_MATRIX_3X3(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    b = np.roll(y, -1) - np.roll(y, -2)
    c = np.roll(x, -1) - np.roll(x, -2)
    factor = _area(x, y) / _det(x, y)
    b *= factor / 3
    c *= factor / 3
    return np.vstack((b, b, b)).flatten(), np.vstack((c, c, c)).flatten()


def TERM_MATRIX_3X3(x: NDArray, y: NDArray) -> NDArray:
    matrix = np.full((3, 3), 1.0 / 12.0)
    np.fill_diagonal(matrix, 1.0 / 6.0)
    matrix *= _area(x, y)
    return matrix.flatten()


class Fem2d:

    def __init__(self, mesh: PolygonMesh):
        """A class of two-dimensional finite element method (2D-FEM).

        Using this class, boundary value problems for partial  differential equations can be discretized using the 2D-FEM.
        It can handle both triangle and rectangle elements as input mesh data.
        (A rectangular element will be implemented in the future).
        """

        self.mesh = mesh

        if mesh.mesh_type == 3:
            local_laplacian_matrix = LAPLACIAN_MATRIX_3X3
            local_differential_matrix = DIFFERENTIAL_MATRIX_3X3
            local_term_matrix = TERM_MATRIX_3X3
        else:
            # TODO: Implementation of rectangular element.
            pass

        self.laplacian_matrix = lil_matrix((mesh.n_nodes, mesh.n_nodes))
        self.differential_matrix = [lil_matrix((mesh.n_nodes, mesh.n_nodes)), lil_matrix((mesh.n_nodes, mesh.n_nodes))]
        self.term_matrix = lil_matrix((mesh.n_nodes, mesh.n_nodes))
        for idx in mesh.element_nodes:
            x = mesh.x[list(idx)]
            y = mesh.y[list(idx)]
            _update_global_matrix(self.laplacian_matrix, idx, local_laplacian_matrix(x, y))
            _update_global_matrices(self.differential_matrix, idx, local_differential_matrix(x, y))
            _update_global_matrix(self.term_matrix, idx, local_term_matrix(x, y))

    def laplacian(self, vec: NDArray) -> NDArray:
        """Compute the matrix-vector product corresponding to the Laplace operator.

        Args:
            vec (NDArray): discrete data to which the Laplace transform is applied.

        Returns:
            NDArray: discrete data resulting from the application of the Laplace transform.
        """
        return np.array(self.laplacian_matrix.dot(vec))

    def differential(self, vec: NDArray) -> tuple[NDArray, ...]:
        """Compute the matrix-vector product corresponding to the 1st-order differentiation.

        Args:
            vec (NDArray): discrete data to which the 1st-order differentiation is applied.

        Returns:
            NDArray: discrete data resulting from the application of the 1st-order differentiation.
        """
        return tuple(np.array(matrix.dot(vec)) for matrix in self.differential_matrix)

    def term(self, vec: NDArray) -> NDArray:
        """Compute the matrix-vector product corresponding to the general term.

        Args:
            vec (NDArray): discrete data treated as a general term.

        Returns:
            NDArray: discrete data for the general term formulated in FEM.
        """
        return np.array(self.term_matrix.dot(vec))

    def implement_dirichlet(self, coefficient: lil_matrix, rhs: NDArray, values: NDArray) -> None:
        """Apply the Dirichlet conditions to the coefficient matrix and right-hand side vector.

        Arguments `coefficient` and 'rhs' will be overwritten with the results after applying the condition.

        Args:
            coefficient (lil_matrix): coefficient matrix.
            rhs (NDArray): right-hand side vector.
            values (NDArray): boundary values.
        """
        global_index = self.mesh.boundary_node_index(Condition.DIRICHLET)
        d = np.zeros_like(rhs)
        d[global_index] = values[global_index]
        rhs -= coefficient.dot(d)
        rhs[global_index] = values[global_index]
        coefficient[global_index, :] = 0.0
        coefficient[:, global_index] = 0.0
        coefficient[global_index, global_index] = 1.0

    def implement_neumann(self, rhs: NDArray, values: NDArray) -> None:
        """Apply the Neumann conditions to the right-hand side vector.

        Args:
            rhs (NDArray): right-hand side vector.
            values (NDArray): boundary values.
        """
        neumann_elements = self.mesh.boundary_element_index(Condition.NEUMANN, both=True)
        g = np.sum(values[self.mesh.boundary_nodes] * self.mesh.normals, axis=1)
        for s, e in neumann_elements:
            i, j = self.mesh.boundary_nodes[s], self.mesh.boundary_nodes[e]
            st = np.array((self.mesh.x[i], self.mesh.y[i]))
            ed = np.array((self.mesh.x[j], self.mesh.y[j]))
            d = np.linalg.norm(ed - st)
            rhs[i] += g[s] * d / 3 + g[e] * d / 6
            rhs[j] += g[s] * d / 6 + g[e] * d / 3


def _update_global_matrix(matrix: lil_matrix, indexes: tuple[int, ...], local_matrix: NDArray) -> None:
    """Update a global matrix.

    Args:
        matrix (lil_matrix): matrix to be updated. It will be overwritten with the update results.
        indexes (tuple[int, ...]): column and row numbers of the updated elements.
        local_matrix (NDArray): local matrix used for update.
    """
    row_idx = np.vstack((indexes,) * len(indexes)).T.flatten()
    col_idx = np.hstack((indexes,) * len(indexes))
    matrix[row_idx, col_idx] += local_matrix


def _update_global_matrices(matrices: list[lil_matrix], indexes: tuple[int, ...], local_matrices: tuple[NDArray, ...]) -> None:
    """_summary_

    Args:
        matrices (list[lil_matrix]): list of matrices to be updated. It will be overwritten with the update results.
        indexes (tuple[int, ...]): column and row numbers of the updated elements.
        local_matrices (tuple[NDArray, ...]): list of local matrices used for update.
    """
    row_idx = np.vstack((indexes,) * len(indexes)).T.flatten()
    col_idx = np.hstack((indexes,) * len(indexes))
    for i, local_matrix in enumerate(local_matrices):
        matrices[i][row_idx, col_idx] += local_matrix
