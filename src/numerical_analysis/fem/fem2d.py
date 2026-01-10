import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from numerical_analysis.discretization import Condition, PolygonMesh
from numerical_analysis.fem.base import FemBase, implement_dirichlet, update_global_matrix, update_global_matrices


def _area(x: NDArray, y: NDArray) -> np.float64:
    area = np.abs(_det(x, y)) / 2
    return np.float64(area)


def _det(x: NDArray, y: NDArray) -> np.float64:
    x1, x2, x3 = x
    y1, y2, y3 = y
    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    return np.float64(det)


def _length(x: NDArray) -> float:
    xmin, xmax = np.min(x), np.max(x)
    return float(xmax - xmin)


def LAPLACIAN_MATRIX_3X3(x: NDArray, y: NDArray) -> NDArray:
    b = np.roll(y, -1) - np.roll(y, -2)
    c = np.roll(x, -1) - np.roll(x, -2)
    matrix = b[:, np.newaxis] * b[np.newaxis, :]
    matrix += c[:, np.newaxis] * c[np.newaxis, :]
    matrix /= 4.0 * _area(x, y)
    return matrix.T.flatten()


def DIFFERENTIAL_MATRICES_3X3(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
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


def LAPLACIAN_MATRIX_4X4(x: NDArray, y: NDArray) -> NDArray:
    dx, dy = _length(x), _length(y)
    xy, yx = dx / dy, dy / dx
    matrix = np.empty((4, 4))
    matrix[0] = (xy / 3 + yx / 3, -xy / 3 + yx / 6, -xy / 6 - yx / 6, xy / 6 - yx / 3)
    matrix[1] = (-xy / 3 + yx / 6, xy / 3 + yx / 3, xy / 6 - yx / 3, -xy / 6 - yx / 6)
    matrix[2] = (-xy / 6 - yx / 6, xy / 6 - yx / 3, xy / 3 + yx / 3, -xy / 3 + yx / 6)
    matrix[3] = (xy / 6 - yx / 3, -xy / 6 - yx / 6, -xy / 3 + yx / 6, xy / 3 + yx / 3)
    return matrix.flatten()


def DIFFERENTIAL_MATRICES_4X4(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    dx, dy = _length(x), _length(y)
    matrix_x = np.empty((4, 4))
    matrix_x[0] = (-1 / 6, 1 / 6, 1 / 12, -1 / 12)
    matrix_x[1] = (-1 / 6, 1 / 6, 1 / 12, -1 / 12)
    matrix_x[2] = (-1 / 12, 1 / 12, 1 / 6, -1 / 6)
    matrix_x[3] = (-1 / 12, 1 / 12, 1 / 6, -1 / 6)
    matrix_x *= dy
    matrix_y = np.empty((4, 4))
    matrix_y[0] = (-1 / 6, -1 / 12, 1 / 12, 1 / 6)
    matrix_y[1] = (-1 / 12, -1 / 6, 1 / 6, 1 / 12)
    matrix_y[2] = (-1 / 12, -1 / 6, 1 / 6, 1 / 12)
    matrix_y[3] = (-1 / 6, -1 / 12, 1 / 12, 1 / 6)
    matrix_y *= dx
    return matrix_x.flatten(), matrix_y.flatten()


def TERM_MATRIX_4X4(x: NDArray, y: NDArray) -> NDArray:
    dx, dy = _length(x), _length(y)
    area = dx * dy
    matrix = np.empty((4, 4))
    matrix[0] = (area / 9, area / 18, area / 36, area / 18)
    matrix[1] = (area / 18, area / 9, area / 18, area / 36)
    matrix[2] = (area / 36, area / 18, area / 9, area / 18)
    matrix[3] = (area / 18, area / 36, area / 18, area / 9)
    return matrix.flatten()


class Fem2d(FemBase):

    def __init__(self, mesh: PolygonMesh):
        """A class of two-dimensional finite element method (2D-FEM).

        Using this class, boundary value problems for partial  differential equations can be discretized using the 2D-FEM.
        It can handle both triangle and rectangle elements as input mesh data.
        """
        super().__init__(mesh.n_nodes, dim=2)
        self.mesh = mesh

        if mesh.mesh_type == 3:
            local_laplacian_matrix = LAPLACIAN_MATRIX_3X3
            local_differential_matrices = DIFFERENTIAL_MATRICES_3X3
            local_term_matrix = TERM_MATRIX_3X3
        else:
            local_laplacian_matrix = LAPLACIAN_MATRIX_4X4
            local_differential_matrices = DIFFERENTIAL_MATRICES_4X4
            local_term_matrix = TERM_MATRIX_4X4

        for idx in mesh.element_nodes:
            x = mesh.x[list(idx), 0]
            y = mesh.x[list(idx), 1]
            update_global_matrix(self.laplacian_matrix, idx, local_laplacian_matrix(x, y))
            update_global_matrices(self.differential_matrices, idx, local_differential_matrices(x, y))
            update_global_matrix(self.term_matrix, idx, local_term_matrix(x, y))

    def implement_dirichlet(self, coefficient: lil_matrix, rhs: NDArray, values: NDArray) -> None:
        """Apply the Dirichlet conditions to the coefficient matrix and right-hand side vector.

        Arguments `coefficient` and 'rhs' will be overwritten with the results after applying the condition.

        Args:
            coefficient (lil_matrix): coefficient matrix.
            rhs (NDArray): right-hand side vector.
            values (NDArray): boundary values.
        """
        dirichlet_indexes = self.mesh.boundary_node_indexes(Condition.DIRICHLET)
        implement_dirichlet(coefficient, rhs, values, dirichlet_indexes)

    def implement_neumann(self, rhs: NDArray, values: NDArray) -> None:
        """Apply the Neumann conditions to the right-hand side vector.

        Args:
            rhs (NDArray): right-hand side vector.
            values (NDArray): boundary values.
        """
        neumann_elements = self.mesh.boundary_element_indexes(Condition.NEUMANN, both=True)
        for s, e in neumann_elements:
            i, j = self.mesh.boundary_nodes[s], self.mesh.boundary_nodes[e]
            d = np.linalg.norm(self.mesh.x[i] - self.mesh.x[j])
            rhs[i] += values[s] * d / 3 + values[e] * d / 6
            rhs[j] += values[s] * d / 6 + values[e] * d / 3
