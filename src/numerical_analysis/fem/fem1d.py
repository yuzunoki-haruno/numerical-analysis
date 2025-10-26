import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from numerical_analysis.discretization import Condition, LineMesh

LAPLACIAN_MATRIX_2X2 = np.array((1, -1, -1, 1))
DIFFERENTIAL_MATRIX_2X2 = np.array((-1 / 2, 1 / 2, -1 / 2, 1 / 2))
TERM_MATRIX_2X2 = np.array((1 / 3, 1 / 6, 1 / 6, 1 / 3))

LAPLACIAN_MATRIX_3X3 = np.array(
    (
        7 / 3,
        1 / 3,
        -8 / 3,
        1 / 3,
        7 / 3,
        -8 / 3,
        -8 / 3,
        -8 / 3,
        16 / 3,
    )
)
DIFFERENTIAL_MATRIX_3X3 = np.array(
    (
        -1 / 2,
        -1 / 6,
        2 / 3,
        1 / 6,
        1 / 2,
        -2 / 3,
        -2 / 3,
        2 / 3,
        0,
    )
)
TERM_MATRIX_3X3 = np.array(
    (
        2 / 15,
        -1 / 30,
        1 / 15,
        -1 / 30,
        2 / 15,
        1 / 15,
        1 / 15,
        1 / 15,
        8 / 15,
    )
)


class Fem1d:
    """A class of one-dimensional finite element method (1D-FEM).

    Using this class, boundary value problems for ordinary differential equations can be discretised using the 1D-FEM.
    It can handle both 1st-order and 2nd-order elements as input mesh data.

    """

    def __init__(self, mesh: LineMesh):
        """A class of one-dimensional finite element method (1D-FEM)

        Args:
            mesh (LineMesh): mesh data for the analysis region.
        """
        self.mesh = mesh

        if mesh.mesh_type == 3:
            local_laplacian_matrix = LAPLACIAN_MATRIX_3X3
            local_differential_matrix = DIFFERENTIAL_MATRIX_3X3
            local_term_matrix = TERM_MATRIX_3X3
        else:
            local_laplacian_matrix = LAPLACIAN_MATRIX_2X2
            local_differential_matrix = DIFFERENTIAL_MATRIX_2X2
            local_term_matrix = TERM_MATRIX_2X2

        self.laplacian_matrix = lil_matrix((mesh.n_nodes, mesh.n_nodes))
        self.differential_matrix = lil_matrix((mesh.n_nodes, mesh.n_nodes))
        self.term_matrix = lil_matrix((mesh.n_nodes, mesh.n_nodes))
        for idx in mesh.element_nodes:
            i, j = idx[0], idx[1]
            h = mesh.x[j] - mesh.x[i]
            _update_global_matrix(self.laplacian_matrix, idx, local_laplacian_matrix / h)
            _update_global_matrix(self.differential_matrix, idx, local_differential_matrix)
            _update_global_matrix(self.term_matrix, idx, local_term_matrix * h)

    def laplacian(self, vec: NDArray) -> NDArray:
        """Compute the matrix-vector product corresponding to the Laplace operator.

        Args:
            vec (NDArray): discrete data to which the Laplace transform is applied.

        Returns:
            NDArray: discrete data resulting from the application of the Laplace transform.
        """
        return np.array(self.laplacian_matrix.dot(vec))

    def differential(self, vec: NDArray) -> NDArray:
        """Compute the matrix-vector product corresponding to the 1st-order differentiation.

        Args:
            vec (NDArray): discrete data to which the 1st-order differentiation is applied.

        Returns:
            NDArray: discrete data resulting from the application of the 1st-order differentiation.
        """
        return np.array(self.differential_matrix.dot(vec))

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
        local_index = _get_dilichlet_indexes(self.mesh.conditions)
        global_index = [self.mesh.boundary_nodes[i] for i in local_index]
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
        local_index = _get_neumann_indexes(self.mesh.conditions)
        global_index = [self.mesh.boundary_nodes[i] for i in local_index]
        for i, m in zip(global_index, local_index):
            rhs[i] += self.mesh.normals[m] * values[i]


def _update_global_matrix(matrix: lil_matrix, indexes: tuple[int, ...], local_materix: NDArray) -> None:
    """Update the global matrix with the local matrix.

    Argument `matrix` will be overwritten with the results after updating by the local matrix.

    Args:
        matrix (lil_matrix): global matrix.
        indexes (tuple[int, ...]): global node numbers of the local matrix.
        local_materix (NDArray): local matrix.
    """
    row_idx = np.vstack((indexes,) * len(indexes)).T.flatten()
    col_idx = np.hstack((indexes,) * len(indexes))
    matrix[row_idx, col_idx] += local_materix


def _get_dilichlet_indexes(conditions: tuple[Condition, Condition]) -> tuple[int, ...]:
    """Get the indexes of the boundary node to the Dirichlet condition.

    Args:
        conditions (tuple[Condition, Condition]): boundary conditions on boundary nodes.

    Returns:
        tuple[int, ...]: indexes of the boundary node to the Dirichlet condition.
    """
    return tuple(i for i, c in enumerate(conditions) if c == Condition.DIRICHLET)


def _get_neumann_indexes(conditions: tuple[Condition, Condition]) -> tuple[int, ...]:
    """Get the indexes of the boundary node to the Neumann boundary condition.

    Args:
        conditions (tuple[Condition, Condition]): boundary conditions on boundary nodes.

    Returns:
        tuple[int, ...]: indexes of the boundary node to the Neumann condition.
    """
    return tuple(i for i, c in enumerate(conditions) if c == Condition.NEUMANN)
