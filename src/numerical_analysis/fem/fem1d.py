import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from numerical_analysis.discretization import Condition, LineMesh
from numerical_analysis.discretization.mesh1d import MeshType
from numerical_analysis.fem.base import FemBase, implement_dirichlet, update_global_matrix, update_global_matrices

LAPLACIAN_MATRIX_2X2 = np.array((1, -1, -1, 1))
DIFFERENTIAL_MATRICES_2X2 = (np.array((-1 / 2, 1 / 2, -1 / 2, 1 / 2)),)
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
DIFFERENTIAL_MATRICES_3X3 = (
    np.array(
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
    ),
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


class Fem1d(FemBase):
    """A class of one-dimensional finite element method (1D-FEM).

    Using this class, boundary value problems for ordinary differential equations can be discretized using the 1D-FEM.
    It can handle both 1st-order and 2nd-order elements as input mesh data.

    """

    def __init__(self, mesh: LineMesh):
        """A class of one-dimensional finite element method (1D-FEM)

        Args:
            mesh (LineMesh): mesh data for the analysis region.
        """
        super().__init__(mesh.n_nodes, dim=1)
        self.mesh = mesh

        if mesh.mesh_type == MeshType.FirstOrder:
            local_laplacian_matrix = LAPLACIAN_MATRIX_2X2
            local_differential_matrices = DIFFERENTIAL_MATRICES_2X2
            local_term_matrix = TERM_MATRIX_2X2
        elif mesh.mesh_type == MeshType.SecondOrder:
            local_laplacian_matrix = LAPLACIAN_MATRIX_3X3
            local_differential_matrices = DIFFERENTIAL_MATRICES_3X3
            local_term_matrix = TERM_MATRIX_3X3
        else:
            raise ValueError

        for idx in mesh.element_nodes:
            i, j = idx[0], idx[1]
            h = mesh.x[j] - mesh.x[i]
            update_global_matrix(self.laplacian_matrix, idx, local_laplacian_matrix / h)
            update_global_matrices(self.differential_matrices, idx, local_differential_matrices)
            update_global_matrix(self.term_matrix, idx, local_term_matrix * h)

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
        local_index = self.mesh.boundary_node_indexes(Condition.NEUMANN, local=True)
        global_index = [self.mesh.boundary_nodes[i] for i in local_index]
        for i, m in zip(global_index, local_index):
            rhs[i] += self.mesh.normals[m] * values[i]
