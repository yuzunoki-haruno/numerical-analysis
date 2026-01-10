import itertools
from pathlib import Path
import tempfile

import numpy as np
import pytest

from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh1d import (
    LineMesh,
    MeshType,
    generate_line_mesh,
)

CONDITIONS = [Condition.DIRICHLET, Condition.NEUMANN]

MESH_TYPE_LIST = [MeshType.FirstOrder, MeshType.SecondOrder]


class TestLineMesh:

    @pytest.mark.parametrize("conditions", itertools.product(CONDITIONS, repeat=2))
    def test_init_first_order(self, conditions: list[Condition]) -> None:
        n_nodes, xmin, xmax = 4, -2.0, 1.0
        cmin, cmax = conditions
        mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax, MeshType.FirstOrder)
        # verifying expectations and values.
        assert mesh.n_nodes == n_nodes
        assert mesh.n_elements == (n_nodes - 1)
        np.testing.assert_equal(mesh.x, (-2, -1, 0, 1))
        np.testing.assert_equal(mesh.element_nodes, ((0, 1), (1, 2), (2, 3)))
        np.testing.assert_equal(mesh.boundary_nodes, (0, n_nodes - 1))
        np.testing.assert_equal(mesh.boundary_element_nodes, (0, 1))
        np.testing.assert_equal(mesh.normals, (-1, 1))
        np.testing.assert_equal(mesh.conditions, conditions)
        assert mesh.mesh_type == MeshType.FirstOrder and mesh.mesh_type == 2
        assert mesh.x[mesh.boundary_nodes[0]] == xmin
        assert mesh.x[mesh.boundary_nodes[1]] == xmax

    @pytest.mark.parametrize("conditions", itertools.product(CONDITIONS, repeat=2))
    def test_init_second_order(self, conditions: list[Condition]) -> None:
        n_nodes, xmin, xmax = 7, -2.0, 4.0
        cmin, cmax = conditions
        mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax, mesh_type=MeshType.SecondOrder)
        # verifying expectations and values.
        assert mesh.check_data()
        assert mesh.n_nodes == n_nodes
        assert mesh.n_elements == 3
        np.testing.assert_equal(mesh.x, (-2, -1, 0, 1, 2, 3, 4))
        np.testing.assert_equal(mesh.element_nodes, ((0, 2, 1), (2, 4, 3), (4, 6, 5)))
        np.testing.assert_equal(mesh.boundary_nodes, (0, n_nodes - 1))
        np.testing.assert_equal(mesh.boundary_element_nodes, (0, 1))
        np.testing.assert_equal(mesh.normals, (-1, 1))
        np.testing.assert_equal(mesh.conditions, conditions)
        assert mesh.mesh_type == MeshType.SecondOrder and mesh.mesh_type == 3
        assert mesh.x[mesh.boundary_nodes[0]] == xmin
        assert mesh.x[mesh.boundary_nodes[1]] == xmax

    @pytest.mark.parametrize("mesh_type", MESH_TYPE_LIST)
    def test_file_io(self, mesh_type: MeshType) -> None:
        n_nodes, xmin, xmax = 7, -2.0, 4.0
        cmin, cmax = Condition.DIRICHLET, Condition.NEUMANN
        mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax, mesh_type=mesh_type)
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / "save.npz"
            mesh.save_npz(filename)
            loaded_mesh = LineMesh.load_npz(filename)
            # verifying expectations and values.
            assert loaded_mesh.check_data()
            assert mesh.n_nodes == loaded_mesh.n_nodes
            assert mesh.n_elements == loaded_mesh.n_elements
            np.testing.assert_equal(mesh.x, loaded_mesh.x)
            np.testing.assert_equal(mesh.element_nodes, loaded_mesh.element_nodes)
            np.testing.assert_equal(mesh.boundary_nodes, loaded_mesh.boundary_nodes)
            np.testing.assert_equal(mesh.boundary_element_nodes, loaded_mesh.boundary_element_nodes)
            np.testing.assert_equal(mesh.normals, loaded_mesh.normals)
            np.testing.assert_equal(mesh.conditions, loaded_mesh.conditions)
            assert mesh.mesh_type == loaded_mesh.mesh_type
