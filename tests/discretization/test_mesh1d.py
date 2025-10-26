import itertools
from pathlib import Path
import tempfile

import numpy as np
import pytest

from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh1d import (
    LineMesh,
    generate_line_mesh,
)


class TestLineMesh:

    conditions = [Condition.DIRICHLET, Condition.NEUMANN]

    @pytest.mark.parametrize("conditions", itertools.product(conditions, repeat=2))
    def test_init(self, conditions: list[Condition]) -> None:
        n_nodes, xmin, xmax = 4, -2.0, 1.0
        cmin, cmax = conditions
        mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax)
        assert mesh.n_nodes == n_nodes
        assert mesh.n_elements == (n_nodes - 1)
        assert mesh.boundary_nodes == (0, n_nodes - 1)
        assert mesh.conditions == conditions
        assert mesh.mesh_type == 2
        assert mesh.x[mesh.boundary_nodes[0]] == mesh.x.min()
        assert mesh.x[mesh.boundary_nodes[1]] == mesh.x.max()
        np.testing.assert_equal(mesh.x, [-2, -1, 0, 1])
        np.testing.assert_equal(mesh.element_nodes, [(0, 1), (1, 2), (2, 3)])
        np.testing.assert_equal(mesh.normals, [-1, 1])

    @pytest.mark.parametrize("conditions", itertools.product(conditions, repeat=2))
    def test_init_high_order(self, conditions: list[Condition]) -> None:
        n_nodes, xmin, xmax = 7, -2.0, 4.0
        cmin, cmax = conditions
        mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax, mesh_type=3)
        assert mesh.n_nodes == n_nodes
        assert mesh.n_elements == 3
        assert mesh.boundary_nodes == (0, n_nodes - 1)
        assert mesh.conditions == conditions
        assert mesh.mesh_type == 3
        assert mesh.x[mesh.boundary_nodes[0]] == mesh.x.min()
        assert mesh.x[mesh.boundary_nodes[1]] == mesh.x.max()
        np.testing.assert_equal(mesh.x, [-2, -1, 0, 1, 2, 3, 4])
        np.testing.assert_equal(mesh.element_nodes, [(0, 2, 1), (2, 4, 3), (4, 6, 5)])
        np.testing.assert_equal(mesh.normals, [-1, 1])

    def test_file_io(self) -> None:
        n_nodes, xmin, xmax = 7, -2.0, 4.0
        cmin, cmax = Condition.DIRICHLET, Condition.NEUMANN
        mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax, mesh_type=3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / "save.npz"
            mesh.save(filename)
            loaded_mesh = LineMesh.load(filename)
            assert mesh.n_nodes == loaded_mesh.n_nodes
            assert mesh.n_elements == loaded_mesh.n_elements
            assert mesh.mesh_type == loaded_mesh.mesh_type
            np.testing.assert_equal(mesh.boundary_nodes, loaded_mesh.boundary_nodes)
            np.testing.assert_equal(mesh.conditions, loaded_mesh.conditions)
            np.testing.assert_equal(mesh.x, loaded_mesh.x)
            np.testing.assert_equal(mesh.element_nodes, loaded_mesh.element_nodes)
            np.testing.assert_equal(mesh.normals, loaded_mesh.normals)
