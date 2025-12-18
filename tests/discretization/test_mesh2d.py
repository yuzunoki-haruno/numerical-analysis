import itertools
import tempfile
from pathlib import Path

import numpy as np
import pytest

from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh2d import (
    MeshType,
    PolygonMesh,
    generate_polygon_mesh,
)

CONDITIONS = itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4)

DIRICHLET_NODE_INDEXES: tuple[list, ...] = (
    [0, 1, 2, 3, 7, 11, 10, 9, 8, 4],
    [0, 1, 2, 3, 7, 11, 10, 9, 8],
    [0, 1, 2, 3, 7, 11, 8, 4],
    [0, 1, 2, 3, 7, 11, 8],
    [0, 1, 2, 3, 11, 10, 9, 8, 4],
    [0, 1, 2, 3, 11, 10, 9, 8],
    [0, 1, 2, 3, 11, 8, 4],
    [0, 1, 2, 3, 11, 8],
    [0, 3, 7, 11, 10, 9, 8, 4],
    [0, 3, 7, 11, 10, 9, 8],
    [0, 3, 7, 11, 8, 4],
    [0, 3, 7, 11, 8],
    [0, 3, 11, 10, 9, 8, 4],
    [0, 3, 11, 10, 9, 8],
    [0, 3, 11, 8, 4],
)

NEUMAN_NODE_INDEXES: tuple[list, ...] = (
    [],
    [4],
    [10, 9],
    [10, 9, 4],
    [7],
    [7, 4],
    [7, 10, 9],
    [7, 10, 9, 4],
    [1, 2],
    [1, 2, 4],
    [1, 2, 10, 9],
    [1, 2, 10, 9, 4],
    [1, 2, 7],
    [1, 2, 7, 4],
    [1, 2, 7, 10, 9],
    [1, 2, 7, 10, 9, 4],
)


class TestPolygonMesh:

    @pytest.mark.parametrize(
        "conditions, dirichlet_idx, neumann_idx",
        zip(CONDITIONS, DIRICHLET_NODE_INDEXES, NEUMAN_NODE_INDEXES),
    )
    def test_init(self, conditions: list[Condition], dirichlet_idx: list[int], neumann_idx: list[int]) -> None:
        n_x, xmin, xmax = 4, -1.0, 2.0
        n_y, ymin, ymax = 3, 0.0, 2.0
        c1, c2, c3, c4 = conditions
        mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4)
        assert mesh.n_nodes == n_x * n_y
        assert mesh.n_elements == 2 * (n_x - 1) * (n_y - 1)
        np.testing.assert_equal(mesh.boundary_nodes, (0, 1, 2, 3, 7, 11, 10, 9, 8, 4))
        # assert mesh.conditions == conditions
        assert mesh.mesh_type == MeshType.Triangle and mesh.mesh_type == 3
        np.testing.assert_equal(mesh.x, (-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2))
        np.testing.assert_equal(mesh.y, (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2))
        np.testing.assert_equal(
            mesh.element_nodes,
            (
                (0, 1, 5),
                (1, 2, 6),
                (2, 3, 7),
                (4, 5, 9),
                (5, 6, 10),
                (6, 7, 11),
                (0, 4, 5),
                (1, 5, 6),
                (2, 6, 7),
                (4, 8, 9),
                (5, 9, 10),
                (6, 10, 11),
            ),
        )
        np.testing.assert_equal(
            mesh.normals, ([0, -1], [0, -1], [0, -1], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [-1, 0], [-1, 0])
        )
        np.testing.assert_equal(mesh.boundary_nodes, (0, 1, 2, 3, 7, 11, 10, 9, 8, 4))
        np.testing.assert_equal(
            mesh.boundary_element_nodes, ([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0])
        )
        np.testing.assert_equal(mesh.boudary_node_index(Condition.DIRICHLET), dirichlet_idx)
        np.testing.assert_equal(mesh.boudary_node_index(Condition.NEUMANN), neumann_idx)

    def test_file_io(self) -> None:
        n_x, xmin, xmax = 4, -1.0, 2.0
        n_y, ymin, ymax = 3, 0.0, 2.0
        c1, c2, c3, c4 = Condition.DIRICHLET, Condition.NEUMANN, Condition.DIRICHLET, Condition.NEUMANN
        mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / "save.npz"
            mesh.save(filename)
            loaded_mesh = PolygonMesh.load(filename)
            assert mesh.n_nodes == loaded_mesh.n_nodes
            assert mesh.n_elements == loaded_mesh.n_elements
            assert mesh.mesh_type == loaded_mesh.mesh_type
            np.testing.assert_equal(mesh.boundary_nodes, loaded_mesh.boundary_nodes)
            np.testing.assert_equal(mesh.conditions, loaded_mesh.conditions)
            np.testing.assert_equal(mesh.x, loaded_mesh.x)
            np.testing.assert_equal(mesh.y, loaded_mesh.y)
            np.testing.assert_equal(mesh.element_nodes, loaded_mesh.element_nodes)
            np.testing.assert_equal(mesh.normals, loaded_mesh.normals)
            np.testing.assert_equal(mesh.mesh_type, loaded_mesh.mesh_type)
