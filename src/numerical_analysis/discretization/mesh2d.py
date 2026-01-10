from enum import IntEnum
from pathlib import Path
from typing import Self

import numpy as np

from .base import Mesh
from .condition import Condition


class MeshType(IntEnum):
    Triangle = 3
    Rectangle = 4


class PolygonMesh(Mesh):
    """A two-dimensional mesh.

    This class corresponds to triangle and rectangle elements (A rectangular element will be implemented in the future).
    This class cannot contain a mixture of triangles and quadrilaterals.
    The following function provides the result of dividing the analysis domain into homogeneous triangles/rectangles.
        - numerical_analysis.discretization.mesh2d.generate_polygon_mesh(...)

    Attributes:
        n_x (int): number of nodes along the x-axis.
        n_y (int): number of nodes along the y-axis.
        n_nodes (int): total number of the nodes.
        n_elements (int): number of the elements.
        x (NDArray[np.float64]): x-coordinate of the node.
        y (NDArray[np.float64]): y-coordinate of the node.
        element_nodes (list[tuple[int, ...]]): node indexes constituting the elements.
        boundary_nodes (NDArray[np.int64]): node indexes on the boundary.
        boundary_element_nodes (NDArray[np.int64]): boundary node indexes constituting the boundary elements.
        normals (NDArray[np.float64]): outward unit normal vectors on the boundary.
        conditions (list[Condition]): boundary conditions on boundary nodes.
        mesh_type (MeshType): The shape of the mesh (MeshType.Triangle or MeshType.Rectangle).
    """

    def check_data(self) -> bool:
        num_boundary_nodes = len(self.boundary_nodes)
        if self.x.shape != (self.n_nodes, 2):
            return False
        if self.element_nodes.shape != (self.n_elements, int(self.mesh_type)):
            return False
        if self.boundary_nodes.shape != (num_boundary_nodes,):
            return False
        if self.boundary_element_nodes.shape != (num_boundary_nodes, 2):
            return False
        if self.normals.shape != (num_boundary_nodes, 2):
            return False
        if len(self.conditions) != num_boundary_nodes:
            return False
        if not np.all([isinstance(c, Condition) for c in self.conditions]):
            return False
        if not isinstance(self.mesh_type, MeshType):
            return False
        return True

    def __repr__(self) -> str:
        return f"<PolygonMesh number of elements: {self.n_elements}, boundary conditions: {self.conditions}, mesh type: {self.mesh_type}>"

    def save_npz(self, filename: str | Path) -> None:
        """Save the mesh data as an npz file.

        Args:
            filename (str | Path): file path of the npz file.
        """
        np.savez_compressed(
            filename,
            n_nodes=self.n_nodes,
            n_elements=self.n_elements,
            x=self.x,
            element_nodes=self.element_nodes,
            boundary_nodes=self.boundary_nodes,
            boundary_element_nodes=self.boundary_element_nodes,
            normals=self.normals,
            conditions=self.conditions,
            mesh_type=self.mesh_type,
        )

    @classmethod
    def load_npz(cls, filename: str | Path) -> Self:
        """Load the mesh data as an npz file.

        Args:
            filename (str | Path): file path of the npz file.

        Returns:
            LineMesh: mesh data
        """
        data = np.load(filename)
        return cls(
            data["n_nodes"],
            data["n_elements"],
            data["x"],
            data["element_nodes"],
            data["boundary_nodes"],
            data["boundary_element_nodes"],
            data["normals"],
            tuple(Condition(c) for c in data["conditions"]),
            MeshType(data["mesh_type"]),
        )


def generate_polygon_mesh(
    n_x: int,
    xmin: float,
    xmax: float,
    n_y: int,
    ymin: float,
    ymax: float,
    c1: Condition,
    c2: Condition,
    c3: Condition,
    c4: Condition,
    mesh_type: MeshType = MeshType.Triangle,
) -> PolygonMesh:
    """Generate a two-dimensional mesh data.

    The following function provides the result of dividing a rectangle into homogeneous triangles/rectangles.

    Args:
        n_x (int): number of nodes along the x-axis.
        xmin (float): minimum along the x-axis
        xmax (float): maximum along the x-axis.
        n_y (int): number of nodes along the y-axis.
        ymin (float): minimum along the y-axis
        ymax (float): maximum along the y-axis.
        c1 (Condition): boundary conditions imposed on the bottom side.
        c2 (Condition): boundary conditions imposed on the right side.
        c3 (Condition): boundary conditions imposed on the top side.
        c4 (Condition): boundary conditions imposed on the left side.
        mesh_type (MeshType, optional): The shape of the mesh (MeshType.Triangle or MeshType.Rectangle). Defaults to MeshType.Triangle.

    Returns:
        PolygonMesh: two-dimensional mesh data.
    """

    m_x, m_y = n_x - 1, n_y - 1
    x_ = np.linspace(xmin, xmax, num=n_x)
    y_ = np.linspace(ymin, ymax, num=n_y)
    x, y = np.meshgrid(x_, y_, indexing="xy")
    element_nodes: list[tuple[int, int, int]] | list[tuple[int, int, int, int]]
    if mesh_type == MeshType.Triangle:
        n_nodes = n_x * n_y
        n_elements = 2 * m_x * m_y
        element_nodes = [(i, i + 1, i + n_x + 1) for i in range(n_nodes) if (i + 1) % n_x > 0 and i < n_x * m_y]
        element_nodes += [(i, i + n_x, i + n_x + 1) for i in range(n_nodes) if (i + 1) % n_x > 0 and i < n_x * m_y]
    elif mesh_type == MeshType.Rectangle:
        n_nodes = n_x * n_y
        n_elements = m_x * m_y
        element_nodes = [(i, i + 1, i + n_x + 1, i + n_x) for i in range(n_nodes) if (i + 1) % n_x > 0 and i < n_x * m_y]
    else:
        raise ValueError()
    boundary_nodes = list(i for i in range(m_x))
    boundary_nodes.extend(list(n_x * i - 1 for i in range(1, n_y)))
    boundary_nodes.extend(list(i - 1 for i in range(n_nodes, n_nodes - m_x, -1)))
    boundary_nodes.extend(list(n_x * i for i in range(m_y, 0, -1)))
    num_boundary_nodes = len(boundary_nodes)
    boundary_element_nodes = [(i, (i + 1) % num_boundary_nodes) for i in range(num_boundary_nodes)]
    normals = [[0, -1]] * m_x + [[1, 0]] * m_y + [[0, 1]] * m_x + [[-1, 0]] * m_y
    conditions = list()
    conditions.extend([c1] * m_x)
    conditions.extend([c2] * m_y)
    conditions.extend([c3] * m_x)
    conditions.extend([c4] * m_y)
    conditions[0] = Condition.DIRICHLET
    conditions[n_x - 1] = Condition.DIRICHLET
    conditions[n_x + n_y - 2] = Condition.DIRICHLET
    conditions[2 * n_x + n_y - 3] = Condition.DIRICHLET
    return PolygonMesh(
        n_nodes,
        n_elements,
        np.vstack((x.flatten(), y.flatten())).T,
        np.array(element_nodes),
        np.array(boundary_nodes, dtype=np.int64),
        np.array(boundary_element_nodes, dtype=np.int64),
        np.array(normals, dtype=float),
        tuple(conditions),
        mesh_type,
    )
