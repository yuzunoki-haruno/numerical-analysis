from enum import IntEnum
from pathlib import Path
from typing import NamedTuple, Self

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from .condition import Condition


class MeshType(IntEnum):
    Triangle = 3
    Rectangle = 4


class PolygonMesh(NamedTuple):
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

    n_x: int
    n_y: int
    n_nodes: int
    n_elements: int
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    element_nodes: list[tuple[int, int, int]] | list[tuple[int, int, int, int]]
    boundary_nodes: NDArray[np.int64]
    boundary_element_nodes: NDArray[np.int64]
    normals: NDArray[np.float64]
    conditions: list[Condition]
    mesh_type: MeshType

    def __repr__(self) -> str:
        return f"<PolygonMesh number of elements: {self.n_elements}, boundray condirions: {self.conditions}, mesh type: {self.mesh_type}>"

    def save(self, filename: str | Path) -> None:
        """Save the mesh data as an npz file.

        Args:
            filename (str | Path): file path of the npz file.
        """
        np.savez_compressed(
            filename,
            n_x=self.n_x,
            n_y=self.n_y,
            n_nodes=self.n_nodes,
            n_elements=self.n_elements,
            x=self.x,
            y=self.y,
            element_nodes=self.element_nodes,
            boundary_nodes=self.boundary_nodes,
            boundary_element_nodes=self.boundary_element_nodes,
            normals=self.normals,
            conditions=self.conditions,
            mesh_type=self.mesh_type,
        )

    @classmethod
    def load(cls, filename: str | Path) -> Self:
        """Load the mesh data as an npz file.

        Args:
            filename (str | Path): file path of the npz file.

        Returns:
            LineMesh: mesh data
        """
        data = np.load(filename)
        return cls(
            data["n_x"],
            data["n_y"],
            data["n_nodes"],
            data["n_elements"],
            data["x"],
            data["y"],
            data["element_nodes"],
            data["boundary_nodes"],
            data["boundary_element_nodes"],
            data["normals"],
            data["conditions"],
            data["mesh_type"],
        )

    def boudary_node_index(self, condition: Condition) -> NDArray[np.int64]:
        """Extract the global node numbers of boundary nodes with the specified condition applied.

        Args:
            condition (Condition): target condition.

        Returns:
            NDArray[np.int64]: global node numbers of boundary nodes with `condition` applied.
        """
        is_vaild = [c == condition for c in self.conditions]
        return self.boundary_nodes[is_vaild]

    def boudary_element_index(self, condition: Condition, both: bool = True) -> NDArray[np.int64]:
        """Extract the node numbers of the start and end points of boundary elements with the specified condition applied.
        Args:
            condition (Condition): target condition.
            both (bool, optional): _description_. Defaults to True.

        Returns:
            NDArray[np.int64]: boundary node numbers of the start and end points of boundary elements with `condition` applied.
        """
        if both:
            is_vaild = [
                self.conditions[i] == condition and self.conditions[j] == condition for i, j in self.boundary_element_nodes
            ]
        else:
            is_vaild = [
                self.conditions[i] == condition or self.conditions[j] == condition for i, j in self.boundary_element_nodes
            ]
        return self.boundary_element_nodes[is_vaild]

    def plot(self, filename: Path | str) -> None:
        """Visualize this mesh data.

        Args:
            filename (Path | str): file path for saving visualization results.
        """
        fig, ax = plt.subplots()
        ax.scatter(self.x, self.y, c="black")
        for idx in self.element_nodes:
            if len(idx) == 4:
                index = list(idx)
            else:
                index = list(idx) + [idx[0]]
            ax.plot(self.x[index], self.y[index], c="black")
        indexes = self.boudary_node_index(Condition.DIRICHLET)
        ax.scatter(self.x[indexes], self.y[indexes], c="blue")
        indexes = self.boudary_node_index(Condition.NEUMANN)
        ax.scatter(self.x[indexes], self.y[indexes], c="red")
        for idx in self.boudary_element_index(Condition.DIRICHLET, both=False):
            index = self.boundary_nodes[idx]
            ax.plot(self.x[index], self.y[index], c="blue")
        for idx in self.boudary_element_index(Condition.NEUMANN, both=True):
            index = self.boundary_nodes[idx]
            ax.plot(self.x[index], self.y[index], c="red")
        fig.tight_layout()
        fig.savefig(filename)


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
    x, y = x.flatten(), y.flatten()
    element_nodes: list[tuple[int, int, int]] | list[tuple[int, int, int, int]]
    if mesh_type == MeshType.Triangle:
        n_nodes = n_x * n_y
        n_elements = 2 * m_x * m_y
        element_nodes = [(i, i + 1, i + n_x + 1) for i in range(n_elements) if (i + 1) % n_x > 0 and i < n_x * m_y]
        element_nodes += [(i, i + n_x, i + n_x + 1) for i in range(n_elements) if (i + 1) % n_x > 0 and i < n_x * m_y]
    else:
        n_nodes = n_x * n_y
        n_elements = m_x * m_y
        element_nodes = [(i, i + 1, n_x * i, n_x * (i + 1)) for i in range(m_y)]
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
        n_x,
        n_y,
        n_nodes,
        n_elements,
        x,
        y,
        element_nodes,
        np.array(boundary_nodes, dtype=np.int64),
        np.array(boundary_element_nodes, dtype=np.int64),
        np.array(normals, dtype=float),
        conditions,
        mesh_type,
    )
