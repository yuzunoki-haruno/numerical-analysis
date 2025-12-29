from enum import IntEnum
from pathlib import Path
from typing import Self

import numpy as np

from .base import Mesh
from .condition import Condition


class MeshType(IntEnum):
    FirstOrder = 2
    SecondOrder = 3


class LineMesh(Mesh):
    """A one-dimensional mesh.

    This class corresponds to 1st-order and 2nd-order elements.
    The following function calculates the result of dividing the analysis interval into equal intervals.
        - numerical_analysis.discretization.mesh1d.generate_line_mesh(...)

    Attributes:
        n_nodes (int): number of the nodes.
        n_elements (int): number of the elements.
        x (NDArray[np.float64]): position of the nodes.
        element_nodes (list[tuple[int, int]]): node indexes constituting the elements.
        boundary_nodes (tuple[int, int]): node indexes on the boundary.
        normals (NDArray[np.float64]): outward unit normal vectors on the boundary.
        conditions (tuple[Condition, Condition]): boundary conditions on boundary nodes.
        mesh_type (int): number of nodes per mesh.
    """

    def check_data(self) -> bool:
        if self.x.shape != (self.n_nodes,):
            return False
        if self.element_nodes.shape != (self.n_elements, int(self.mesh_type)):
            return False
        if self.boundary_nodes.shape != (2,):
            return False
        if self.boundary_element_nodes.shape != (2,):
            return False
        if self.normals.shape != (2,):
            return False
        if len(self.conditions) != 2:
            return False
        if not np.all([isinstance(c, Condition) for c in self.conditions]):
            return False
        if not isinstance(self.mesh_type, MeshType):
            return False
        return True

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


def generate_line_mesh(
    n_nodes: int,
    xmin: float,
    xmax: float,
    cmin: Condition,
    cmax: Condition,
    mesh_type: MeshType = MeshType.FirstOrder,
) -> LineMesh:
    """Generate a one-dimensional mesh data.

    This function calculates the result of dividing the analysis interval into equal intervals.

    Args:
        n_nodes (int): number of the nodes.
        xmin (float): minimum value of the analyzed interval.
        xmax (float): maximum value of the analyzed interval.
        cmin (Condition): boundary condition on xmin.
        cmax (Condition): boundary condition on xmax.
        mesh_type (int, optional): number of nodes per mesh (2 or 3). Defaults to 2.

    Returns:
        LineMesh: one-dimensional mesh data.
    """
    x = np.linspace(xmin, xmax, num=n_nodes)
    element_nodes: list[tuple[int, int]] | list[tuple[int, int, int]]
    if mesh_type == 3:
        n_elements = n_nodes // 2
        element_nodes = [(2 * i, 2 * (i + 1), 2 * i + 1) for i in range(n_elements)]
    else:
        n_elements = n_nodes - 1
        element_nodes = [(i, i + 1) for i in range(n_elements)]
    boundary_nodes = (0, n_nodes - 1)
    boundary_element_nodes = (0, 1)
    normals = np.linspace(-1.0, 1.0, num=2)
    conditions = (cmin, cmax)
    return LineMesh(
        n_nodes,
        n_elements,
        x,
        np.array(element_nodes),
        np.array(boundary_nodes),
        np.array(boundary_element_nodes),
        normals,
        conditions,
        mesh_type,
    )
