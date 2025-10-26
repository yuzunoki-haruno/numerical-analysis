from pathlib import Path
from typing import NamedTuple, Self

import numpy as np
from numpy.typing import NDArray

from .condition import Condition


class LineMesh(NamedTuple):
    """A one-dimensional mesh.

    This class corresponds to 1st-orfer and 2nd-orfer elements.
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

    n_nodes: int
    n_elements: int
    x: NDArray[np.float64]
    element_nodes: list[tuple[int, int]] | list[tuple[int, int, int]]
    boundary_nodes: tuple[int, int]
    normals: NDArray[np.float64]
    conditions: tuple[Condition, Condition]
    mesh_type: int

    def __repr__(self) -> str:
        return f"<LineMesh number of elements: {self.n_elements}, boundray condirions: {self.conditions}, mesh type: {self.mesh_type}>"

    def save(self, filename: str | Path) -> None:
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
            data["n_nodes"],
            data["n_elements"],
            data["x"],
            data["element_nodes"],
            data["boundary_nodes"],
            data["normals"],
            data["conditions"],
            data["mesh_type"],
        )


def generate_line_mesh(
    n_nodes: int,
    xmin: float,
    xmax: float,
    cmin: Condition,
    cmax: Condition,
    mesh_type: int = 2,
) -> LineMesh:
    """Generate a one-dimensional mesh data.

    This function calculates the result of dividing the analysis interval into equal intervals.

    Args:
        n_nodes (int): number of the nodes.
        xmin (float): minimum value of the analysed interval.
        xmax (float): maximum value of the analysed interval.
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
    normals = np.linspace(-1.0, 1.0, num=2)
    conditions = (cmin, cmax)
    return LineMesh(
        n_nodes,
        n_elements,
        x,
        element_nodes,
        boundary_nodes,
        normals,
        conditions,
        mesh_type,
    )
