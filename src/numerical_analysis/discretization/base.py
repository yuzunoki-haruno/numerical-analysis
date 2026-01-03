from abc import ABCMeta, abstractmethod
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray


class MeshData(NamedTuple):
    n_nodes: int  # number of Nodes
    n_elements: int  # number of elements
    x: NDArray[np.float64]  # positions of nodes
    element_nodes: NDArray[np.int64]  # node indexes constituting each element.
    boundary_nodes: NDArray[np.int64]  # node indexes on each boundary node.
    boundary_element_nodes: NDArray[np.int64]  # boundary node indexes constituting each boundary element.
    normals: NDArray[np.float64]  # outward unit normal vectors on each boundary node.
    conditions: tuple[StrEnum, ...]  # boundary conditions on boundary nodes.
    mesh_type: IntEnum  # mesh type (Number of nodes constituting each element).

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} number of elements: {self.n_elements}, boundary conditions: {self.conditions}, mesh type: {self.mesh_type.name}>"

    def boundary_node_indexes(self, condition: StrEnum, local: bool = False) -> NDArray[np.int64]:
        mask = [c == condition for c in self.conditions]
        if local:
            local_index = np.array(range(len(self.boundary_nodes)))
            return local_index[mask]
        else:
            return self.boundary_nodes[mask]

    def boundary_element_indexes(self, condition: StrEnum, both: bool = True) -> NDArray[np.int64]:
        mask_st = [c == condition for c in self.conditions]
        mask_ed = [self.conditions[-1] == condition] + [c == condition for c in self.conditions[:-1]]
        if both:
            mask = np.logical_and(mask_st, mask_ed)
        else:
            mask = np.logical_or(mask_st, mask_ed)
        return self.boundary_element_nodes[mask]


class Mesh(MeshData, metaclass=ABCMeta):

    @abstractmethod
    def check_data(self) -> bool: ...

    @abstractmethod
    def save_npz(self, filename: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load_npz(cls, filename: str | Path) -> Any: ...
