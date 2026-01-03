import abc

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from numerical_analysis.discretization.mesh1d import LineMesh
from numerical_analysis.discretization.mesh2d import PolygonMesh
from numerical_analysis.fem import Fem1d, Fem2d


class Problem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, fem: LineMesh | PolygonMesh) -> None: ...

    @abc.abstractmethod
    def formulate(self, fem: Fem1d | Fem2d) -> tuple[lil_matrix, NDArray]: ...

    @property
    @abc.abstractmethod
    def u(self) -> NDArray: ...

    @property
    @abc.abstractmethod
    def g(self) -> NDArray: ...

    @property
    @abc.abstractmethod
    def f(self) -> NDArray: ...


class LaplaceProblem1D(Problem):
    def __init__(self, mesh: LineMesh) -> None:
        a, b = 2.0, 1.0
        self.u_ = a * mesh.x + b
        self.g_ = a * np.ones_like(self.u_)
        self.f_ = np.zeros_like(self.u_)

    def formulate(self, fem: Fem1d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        rhs = fem.term(self.f_)
        return coefficient, rhs

    @property
    def u(self) -> NDArray:
        return self.u_

    @property
    def g(self) -> NDArray:
        return self.g_

    @property
    def f(self) -> NDArray:
        return self.f_


class PoissonProblem1D(Problem):
    def __init__(self, mesh: LineMesh) -> None:
        a = 2.0 * np.pi
        self.u_ = np.cos(a * mesh.x)
        self.g_ = -a * np.sin(a * mesh.x)
        self.f_ = a**2 * np.cos(a * mesh.x)

    def formulate(self, fem: Fem1d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        rhs = fem.term(self.f_)
        return coefficient, rhs

    @property
    def u(self) -> NDArray:
        return self.u_

    @property
    def g(self) -> NDArray:
        return self.g_

    @property
    def f(self) -> NDArray:
        return self.f_


class HelmholtzProblem1D(Problem):
    def __init__(self, mesh: LineMesh) -> None:
        a = 2.0 * np.pi
        self.u_ = np.cos(a * mesh.x)
        self.g_ = -a * np.sin(a * mesh.x)
        self.f_ = np.zeros_like(self.u_)
        self.k2 = a**2

    def formulate(self, fem: Fem1d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        coefficient -= self.k2 * fem.term_matrix
        rhs = fem.term(self.f_)
        return coefficient, rhs

    @property
    def u(self) -> NDArray:
        return self.u_

    @property
    def g(self) -> NDArray:
        return self.g_

    @property
    def f(self) -> NDArray:
        return np.zeros_like(self.u_)


class LinearProblem1D(Problem):
    def __init__(self, mesh: LineMesh) -> None:
        self.u_ = np.cos(2 * mesh.x) + np.exp(np.pi / 2) * np.sin(2 * mesh.x)
        self.u_ *= np.exp(-2 * mesh.x)
        self.g_ = (np.exp(np.pi / 2) - 1) * np.cos(2 * mesh.x)
        self.g_ -= (np.exp(np.pi / 2) + 1) * np.sin(2 * mesh.x)
        self.g_ *= 2 * np.exp(-2 * mesh.x)
        self.f_ = np.zeros_like(self.u_)

    def formulate(self, fem: Fem1d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        coefficient += -4 * fem.differential_matrices[0]
        coefficient += -8 * fem.term_matrix
        rhs = fem.term(self.f_)
        return coefficient, rhs

    @property
    def u(self) -> NDArray:
        return self.u_

    @property
    def g(self) -> NDArray:
        return self.g_

    @property
    def f(self) -> NDArray:
        return self.f_


class LaplaceProblem2D(Problem):
    def __init__(self, mesh: PolygonMesh) -> None:
        x, y = mesh.x[:, 0], mesh.x[:, 1]
        self.u_ = np.sin(2.0 * np.pi * x) * np.sinh(2.0 * np.pi * y)
        gx = 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.sinh(2.0 * np.pi * y)
        gy = 2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.cosh(2.0 * np.pi * y)
        self.g_ = _normal_derivative(gx, gy, mesh.normals, mesh.boundary_nodes)
        self.f_ = np.zeros_like(self.u_)

    def formulate(self, fem: Fem2d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        rhs = fem.term(self.f_)
        return coefficient, rhs

    @property
    def u(self) -> NDArray:
        return self.u_

    @property
    def g(self) -> NDArray:
        return self.g_

    @property
    def f(self) -> NDArray:
        return self.f_


class PoissonProblem2D(Problem):
    def __init__(self, mesh: PolygonMesh) -> None:
        x, y = mesh.x[:, 0], mesh.x[:, 1]
        self.u_ = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        gx = 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        gy = -2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        self.g_ = _normal_derivative(gx, gy, mesh.normals, mesh.boundary_nodes)
        self.f_ = 8.0 * np.pi**2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

    def formulate(self, fem: Fem2d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        rhs = fem.term(self.f)
        return coefficient, rhs

    @property
    def u(self) -> NDArray:
        return self.u_

    @property
    def g(self) -> NDArray:
        return self.g_

    @property
    def f(self) -> NDArray:
        return self.f_


class HelmholtzProblem2D(Problem):
    def __init__(self, mesh: PolygonMesh) -> None:
        x, y = mesh.x[:, 0], mesh.x[:, 1]
        self.u_ = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        gx = 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        gy = -2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        self.g_ = _normal_derivative(gx, gy, mesh.normals, mesh.boundary_nodes)
        self.f_ = np.zeros_like(self.u_)

    def formulate(self, fem: Fem2d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        coefficient += -8.0 * np.pi**2 * fem.term_matrix
        rhs = fem.term(self.f_)
        return coefficient, rhs

    @property
    def u(self) -> NDArray:
        return self.u_

    @property
    def g(self) -> NDArray:
        return self.g_

    @property
    def f(self) -> NDArray:
        return self.f_


class LinearProblem2D(Problem):
    def __init__(self, mesh: PolygonMesh) -> None:
        x, y = mesh.x[:, 0], mesh.x[:, 1]
        self.u_ = np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        gx = 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        gy = 2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        self.g_ = _normal_derivative(gx, gy, mesh.normals, mesh.boundary_nodes)
        self.f_ = 8.0 * np.pi**2 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        self.f_ -= 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y) / 3.0
        self.f_ -= 2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y) / 5.0

    def formulate(self, fem: Fem2d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        coefficient += fem.differential_matrices[0] / 3.0
        coefficient += fem.differential_matrices[1] / 5.0
        rhs = fem.term(self.f_)
        return coefficient, rhs

    @property
    def u(self) -> NDArray:
        return self.u_

    @property
    def g(self) -> NDArray:
        return self.g_

    @property
    def f(self) -> NDArray:
        return self.f_


def _normal_derivative(grad_x: NDArray, grad_y: NDArray, normal: NDArray, boundary_nodes: NDArray) -> NDArray:
    grad = np.vstack((grad_x, grad_y)).T
    return np.array(np.sum(grad[boundary_nodes] * normal, axis=1))
