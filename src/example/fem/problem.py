import abc

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from numerical_analysis.discretization.mesh1d import LineMesh
from numerical_analysis.fem import Fem1d


class Problem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def formulate(self, fem: Fem1d) -> tuple[lil_matrix, NDArray]: ...

    @property
    @abc.abstractmethod
    def u(self) -> NDArray: ...

    @property
    @abc.abstractmethod
    def g(self) -> NDArray: ...

    @property
    @abc.abstractmethod
    def f(self) -> NDArray: ...


class LaplaceProblem(Problem):
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


class PoissonProblem(Problem):
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


class HelmholtzProblem(Problem):
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


class LinearProblem(Problem):
    def __init__(self, mesh: LineMesh) -> None:
        self.u_ = np.cos(2 * mesh.x) + np.exp(np.pi / 2) * np.sin(2 * mesh.x)
        self.u_ *= np.exp(-2 * mesh.x)
        self.g_ = (np.exp(np.pi / 2) - 1) * np.cos(2 * mesh.x)
        self.g_ -= (np.exp(np.pi / 2) + 1) * np.sin(2 * mesh.x)
        self.g_ *= 2 * np.exp(-2 * mesh.x)
        self.f_ = np.zeros_like(self.u_)

    def formulate(self, fem: Fem1d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        coefficient += -4 * fem.differential_matrix
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
