import abc

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu

from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh2d import generate_polygon_mesh
from numerical_analysis.fem import Fem2d
from numerical_analysis.util import metrics, vis

NAME_PLOT = "result.png"
NAME_TXT = "result.txt"


class Problem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def formulate(self, fem: Fem2d) -> tuple[lil_matrix, NDArray]: ...

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
    def __init__(self, x: NDArray, y: NDArray) -> None:
        self.u_ = np.sin(2.0 * np.pi * x) * np.sinh(2.0 * np.pi * y)
        gx = 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.sinh(2.0 * np.pi * y)
        gy = 2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.cosh(2.0 * np.pi * y)
        self.g_ = np.vstack((gx, gy)).T

    def formulate(self, fem: Fem2d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        rhs = np.zeros_like(self.u_)
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


class PoissonProblem(Problem):
    def __init__(self, x: NDArray, y: NDArray) -> None:
        self.u_ = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        gx = 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        gy = -2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        self.g_ = np.vstack((gx, gy)).T
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


class HelmholtzProblem(Problem):
    def __init__(self, x: NDArray, y: NDArray) -> None:
        self.u_ = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        gx = 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        gy = -2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        self.g_ = np.vstack((gx, gy)).T

    def formulate(self, fem: Fem2d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        coefficient += -8.0 * np.pi**2 * fem.term_matrix
        rhs = np.zeros_like(self.u_)
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
    def __init__(self, x: NDArray, y: NDArray) -> None:
        self.u_ = np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        gx = 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        gy = 2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        self.g_ = np.vstack((gx, gy)).T
        self.f_ = 8.0 * np.pi**2 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        self.f_ -= 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y) / 3.0
        self.f_ -= 2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y) / 5.0

    def formulate(self, fem: Fem2d) -> tuple[lil_matrix, NDArray]:
        coefficient = fem.laplacian_matrix
        coefficient += fem.differential_matrix[0] / 3.0
        coefficient += fem.differential_matrix[1] / 5.0
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


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="2D Finite Element Analysis.")
    parser.add_argument("--n_x", type=int, default=41, help="Number of nodes.")
    parser.add_argument("--xmin", type=float, default=0.25, help="Minimum of x-axis.")
    parser.add_argument("--xmax", type=float, default=1.25, help="Maximum of x-axis.")
    parser.add_argument("--n_y", type=int, default=41, help="Number of nodes.")
    parser.add_argument("--ymin", type=float, default=-0.5, help="Minimum of y-axis.")
    parser.add_argument("--ymax", type=float, default=-1.5, help="Maximum of y-axis.")
    parser.add_argument("--condition", type=str, nargs=4, default=("D", "D", "D", "D"), help="Boundary condition.")
    parser.add_argument("--problem", type=str, default="poisson", help="Problem name.")
    parser.add_argument("--output_dir", type=str, default="result_fem2d", help="Path of output directory.")
    args = parser.parse_args()

    problem = str(args.problem).lower()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    c1 = Condition.DIRICHLET if args.condition[0].lower() == "d" else Condition.NEUMANN
    c2 = Condition.DIRICHLET if args.condition[1].lower() == "d" else Condition.NEUMANN
    c3 = Condition.DIRICHLET if args.condition[2].lower() == "d" else Condition.NEUMANN
    c4 = Condition.DIRICHLET if args.condition[3].lower() == "d" else Condition.NEUMANN

    # mesh generation.
    mesh = generate_polygon_mesh(args.n_x, args.xmin, args.xmax, args.n_y, args.ymin, args.ymax, c1, c2, c3, c4)

    # FEA formulation.
    fem = Fem2d(mesh)
    if problem == "laplace":
        prob: Problem = LaplaceProblem(mesh.x[:, 0], mesh.x[:, 1])
    elif problem == "poisson":
        prob = PoissonProblem(mesh.x[:, 0], mesh.x[:, 1])
    elif problem == "helmholtz":
        prob = HelmholtzProblem(mesh.x[:, 0], mesh.x[:, 1])
    elif problem == "linear":
        prob = LinearProblem(mesh.x[:, 0], mesh.x[:, 1])
    else:
        raise NameError("This program can numerically solve `Laplace`, `Poisson`, `Helmholtz`, and `Linear` problems.")
    coefficient, rhs = prob.formulate(fem)
    fem.implement_dirichlet(coefficient, rhs, prob.u)
    fem.implement_neumann(rhs, prob.g)

    # solving FE linear system.
    sol = splu(csc_matrix(coefficient)).solve(rhs)
    rerror = metrics.relative_error(sol, prob.u)
    print(f"Relative Error: {rerror}")

    # plot result.
    filename = output_dir / NAME_PLOT
    vis.plot2d(filename, mesh.x[:, 0], mesh.x[:, 1], sol, prob.u, args.n_x, args.n_y)

    # write relative error
    with open(output_dir / NAME_TXT, mode="w") as file:
        file.write(f"{problem.title()} problem\n")
        file.write(f"{mesh}\n")
        file.write(f"Relative Error: {rerror}\n")


if __name__ == "__main__":
    main()
