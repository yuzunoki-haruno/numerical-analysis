import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu

from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh1d import generate_line_mesh
from numerical_analysis.fem import Fem1d
from numerical_analysis.util import metrics, vis

NAME_PLOT = "result.png"
NAME_TXT = "result.txt"


def laplace_problem(x: NDArray) -> tuple[NDArray, NDArray]:
    u = 2.0 * x + 1.0
    g = 2.0 * np.ones_like(x)
    return u, g


def formulate_laplace(fem: Fem1d) -> tuple[lil_matrix, NDArray]:
    coefficient = fem.laplacian_matrix
    rhs = np.zeros_like(fem.mesh.x)
    return coefficient, rhs


def poisson_problem(x: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    u = np.cos(2.0 * np.pi * x)
    g = -2.0 * np.pi * np.sin(2.0 * np.pi * x)
    f = 4.0 * np.pi**2 * np.cos(2.0 * np.pi * x)
    return u, g, f


def formulate_poisson(fem: Fem1d, f: NDArray) -> tuple[lil_matrix, NDArray]:
    coefficient = fem.laplacian_matrix
    rhs = fem.term(f)
    return coefficient, rhs


def helmholtz_problem(x: NDArray) -> tuple[NDArray, NDArray]:
    u = np.cos(2.0 * np.pi * x)
    g = -2.0 * np.pi * np.sin(2.0 * np.pi * x)
    return u, g


def formulate_helmholtz(fem: Fem1d) -> tuple[lil_matrix, NDArray]:
    coefficient = fem.laplacian_matrix
    coefficient += -4.0 * np.pi**2 * fem.term_matrix
    rhs = np.zeros_like(fem.mesh.x)
    return coefficient, rhs


def linear_problem(x: NDArray) -> tuple[NDArray, NDArray]:
    u = np.cos(2 * x) + np.exp(np.pi / 2) * np.sin(2 * x)
    u *= np.exp(-2 * x)
    g = 2 * (np.exp(np.pi / 2) - 1) * np.cos(2 * x)
    g += -2 * (np.exp(np.pi / 2) + 1) * np.sin(2 * x)
    g *= np.exp(-2 * x)
    return u, g


def formulate_linear(fem: Fem1d) -> tuple[lil_matrix, NDArray]:
    coefficient = fem.laplacian_matrix
    coefficient += -4 * fem.differential_matrix
    coefficient += -8 * fem.term_matrix
    rhs = np.zeros_like(fem.mesh.x)
    return coefficient, rhs


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="1D Finite Element Analysis.")
    parser.add_argument("xmin", type=float, help="Minimum of x-axis.")
    parser.add_argument("xmax", type=float, help="Maximum of x-axis.")
    parser.add_argument("n_nodes", type=int, help="Number of nodes.")
    parser.add_argument("condition", type=str, nargs=2, help="Boundary condition.")
    parser.add_argument("--problem", type=str, default="poisson", help="Problem name.")
    parser.add_argument("--output_dir", type=str, default="result_fem1d", help="Path of output directory.")
    parser.add_argument("--high_order", action="store_true", help="Use 2nd-order element.")
    args = parser.parse_args()

    problem = str(args.problem).lower()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmin = Condition.DIRICHLET if args.condition[0].lower() == "d" else Condition.NEUMANN
    cmax = Condition.DIRICHLET if args.condition[1].lower() == "d" else Condition.NEUMANN
    mesh_type = 3 if args.high_order else 2

    # mesh generation.
    mesh = generate_line_mesh(args.n_nodes, args.xmin, args.xmax, cmin, cmax, mesh_type)

    # FEA formulation.
    fem = Fem1d(mesh)
    if problem == "laplace":
        u, g = laplace_problem(mesh.x)
        coefficient, rhs = formulate_laplace(fem)
    elif problem == "poisson":
        u, g, f = poisson_problem(mesh.x)
        coefficient, rhs = formulate_poisson(fem, f)
    elif problem == "helmholtz":
        u, g = helmholtz_problem(mesh.x)
        coefficient, rhs = formulate_helmholtz(fem)
    elif problem == "linear":
        u, g = linear_problem(mesh.x)
        coefficient, rhs = formulate_linear(fem)
    else:
        raise NameError("This program can numerically solve `Laplace`, `Poisson`, `Helmholtz`, and `Linear` problems.")
    fem.implement_dirichlet(coefficient, rhs, u)
    fem.implement_neumann(rhs, g)

    # solving FE linear system.
    sol = splu(csc_matrix(coefficient)).solve(rhs)
    rerror = metrics.relative_error(sol, u)

    # plot result.
    filename = output_dir / NAME_PLOT
    vis.plot1d(filename, mesh.x, sol, u)

    # write relative error
    with open(output_dir / NAME_TXT, mode="w") as file:
        file.write(f"{problem.title()} problem\n")
        file.write(f"{mesh}\n")
        file.write(f"Relative Error: {rerror}\n")


if __name__ == "__main__":
    main()
