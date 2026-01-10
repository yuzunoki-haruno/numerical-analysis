from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from example.fem.problem import PoissonProblem2D, LinearProblem2D, LaplaceProblem2D, HelmholtzProblem2D, Problem
from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh2d import MeshType, generate_polygon_mesh
from numerical_analysis.fem import Fem2d
from numerical_analysis.util import metrics, vis

NAME_PLOT = "result.png"
NAME_TXT = "result.txt"


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
    parser.add_argument("--rectangle", action="store_true", help="Use rectangle element.")
    args = parser.parse_args()

    problem = str(args.problem).lower()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    c1 = Condition.DIRICHLET if args.condition[0].lower() == "d" else Condition.NEUMANN
    c2 = Condition.DIRICHLET if args.condition[1].lower() == "d" else Condition.NEUMANN
    c3 = Condition.DIRICHLET if args.condition[2].lower() == "d" else Condition.NEUMANN
    c4 = Condition.DIRICHLET if args.condition[3].lower() == "d" else Condition.NEUMANN
    mesh_type = MeshType.Rectangle if args.rectangle else MeshType.Triangle

    # mesh generation.
    mesh = generate_polygon_mesh(args.n_x, args.xmin, args.xmax, args.n_y, args.ymin, args.ymax, c1, c2, c3, c4, mesh_type)

    # FEA formulation.
    fem = Fem2d(mesh)
    if problem == "laplace":
        prob: Problem = LaplaceProblem2D(mesh)
    elif problem == "poisson":
        prob = PoissonProblem2D(mesh)
    elif problem == "helmholtz":
        prob = HelmholtzProblem2D(mesh)
    elif problem == "linear":
        prob = LinearProblem2D(mesh)
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
