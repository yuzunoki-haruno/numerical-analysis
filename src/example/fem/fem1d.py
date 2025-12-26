from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh1d import generate_line_mesh
from numerical_analysis.fem import Fem1d
from numerical_analysis.util import metrics, vis

from example.fem.problem import PoissonProblem, LinearProblem, LaplaceProblem, HelmholtzProblem, Problem

NAME_PLOT = "result.png"
NAME_TXT = "result.txt"


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="1D Finite Element Analysis.")
    parser.add_argument("--xmin", type=float, default=0.0, help="Minimum of x-axis.")
    parser.add_argument("--xmax", type=float, default=1.0, help="Maximum of x-axis.")
    parser.add_argument("--n_nodes", type=int, default=51, help="Number of nodes.")
    parser.add_argument("--condition", type=str, nargs=2, default=("d", "d"), help="Boundary condition.")
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
        prob: Problem = LaplaceProblem(mesh)
    elif problem == "poisson":
        prob = PoissonProblem(mesh)
    elif problem == "helmholtz":
        prob = HelmholtzProblem(mesh)
    elif problem == "linear":
        prob = LinearProblem(mesh)
    else:
        raise NameError("This program can numerically solve `Laplace`, `Poisson`, `Helmholtz`, and `Linear` problems.")
    coefficient, rhs = prob.formulate(fem)
    fem.implement_dirichlet(coefficient, rhs, prob.u)
    fem.implement_neumann(rhs, prob.g)

    # solving FE linear system.
    sol = splu(csc_matrix(coefficient)).solve(rhs)
    rerror = metrics.relative_error(sol, prob.u)
    print(f"Relative Error: {rerror:.5e}")

    # plot result.
    filename = output_dir / NAME_PLOT
    vis.plot1d(filename, mesh.x, sol, prob.u)

    # write relative error
    with open(output_dir / NAME_TXT, mode="w") as file:
        file.write(f"{problem.title()} problem\n")
        file.write(f"{mesh}\n")
        file.write(f"Relative Error: {rerror}\n")


if __name__ == "__main__":
    main()
