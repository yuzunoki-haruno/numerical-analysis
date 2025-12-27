import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from example.fem.problem import PoissonProblem, LinearProblem, LaplaceProblem, HelmholtzProblem
from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh1d import generate_line_mesh, MeshType
from numerical_analysis.fem import Fem1d
from numerical_analysis.util import metrics

CASES = (11, 21, 81)

XMIN, XMAX = -0.5, 1.0

CONDITIONS_LIST = [
    [Condition.DIRICHLET, Condition.DIRICHLET],
    [Condition.DIRICHLET, Condition.NEUMANN],
    [Condition.NEUMANN, Condition.DIRICHLET],
]


class TestFem1D:

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_laplace(self, conditions: list[Condition]) -> None:
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh_type = MeshType.FirstOrder
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, mesh_type)
            fem = Fem1d(mesh)

            prob = LaplaceProblem(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert relative_error < 10 ** (-8)

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_poisson(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh_type = MeshType.FirstOrder
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, mesh_type)
            fem = Fem1d(mesh)

            prob = PoissonProblem(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_helmholtz(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh_type = MeshType.FirstOrder
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, mesh_type)
            fem = Fem1d(mesh)

            prob = HelmholtzProblem(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_linear(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh_type = MeshType.FirstOrder
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, mesh_type)
            fem = Fem1d(mesh)

            prob = LinearProblem(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert relative_error < error_old / 2
            error_old = relative_error


class TestFem1DHighOrder:

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_laplace(self, conditions: list[Condition]) -> None:
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh_type = MeshType.SecondOrder
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, mesh_type)
            fem = Fem1d(mesh)

            prob = LaplaceProblem(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert relative_error < 10 ** (-8)

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_poisson(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh_type = MeshType.SecondOrder
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, mesh_type)
            fem = Fem1d(mesh)

            prob = PoissonProblem(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_helmholtz(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh_type = MeshType.SecondOrder
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, mesh_type)
            fem = Fem1d(mesh)

            prob = HelmholtzProblem(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_linear(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh_type = MeshType.SecondOrder
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, mesh_type)
            fem = Fem1d(mesh)

            prob = LinearProblem(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert relative_error < error_old / 2
            error_old = relative_error
