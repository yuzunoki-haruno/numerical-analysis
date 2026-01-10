import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from example.fem.problem import PoissonProblem1D, LinearProblem1D, LaplaceProblem1D, HelmholtzProblem1D
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


class TestFem1dFirstOrder:

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_laplace(self, conditions: list[Condition]) -> None:
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, MeshType.FirstOrder)
            fem = Fem1d(mesh)

            prob = LaplaceProblem1D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.FirstOrder
            assert relative_error < 10 ** (-8)

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_poisson(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, MeshType.FirstOrder)
            fem = Fem1d(mesh)

            prob = PoissonProblem1D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.FirstOrder
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_helmholtz(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, MeshType.FirstOrder)
            fem = Fem1d(mesh)

            prob = HelmholtzProblem1D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.FirstOrder
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_linear(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, MeshType.FirstOrder)
            fem = Fem1d(mesh)

            prob = LinearProblem1D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.FirstOrder
            assert relative_error < error_old / 2
            error_old = relative_error


class TestFem1dSecondOrder:

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_laplace(self, conditions: list[Condition]) -> None:
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, MeshType.SecondOrder)
            fem = Fem1d(mesh)

            prob = LaplaceProblem1D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.SecondOrder
            assert relative_error < 10 ** (-8)

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_poisson(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, MeshType.SecondOrder)
            fem = Fem1d(mesh)

            prob = PoissonProblem1D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.SecondOrder
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_helmholtz(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, MeshType.SecondOrder)
            fem = Fem1d(mesh)

            prob = HelmholtzProblem1D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.SecondOrder
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", CONDITIONS_LIST)
    def test_linear(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_nodes in CASES:
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, XMIN, XMAX, cmin, cmax, MeshType.SecondOrder)
            fem = Fem1d(mesh)

            prob = LinearProblem1D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.SecondOrder
            assert relative_error < error_old / 2
            error_old = relative_error
