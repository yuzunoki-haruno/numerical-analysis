import itertools

import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from example.fem.problem import PoissonProblem2D, LinearProblem2D, LaplaceProblem2D, HelmholtzProblem2D
from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh2d import MeshType, generate_polygon_mesh
from numerical_analysis.fem import Fem2d
from numerical_analysis.util import metrics


CASES = ((11, 11), (41, 41))
CASES_HElMHOLTZ = ((21, 21), (41, 41))
XMIN, XMAX, YMIN, YMAX = -1.25, -0.25, 0.5, 1.5


class TestFem2dTriangle:

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_laplace(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4, MeshType.Triangle)
            fem = Fem2d(mesh)

            prob = LaplaceProblem2D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.Triangle
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_poisson(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4, MeshType.Triangle)
            fem = Fem2d(mesh)

            prob = PoissonProblem2D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.Triangle
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_helmholtz(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES_HElMHOLTZ:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4, MeshType.Triangle)
            fem = Fem2d(mesh)

            prob = HelmholtzProblem2D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.Triangle
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_linear(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4, MeshType.Triangle)
            fem = Fem2d(mesh)

            prob = LinearProblem2D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.Triangle
            assert relative_error < error_old
            error_old = relative_error


class TestFem2dRectangle:

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_laplace(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4, MeshType.Rectangle)
            fem = Fem2d(mesh)

            prob = LaplaceProblem2D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.Rectangle
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_poisson(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4, MeshType.Rectangle)
            fem = Fem2d(mesh)

            prob = PoissonProblem2D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.Rectangle
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_helmholtz(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES_HElMHOLTZ:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4, MeshType.Rectangle)
            fem = Fem2d(mesh)

            prob = HelmholtzProblem2D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.Rectangle
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_linear(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4, MeshType.Rectangle)
            fem = Fem2d(mesh)

            prob = LinearProblem2D(mesh)
            coefficient, rhs = prob.formulate(fem)
            fem.implement_dirichlet(coefficient, rhs, prob.u)
            fem.implement_neumann(rhs, prob.g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = metrics.relative_error(sol, prob.u)
            assert mesh.mesh_type == MeshType.Rectangle
            assert relative_error < error_old
            error_old = relative_error
