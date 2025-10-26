import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh1d import generate_line_mesh
from numerical_analysis.fem import Fem1d


CASES = (11, 21, 81)


class TestFem1D:

    conditions_list = [
        [Condition.DIRICHLET, Condition.DIRICHLET],
        [Condition.DIRICHLET, Condition.NEUMANN],
        [Condition.NEUMANN, Condition.DIRICHLET],
    ]

    @pytest.mark.parametrize("conditions", conditions_list)
    def test_laplace(self, conditions: list[Condition]) -> None:
        for n in CASES:
            n_nodes, xmin, xmax = n, -1, -1.5
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax)
            fem = Fem1d(mesh)

            coef, intercept = 2.0, 1.0
            u = coef * mesh.x + intercept
            g = coef * np.ones_like(mesh.x)
            f = np.zeros_like(mesh.x)
            coefficient = fem.laplacian_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < 10 ** (-8)

    @pytest.mark.parametrize("conditions", conditions_list)
    def test_poisson(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n in CASES:
            n_nodes, xmin, xmax = n, -1, -1.5
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax)
            fem = Fem1d(mesh)

            coef = 2.0 * np.pi
            coef_x = coef * mesh.x
            u = np.sin(coef_x)
            g = np.cos(coef_x) * coef
            f = np.sin(coef_x) * coef**2
            coefficient = fem.laplacian_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", conditions_list)
    def test_helmholtz(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n in CASES:
            n_nodes, xmin, xmax = n, -1, -1.5
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax)
            fem = Fem1d(mesh)

            coef = 2.0 * np.pi
            coef_x = coef * mesh.x
            u = np.cos(coef_x)
            g = -np.sin(coef_x) * coef
            f = np.zeros_like(mesh.x)
            coefficient = fem.laplacian_matrix
            coefficient -= (coef**2) * fem.term_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", conditions_list)
    def test_linear(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n in CASES:
            n_nodes, xmin, xmax = n, -1, -1.5
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax)
            fem = Fem1d(mesh)

            two_x = 2 * mesh.x
            coef = np.exp(np.pi / 2)
            u = np.exp(-two_x) * (np.cos(two_x) + coef * np.sin(two_x))
            g = np.exp(-two_x) * (2 * (coef - 1) * np.cos(two_x) - 2 * (coef + 1) * np.sin(two_x))
            coefficient = fem.laplacian_matrix
            coefficient -= 4 * fem.differential_matrix
            coefficient -= 8 * fem.term_matrix
            rhs = np.zeros_like(mesh.x)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old / 2
            error_old = relative_error


class TestFem1DHighOrder:

    conditions_list = [
        [Condition.DIRICHLET, Condition.DIRICHLET],
        [Condition.DIRICHLET, Condition.NEUMANN],
        [Condition.NEUMANN, Condition.DIRICHLET],
    ]

    @pytest.mark.parametrize("conditions", conditions_list)
    def test_laplace(self, conditions: list[Condition]) -> None:
        for n in CASES:
            n_nodes, xmin, xmax = n, -1, -1.5
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax, mesh_type=3)
            fem = Fem1d(mesh)

            coef, intercept = 2.0, 1.0
            u = coef * mesh.x + intercept
            g = coef * np.ones_like(mesh.x)
            f = np.zeros_like(mesh.x)
            coefficient = fem.laplacian_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < 10 ** (-8)

    @pytest.mark.parametrize("conditions", conditions_list)
    def test_poisson(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n in CASES:
            n_nodes, xmin, xmax = n, -1, -1.5
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax, mesh_type=3)
            fem = Fem1d(mesh)

            coef = 2.0 * np.pi
            coef_x = coef * mesh.x
            u = np.sin(coef_x)
            g = np.cos(coef_x) * coef
            f = np.sin(coef_x) * coef**2
            coefficient = fem.laplacian_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", conditions_list)
    def test_helmholtz(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n in CASES:
            n_nodes, xmin, xmax = n, -1, -1.5
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax, mesh_type=3)
            fem = Fem1d(mesh)

            coef = 2.0 * np.pi
            coef_x = coef * mesh.x
            u = np.cos(coef_x)
            g = -np.sin(coef_x) * coef
            f = np.zeros_like(mesh.x)
            coefficient = fem.laplacian_matrix
            coefficient -= (coef**2) * fem.term_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old / 2
            error_old = relative_error

    @pytest.mark.parametrize("conditions", conditions_list)
    def test_linear(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n in CASES:
            n_nodes, xmin, xmax = n, -1, -1.5
            cmin, cmax = conditions
            mesh = generate_line_mesh(n_nodes, xmin, xmax, cmin, cmax, mesh_type=3)
            fem = Fem1d(mesh)

            two_x = 2 * mesh.x
            coef = np.exp(np.pi / 2)
            u = np.exp(-two_x) * (np.cos(two_x) + coef * np.sin(two_x))
            g = np.exp(-two_x) * (2 * (coef - 1) * np.cos(two_x) - 2 * (coef + 1) * np.sin(two_x))
            coefficient = fem.laplacian_matrix
            coefficient -= 4 * fem.differential_matrix
            coefficient -= 8 * fem.term_matrix
            rhs = np.zeros_like(mesh.x)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old / 2
            error_old = relative_error
