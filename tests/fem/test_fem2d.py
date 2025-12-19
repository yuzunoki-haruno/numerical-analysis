import itertools

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from numerical_analysis.discretization import Condition
from numerical_analysis.discretization.mesh2d import generate_polygon_mesh
from numerical_analysis.fem import Fem2d


CASES = ((11, 11), (41, 41))
CASES_HElMHOLTZ = ((21, 21), (41, 41))
XMIN, XMAX, YMIN, YMAX = -1.125, -0.125, 0.125, 1.125


class TestFem2D:

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_laplace(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4)
            fem = Fem2d(mesh)

            u = np.sin(2.0 * np.pi * mesh.x) * np.sinh(2.0 * np.pi * mesh.y)
            gx = 2.0 * np.pi * np.cos(2.0 * np.pi * mesh.x) * np.sinh(2.0 * np.pi * mesh.y)
            gy = 2.0 * np.pi * np.sin(2.0 * np.pi * mesh.x) * np.cosh(2.0 * np.pi * mesh.y)
            g = np.vstack((gx, gy)).T
            f = np.zeros_like(u)
            coefficient = fem.laplacian_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_poisson(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4)
            fem = Fem2d(mesh)

            u = np.sin(2.0 * np.pi * mesh.x) * np.cos(2.0 * np.pi * mesh.y)
            gx = 2.0 * np.pi * np.cos(2.0 * np.pi * mesh.x) * np.cos(2.0 * np.pi * mesh.y)
            gy = -2.0 * np.pi * np.sin(2.0 * np.pi * mesh.x) * np.sin(2.0 * np.pi * mesh.y)
            g = np.vstack((gx, gy)).T
            f = 8.0 * np.pi**2 * np.sin(2.0 * np.pi * mesh.x) * np.cos(2.0 * np.pi * mesh.y)
            coefficient = fem.laplacian_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_helmholtz(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES_HElMHOLTZ:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4)
            fem = Fem2d(mesh)

            u = np.sin(2.0 * np.pi * mesh.x) * np.cos(2.0 * np.pi * mesh.y)
            gx = 2.0 * np.pi * np.cos(2.0 * np.pi * mesh.x) * np.cos(2.0 * np.pi * mesh.y)
            gy = -2.0 * np.pi * np.sin(2.0 * np.pi * mesh.x) * np.sin(2.0 * np.pi * mesh.y)
            g = np.vstack((gx, gy)).T
            f = np.zeros_like(u)
            coefficient = fem.laplacian_matrix
            coefficient += -8.0 * np.pi**2 * fem.term_matrix
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old
            error_old = relative_error

    @pytest.mark.parametrize("conditions", itertools.product([Condition.DIRICHLET, Condition.NEUMANN], repeat=4))
    def test_linear(self, conditions: list[Condition]) -> None:
        error_old = 1
        for n_x, n_y in CASES:
            xmin, xmax = XMIN, XMAX
            ymin, ymax = YMIN, YMAX
            c1, c2, c3, c4 = conditions
            mesh = generate_polygon_mesh(n_x, xmin, xmax, n_y, ymin, ymax, c1, c2, c3, c4)
            fem = Fem2d(mesh)

            u = np.sin(2.0 * np.pi * mesh.x) * np.sin(2.0 * np.pi * mesh.y)
            gx = 2.0 * np.pi * np.cos(2.0 * np.pi * mesh.x) * np.sin(2.0 * np.pi * mesh.y)
            gy = 2.0 * np.pi * np.sin(2.0 * np.pi * mesh.x) * np.cos(2.0 * np.pi * mesh.y)
            g = np.vstack((gx, gy)).T
            f = 8.0 * np.pi**2 * np.sin(2.0 * np.pi * mesh.x) * np.sin(2.0 * np.pi * mesh.y)
            f -= 2.0 * np.pi * np.cos(2.0 * np.pi * mesh.x) * np.sin(2.0 * np.pi * mesh.y) / 3.0
            f -= 2.0 * np.pi * np.sin(2.0 * np.pi * mesh.x) * np.cos(2.0 * np.pi * mesh.y) / 5.0
            coefficient = fem.laplacian_matrix
            coefficient += fem.differential_matrix[0] / 3.0
            coefficient += fem.differential_matrix[1] / 5.0
            rhs = fem.term(f)

            fem.implement_dirichlet(coefficient, rhs, u)
            fem.implement_neumann(rhs, g)

            sol = splu(csc_matrix(coefficient)).solve(rhs)
            relative_error = np.max(np.abs(u - sol) / np.abs(u).max())
            assert relative_error < error_old
            error_old = relative_error
