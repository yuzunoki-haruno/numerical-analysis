# Change Log

## 0.0.3 (2026/01/03)

- Refactor the Mesh class, the FEM class, example codes, and test codes.
    - `numerical_analysis.discretization.PolygonMesh`
    - `numerical_analysis.discretization.LineMesh`
    - `numerical_analysis.fem.Fem1d`
    - `numerical_analysis.fem.Fem2d`
    - `example/fem/fem1d.py`, `example/fem/fem2d.py`, etc.
    - `tests/**`

## 0.0.2 (2025/12/18)

- Implemented code to numerically solve two-dimensional boundary-value problems using the FEM.
    - `numerical_analysis.fem.Fem2d`, `numerical_analysis.discretization.PolygonMesh`, etc.

## 0.0.1 (2025/10/26)

- The repository has been initialized.
- Implemented code to numerically solve one-dimensional boundary-value problems using the FEM.
    - `numerical_analysis.fem.Fem1d`, `numerical_analysis.discretization.LineMesh`, etc.
