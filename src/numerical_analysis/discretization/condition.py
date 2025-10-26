from enum import StrEnum


class Condition(StrEnum):
    """A name of boundary conditions.

    Attributes:
        DIRICHLET (str): Dirichlet boundary condition.
        NEUMANN (str): Neumann boundary condition

    """

    DIRICHLET = "Dirichlet"
    NEUMANN = "Neumann"
