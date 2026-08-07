"""Third-party cross-check solver: Allen-Cahn via py-pde.

Adapted from HeterogeneousDiffusionPDE in phase0_phase1_local_pipeline.py
(L332). This is deliberately a completely independent code path (different
library, finite-difference spatial discretization, adaptive scipy time
integration) used only as a spot check on a handful of samples - it is far
slower than the in-house spectral solvers.
"""

import numpy as np
import pde


class AllenCahnPDE(pde.PDEBase):
    """
    Allen-Cahn with spatially varying mobility and tilted double well:

        du/dt = M(x) * (eps^2 lap(u) + u - u^3 + g)

    Inputs:
        mobility_field: pde.ScalarField, M(x) >= 0.
        eps_param: float, Allen-Cahn epsilon.
        g: float, well tilt.
        bc: py-pde boundary condition specifier.
    """

    def __init__(
        self,
        mobility_field: pde.ScalarField,
        eps_param: float,
        g: float = 0.0,
        bc: str = "periodic",
    ) -> None:
        super().__init__()
        self.mobility = mobility_field
        self.eps_param = eps_param
        self.g = g
        self.bc = bc

    def evolution_rate(self, state: pde.ScalarField, t: float = 0) -> pde.ScalarField:
        """
        Compute du/dt.

        Inputs:
            state: pde.ScalarField, shape (nx, ny).
            t: float, time.

        Outputs:
            rate: pde.ScalarField, shape (nx, ny).
        """
        lap_u = state.laplace(bc=self.bc)
        reaction = state.data - state.data**3 + self.g
        rate_data = self.mobility.data * (self.eps_param**2 * lap_u.data + reaction)
        return pde.ScalarField(state.grid, rate_data)


def solve_allen_cahn_pypde(
    u0: np.ndarray,
    eps_param: float,
    t_final: float,
    dt: float,
    domain_size: float = 1.0,
    mobility: np.ndarray = None,
    g: float = 0.0,
    solver_name: str = "scipy",
) -> np.ndarray:
    """
    Solve one Allen-Cahn case with py-pde (single sample, not batched).

    Inputs:
        u0: np.ndarray, shape (nx, ny), initial condition.
        eps_param: float, Allen-Cahn epsilon.
        t_final: float, integration horizon.
        dt: float, solver control interval.
        domain_size: float, square periodic domain side length.
        mobility: np.ndarray, shape (nx, ny), M(x); None means M = 1.
        g: float, well tilt.
        solver_name: str, py-pde solver name.

    Outputs:
        u_final: np.ndarray, shape (nx, ny), float32.
    """
    nx, ny = u0.shape

    grid = pde.CartesianGrid(
        [[0.0, domain_size], [0.0, domain_size]],
        [nx, ny],
        periodic=True,
    )

    m_data = np.ones_like(u0, dtype=np.float64) if mobility is None else mobility
    mobility_field = pde.ScalarField(grid, m_data)
    u0_field = pde.ScalarField(grid, u0)

    equation = AllenCahnPDE(
        mobility_field=mobility_field,
        eps_param=eps_param,
        g=g,
        bc="periodic",
    )

    u_final_field = equation.solve(
        u0_field,
        t_range=t_final,
        dt=dt,
        solver=solver_name,
        tracker=None,
    )

    return u_final_field.data.astype(np.float32)
