"""fnocausal: Phases 2-5 of the FNO material-degradation project.

Subpackages:
    common   -- config, seeding, GRF sampling, normalization, metrics, IO.
    sim      -- Allen-Cahn solvers (spectral IMEX, RK4 validator, py-pde
                cross-check), resampling, dataset generation.
    models   -- FNO surrogate, convolutional autoencoder, training loops.
    analysis -- solver verification, surrogate evaluation, causal audit.

Claims discipline (see experiments/phase2_problem_statement.md):
    claim1 -- the FNO is an accurate surrogate for the numerical solver.
    claim2 -- the audit separates causal from spurious features.
    claim3 -- the PDE describes real degradation (NOT claimed in this project).
"""
