"""Diagnostics for LMJGP / DJGP (projection posterior health, geometry)."""

from djgp.diagnostics.projection_posterior import (
    attach_abs_errors,
    compute_full_projection_diagnostics,
    flatten_per_anchor_for_export,
    mc_normalized_w_spectral_and_kernels,
    metric_A_from_moment,
    metric_B_from_moment,
    sanitize_for_json,
)
from djgp.diagnostics.elbo_gradients import (
    compute_lmjgp_elbo_parts,
    compute_lmjgp_qv_gradient_diagnostics,
)

__all__ = [
    "attach_abs_errors",
    "compute_full_projection_diagnostics",
    "compute_lmjgp_elbo_parts",
    "compute_lmjgp_qv_gradient_diagnostics",
    "flatten_per_anchor_for_export",
    "mc_normalized_w_spectral_and_kernels",
    "metric_A_from_moment",
    "metric_B_from_moment",
    "sanitize_for_json",
]
