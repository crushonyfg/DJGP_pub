"""Self-contained smoke test for analytic ISM-LMJGP.

Imports ONLY the installed `djgp` package (no experiment-tree helpers), builds a
tiny deterministic problem, trains a few steps and predicts. Runnable as a script
to print a reproducible summary used for the DJGP / DJGP_ consistency check.
"""
from __future__ import annotations

import numpy as np
import torch

from djgp.projections.structured_metric_lmjgp import (
    AnalyticStructuredMetricLMJGPConfig,
    build_fixed_local_inducing_from_w0,
    predict_analytic_structured_metric_lmjgp,
    train_analytic_structured_metric_lmjgp,
)


def _knn(Xte, Xtr, ytr, k):
    d = torch.cdist(Xte, Xtr)
    idx = torch.topk(d, k, largest=False).indices
    return Xtr[idx], ytr[idx]


def run(seed=0, steps=8, noise_mode="data_driven"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cpu")
    D, Ntr, T, Q, n = 8, 200, 30, 3, 15
    g = torch.Generator(device="cpu")
    g.manual_seed(123)
    Xtr = torch.randn(Ntr, D, generator=g, dtype=torch.float64)
    w_true = torch.randn(D, generator=g, dtype=torch.float64)
    ytr = (Xtr @ w_true).sin() + 0.1 * torch.randn(Ntr, generator=g, dtype=torch.float64)
    Xte = torch.randn(T, D, generator=g, dtype=torch.float64)
    Xtr, ytr, Xte = Xtr.to(dev), ytr.to(dev), Xte.to(dev)
    Xn, yn = _knn(Xte, Xtr, ytr, n)
    W0 = torch.randn(Q, D, generator=g, dtype=torch.float64).to(dev)
    W0 = W0 / W0.norm(dim=1, keepdim=True)
    C, mu_u0 = build_fixed_local_inducing_from_w0(Xte, Xn, yn, W0, m_inducing=8)
    extra = {} if noise_mode == "fixed" else dict(obs_noise_prior_mode="data_driven", init_noise_at_prior=True)
    cfg = AnalyticStructuredMetricLMJGPConfig(
        steps=steps, lr=0.01, batch_anchors=T, trace_target=float(Q), trace_normalize=True,
        eval_interval=steps, restore_state="final", seed=seed, **extra)
    res = train_analytic_structured_metric_lmjgp(Xte, Xn, yn, C, init_W=W0, init_mu_u=mu_u0, config=cfg)
    mu, var, _ = predict_analytic_structured_metric_lmjgp(Xte, res, trace_target=float(Q), trace_normalize=True)
    return float(mu.sum().item()), float(var.sum().item()), float(res.diagnostics.get("best_loss", float("nan")))


def test_smoke_ism_lmjgp():
    for mode in ("fixed", "data_driven"):
        mu_sum, var_sum, loss = run(noise_mode=mode)
        assert np.isfinite(mu_sum) and np.isfinite(var_sum) and np.isfinite(loss)
        assert var_sum > 0


if __name__ == "__main__":
    import djgp
    print(f"DJGP_PKG {djgp.__file__}")
    for mode in ("fixed", "data_driven"):
        mu_sum, var_sum, loss = run(noise_mode=mode)
        print(f"SUMMARY[{mode}] mu_sum={mu_sum:.10f} var_sum={var_sum:.10f} loss={loss:.10f}")
