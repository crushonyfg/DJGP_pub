"""
Diagnostics for local projections under factorized q(W): interpret **E[W]**, **metric moments**
E[W^T W] and E[W W^T], **kernels** built from Mahalanobis expected squared distance, and
**row-normalized samples** matching `predict_vi`.

These address appendix questions that raw ||E[W]|| alone cannot: E[W] ≈ 0 does not imply
E[W^T W] ≈ 0 when posterior variance is large.

`predict_vi` applies row-wise L2 normalization to sampled W (each latent row q over input D).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def row_normalize_w_predict_vi(W: torch.Tensor) -> torch.Tensor:
    """Match `predict_vi`: W [..., Q, D] — unit L2 norm per Q row."""
    norms = torch.norm(W, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return W / norms


def svd_stats_matrix(W: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    """
    Singular values of W [Q, D]. If ||W||_F is effectively zero, set kappa and r_eff to nan
    (do not report 0 condition number — misleading).
    """
    s = torch.linalg.svdvals(W)
    s_max = float(s[0].item())
    s_min = float(s[-1].item())
    frob = float(torch.sqrt((s**2).sum()).item())
    sum_sq = (s**2).sum()
    sum_fourth = (s**4).sum()
    if frob <= eps or float(sum_fourth.item()) <= eps * eps:
        r_eff = float("nan")
    else:
        r_eff = float((sum_sq**2 / sum_fourth).item())
    if s_max <= eps:
        kappa = float("nan")
    else:
        kappa = float((s[0] / (s[-1] + eps)).item())
    return {
        "frobenius": frob,
        "spectral_norm": float(torch.linalg.matrix_norm(W, ord=2).item()),
        "tr_WWT": float((W * W).sum().item()),
        "kappa": kappa,
        "r_eff": r_eff,
        "s_max": s_max,
        "s_min": s_min,
    }


def metric_A_from_moment(mu_W: torch.Tensor, cov_W: torch.Tensor) -> torch.Tensor:
    """
    A_j = E[W_j^T W_j] under Gaussian q with given covariances per row q.
    cov_W: [T,Q,D,D] — diagonal per (t,q) gives same result as summing variances on diagonal of A.
    """
    A_mean = torch.einsum("tqd,tqe->tde", mu_W, mu_W)
    A_cov = cov_W.sum(dim=1)
    return A_mean + A_cov


def metric_B_from_moment(mu_W: torch.Tensor, cov_W: torch.Tensor) -> torch.Tensor:
    """B_j = E[W_j W_j^T] in R^{Q×Q}."""
    B_mean = torch.einsum("tqd,tpd->tqp", mu_W, mu_W)
    tr_per_q = cov_W.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    Q = mu_W.shape[1]
    idx = torch.arange(Q, device=mu_W.device)
    B = B_mean.clone()
    B[:, idx, idx] = B[:, idx, idx] + tr_per_q
    return B


def sym_matrix_stats(M: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    """Trace, effective rank (eigenvalue participation), top/min positive eigenvalue."""
    M = 0.5 * (M + M.transpose(-2, -1))
    lam = torch.linalg.eigvalsh(M)
    lam = torch.clamp(lam, min=0.0)
    tr = float(lam.sum().item())
    sum_sq = (lam**2).sum()
    if float(sum_sq.item()) <= eps * eps:
        eff_r = float("nan")
    else:
        eff_r = float((lam.sum() ** 2 / sum_sq).item())
    lam_max = float(lam.max().item()) if lam.numel() else float("nan")
    pos = lam[lam > eps]
    lam_min_pos = float(pos.min().item()) if pos.numel() > 0 else float("nan")
    return {"trace": tr, "eff_rank": eff_r, "top_eig": lam_max, "min_eig_pos": lam_min_pos}


def pairwise_sq_metric(X_nb: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """X_nb [n,D], A [D,D] SPD. sq_ij = (x_i-x_j)^T A (x_i-x_j)."""
    delta = X_nb.unsqueeze(1) - X_nb.unsqueeze(0)
    return torch.einsum("ijd,de,ije->ij", delta, A, delta)


def rbf_from_sq_dist(sq: torch.Tensor, sigma_k: float) -> torch.Tensor:
    l = max(float(sigma_k), 1e-8)
    return torch.exp(-0.5 * sq / (l**2))


def condition_numbers_kernel(
    K: torch.Tensor,
    *,
    jitter: float = 1e-6,
    sigma_noise: float | None = None,
) -> dict[str, float]:
    """cond(K), cond(K + jitter I), cond(K + jitter I + sigma_noise^2 I) — last matches noisy GP gram."""
    n = K.shape[0]
    if n <= 1:
        return {"cond_K": 1.0, "cond_K_jitter": 1.0, "cond_K_plus_noise": 1.0}
    I = torch.eye(n, device=K.device, dtype=K.dtype)
    K0 = 0.5 * (K + K.T)
    out: dict[str, float] = {}
    for name, M in (
        ("cond_K", K0),
        ("cond_K_jitter", K0 + jitter * I),
    ):
        w = torch.linalg.eigvalsh(0.5 * (M + M.T))
        w = torch.clamp(w, min=1e-18)
        out[name] = float((w.max() / w.min()).item())
    Kn = K0 + jitter * I
    if sigma_noise is not None and sigma_noise > 0:
        Kn = Kn + (sigma_noise**2) * I
    w = torch.linalg.eigvalsh(0.5 * (Kn + Kn.T))
    w = torch.clamp(w, min=1e-18)
    out["cond_K_plus_noise"] = float((w.max() / w.min()).item())
    return out


def kernel_from_metric_A(
    X_nb: torch.Tensor,
    A_j: torch.Tensor,
    sigma_k: float,
    *,
    jitter: float = 1e-6,
    sigma_noise: float | None = None,
) -> dict[str, float]:
    """Expected squared-distance RBF kernel from A_j = E[W^T W]."""
    sq = pairwise_sq_metric(X_nb, A_j)
    K = rbf_from_sq_dist(sq, sigma_k)
    return condition_numbers_kernel(K, jitter=jitter, sigma_noise=sigma_noise)


def kernel_from_projected_Z(
    X_nb: torch.Tensor,
    W: torch.Tensor,
    sigma_k: float,
    *,
    jitter: float = 1e-6,
    sigma_noise: float | None = None,
) -> dict[str, float]:
    """Legacy / Euclidean in latent after Z = X W^T (same as old projected_neighbor_stats core)."""
    Z = X_nb @ W.T
    d2 = torch.cdist(Z, Z, p=2.0).pow(2)
    K = torch.exp(-0.5 * d2 / (max(sigma_k, 1e-8) ** 2))
    return condition_numbers_kernel(K, jitter=jitter, sigma_noise=sigma_noise)


def trace_q_variance_diagonal(sigma_W: torch.Tensor) -> float:
    return float((sigma_W**2).sum().item())


def relative_uncertainty_trace(tr_var: float, frob_mu: float, eps: float = 1e-8) -> float:
    return tr_var / (max(frob_mu**2, eps))


def latent_alignment_corr(X_nb: torch.Tensor, W: torch.Tensor, z_nb: torch.Tensor) -> float | None:
    if z_nb.shape[0] != X_nb.shape[0] or z_nb.shape[0] < 2:
        return None
    Z = X_nb @ W.T
    d_proj = torch.pdist(Z, p=2.0)
    d_z = torch.pdist(z_nb, p=2.0)
    if d_proj.numel() < 2:
        return None
    d_proj_c = d_proj - d_proj.mean()
    d_z_c = d_z - d_z.mean()
    denom = (d_proj_c.std(unbiased=False) * d_z_c.std(unbiased=False)).clamp_min(1e-12)
    return float((d_proj_c * d_z_c).mean() / denom).item()


def _nanmedian(vals: list[float]) -> float | None:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return None
    if np.all(np.isnan(arr)):
        return None
    return float(np.nanmedian(arr))


def sample_row_normalized_W(
    mu_W: torch.Tensor,
    sigma_W: torch.Tensor,
    M: int,
    *,
    seed: int = 0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """[M,T,Q,D] — same sampling + row normalization as `predict_vi`."""
    g = torch.Generator(device=mu_W.device)
    g.manual_seed(seed)
    eps_smpl = torch.randn((M,) + mu_W.shape, device=mu_W.device, generator=g)
    W_s = mu_W.unsqueeze(0) + eps_smpl * sigma_W.unsqueeze(0)
    norms = torch.norm(W_s, p=2, dim=-1, keepdim=True).clamp_min(eps)
    return W_s / norms


def mc_normalized_w_spectral_and_kernels(
    mu_W: torch.Tensor,
    sigma_W: torch.Tensor,
    regions: list[dict],
    sigma_k_per_anchor: torch.Tensor,
    sigma_noise_per_anchor: torch.Tensor,
    M: int,
    *,
    seed: int = 0,
    eps: float = 1e-12,
    jitter: float = 1e-6,
) -> tuple[dict[str, float | int | None], dict[str, float | None], list[float | None]]:
    """
    Row-normalized samples: spectral summaries + local RBF kernel cond on projected Z per sample.
    Returns (summary_medians, kernel_medians_over_all_pairs, per_anchor_median_kappa_K_plus_noise).
    """
    Ws = sample_row_normalized_W(mu_W, sigma_W, M, seed=seed, eps=eps)
    T = mu_W.shape[0]
    frobs, kappas, r_effs = [], [], []
    condK, condKj, condKn = [], [], []
    per_j_cond_n: list[list[float]] = [[] for _ in range(T)]

    for m in range(M):
        for j in range(T):
            st = svd_stats_matrix(Ws[m, j], eps=eps)
            frobs.append(st["frobenius"])
            kappas.append(st["kappa"])
            r_effs.append(st["r_eff"])

            X_nb = regions[j]["X"]
            if X_nb.dim() != 2:
                X_nb = X_nb.view(X_nb.shape[0], -1)
            sk = float(sigma_k_per_anchor[j].detach().item())
            sn = float(sigma_noise_per_anchor[j].detach().item())
            kk = kernel_from_projected_Z(X_nb, Ws[m, j], sk, jitter=jitter, sigma_noise=sn)
            condK.append(kk["cond_K"])
            condKj.append(kk["cond_K_jitter"])
            kn = kk["cond_K_plus_noise"]
            condKn.append(kn)
            per_j_cond_n[j].append(kn)

    summary = {
        "median_frob": _nanmedian(frobs),
        "median_kappa": _nanmedian(kappas),
        "median_eff_rank": _nanmedian(r_effs),
        "M": M,
    }
    kernel_medians = {
        "median_kappa_K": _nanmedian(condK),
        "median_kappa_K_jitter": _nanmedian(condKj),
        "median_kappa_K_plus_noise": _nanmedian(condKn),
    }
    per_anchor_kn = [_nanmedian(lst) for lst in per_j_cond_n]
    return summary, kernel_medians, per_anchor_kn


def compute_full_projection_diagnostics(
    mu_W: torch.Tensor,
    sigma_W: torch.Tensor,
    cov_W: torch.Tensor,
    regions: list[dict],
    sigma_k_per_anchor: torch.Tensor,
    sigma_noise_per_anchor: torch.Tensor,
    *,
    z_neighbors_list: list[torch.Tensor | None] | None = None,
    mc_M: int = 32,
    mc_seed: int = 0,
    eps: float = 1e-12,
    kernel_jitter: float = 1e-6,
) -> dict[str, Any]:
    """
    Full nested diagnostic dict for JSON export (see experiments/diagnostics README).
    """
    device = mu_W.device
    T, Q, D = mu_W.shape
    A = metric_A_from_moment(mu_W, cov_W)
    B = metric_B_from_moment(mu_W, cov_W)

    # --- per-anchor rows ---
    per_anchor: list[dict[str, Any]] = []
    # lists for medians
    ew_frob, ew_kappa, ew_r_eff = [], [], []
    tr_A, eff_A, eff_B, top_B, min_B = [], [], [], [], []
    k_met_raw, k_met_j, k_met_n = [], [], []
    z_neighbors_list = z_neighbors_list or [None] * T

    for j in range(T):
        Wm = mu_W[j]
        st_ew = svd_stats_matrix(Wm, eps=eps)
        trv = trace_q_variance_diagonal(sigma_W[j])
        rel = relative_uncertainty_trace(trv, st_ew["frobenius"], eps=eps)
        Wn = row_normalize_w_predict_vi(Wm.unsqueeze(0)).squeeze(0)
        st_n = svd_stats_matrix(Wn, eps=eps)

        st_a = sym_matrix_stats(A[j], eps=eps)
        st_b = sym_matrix_stats(B[j], eps=eps)

        X_nb = regions[j]["X"]
        if X_nb.dim() != 2:
            X_nb = X_nb.view(X_nb.shape[0], -1)
        sk = float(sigma_k_per_anchor[j].item())
        sn = float(sigma_noise_per_anchor[j].item())

        km = kernel_from_metric_A(X_nb, A[j], sk, jitter=kernel_jitter, sigma_noise=sn)

        kk_z = kernel_from_projected_Z(X_nb, Wm, sk, jitter=kernel_jitter, sigma_noise=sn)

        row: dict[str, Any] = {
            "anchor": j,
            "EW_raw": {
                "frob": st_ew["frobenius"],
                "kappa": st_ew["kappa"],
                "eff_rank": st_ew["r_eff"],
                "tr_Var_W": trv,
                "rel_uncertainty": rel,
            },
            "EW_row_normalized_like_predict_vi": {
                "kappa": st_n["kappa"],
                "eff_rank": st_n["r_eff"],
            },
            "metric_moment_matrices": {
                "trace_E_WtW_A": st_a["trace"],
                "eff_rank_A": st_a["eff_rank"],
                "trace_E_WWt_B": st_b["trace"],
                "eff_rank_B": st_b["eff_rank"],
                "top_eig_B": st_b["top_eig"],
                "min_eig_B_positive": st_b["min_eig_pos"],
            },
            "kernel_metric_A_expected_sqdist": km,
            "kernel_projected_Z_from_EW_only": kk_z,
        }

        if z_neighbors_list[j] is not None:
            z_nb = z_neighbors_list[j]
            if z_nb.device != device:
                z_nb = z_nb.to(device)
            c = latent_alignment_corr(X_nb, Wm, z_nb)
            if c is not None:
                row["pairwise_dist_corr_true_z"] = c

        per_anchor.append(row)

        ew_frob.append(st_ew["frobenius"])
        ew_kappa.append(st_ew["kappa"])
        ew_r_eff.append(st_ew["r_eff"])
        tr_A.append(st_a["trace"])
        eff_A.append(st_a["eff_rank"])
        eff_B.append(st_b["eff_rank"])
        top_B.append(st_b["top_eig"])
        min_B.append(st_b["min_eig_pos"])
        k_met_raw.append(km["cond_K"])
        k_met_j.append(km["cond_K_jitter"])
        k_met_n.append(km["cond_K_plus_noise"])

    EW_raw = {
        "median_frob": _nanmedian(ew_frob),
        "median_kappa": _nanmedian(ew_kappa),
        "median_eff_rank": _nanmedian(ew_r_eff),
    }
    metric_block = {
        "median_trace_E_WtW": _nanmedian(tr_A),
        "median_eff_rank_A": _nanmedian(eff_A),
        "median_eff_rank_B": _nanmedian(eff_B),
        "median_top_eig_B": _nanmedian(top_B),
        "median_min_eig_B_positive": _nanmedian(min_B),
    }
    kernel_metric_A = {
        "median_kappa_K": _nanmedian(k_met_raw),
        "median_kappa_K_jitter": _nanmedian(k_met_j),
        "median_kappa_K_plus_noise": _nanmedian(k_met_n),
        "note": "RBF from sq Mahalanobis delta^T E[W^T W] delta (not from E[W] projection alone).",
    }

    samp_spec, samp_kern, samp_kappa_per_anchor = mc_normalized_w_spectral_and_kernels(
        mu_W,
        sigma_W,
        regions,
        sigma_k_per_anchor,
        sigma_noise_per_anchor,
        mc_M,
        seed=mc_seed,
        eps=eps,
        jitter=kernel_jitter,
    )

    for j, row in enumerate(per_anchor):
        row["kernel_sampled_row_normalized_local"] = {
            "median_kappa_K_plus_noise_over_mc": samp_kappa_per_anchor[j],
        }

    out: dict[str, Any] = {
        "EW_raw": EW_raw,
        "metric_E_WtW": metric_block,
        "kernel_metric_A": kernel_metric_A,
        "sampled_W_row_normalized": samp_spec,
        "kernel_sampled_row_normalized": samp_kern,
        "per_anchor": per_anchor,
        "notes": {
            "EW_raw": "If frob≈0, kappa/eff_rank are nan — not evidence that metric E[W^T W] is zero.",
            "kernel_blocks": "kernel_metric_A uses delta^T E[W^T W] delta. kernel_projected_Z_from_EW_only uses Z=X E[W]^T (can degenerate when E[W]≈0).",
            "prediction_aligned": "sampled_W_row_normalized + kernel_sampled_row_normalized match predict_vi.",
        },
    }

    return out


def attach_abs_errors(per_anchor: list[dict], abs_err_np: np.ndarray) -> None:
    for row in per_anchor:
        j = row["anchor"]
        row["abs_error"] = float(abs_err_np[j])


def flatten_per_anchor_for_export(per_anchor: list[dict]) -> list[dict[str, Any]]:
    """Narrow per-anchor dict for tables / scatter plots."""
    flat: list[dict[str, Any]] = []
    for row in per_anchor:
        ew = row["EW_raw"]
        met = row["metric_moment_matrices"]
        km = row["kernel_metric_A_expected_sqdist"]
        kz = row["kernel_projected_Z_from_EW_only"]
        ks = row.get("kernel_sampled_row_normalized_local") or {}
        flat.append(
            {
                "anchor": row["anchor"],
                "abs_error": row.get("abs_error"),
                "kappa_EW_raw": ew.get("kappa"),
                "eff_rank_EW_raw": ew.get("eff_rank"),
                "trace_E_WtW_A": met.get("trace_E_WtW_A"),
                "eff_rank_E_WWt_B": met.get("eff_rank_B"),
                "kappa_K_metric_plus_noise": km.get("cond_K_plus_noise"),
                "kappa_K_sampled_norm_plus_noise": ks.get("median_kappa_K_plus_noise_over_mc"),
                "kappa_K_EW_Z_plus_noise": kz.get("cond_K_plus_noise"),
            }
        )
    return flat


def sanitize_for_json(obj: Any) -> Any:
    """Replace nan/inf with None for strict JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj
