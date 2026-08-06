"""Run uncertain-W CEM prediction-head ablations on paper L2/LH data.

How to run from the repository root with conda env ``jumpGP``:

    conda activate jumpGP
    cd C:\\Users\\yxu59\\files\\spring2026\\park\\DJGP

    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants fixed_label_hyperopt_kcov0,self_cem1_kcov0.05,self_cem1_kcov0.1 ^
        --max_test 200 ^
        --out_dir experiments/synthetic/uncertain_w_cem_l2_lh_seed0_20260514

    # Faster smoke:
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings l2_q2_train1000 ^
        --variants fixed_label_hyperopt_kcov0,self_cem1_kcov0.05 ^
        --max_test 10 --hyper_steps 2 --gate_steps 2 ^
        --out_dir experiments/synthetic/uncertain_w_cem_smoke

    # Alternating q(R)/C-step UQ controls:
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants qr_alt_refresh10_self_cem2_nstd0.1_damp0.5_kcov0.1 ^
        --max_test 200 --m_inducing 40 --qr_outer_cycles 4 ^
        --out_dir experiments/synthetic/uncertain_w_cem_qr_alt_uqfix_seed0

    # Learned-qR controls:
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants qr_alt_refresh10_self_cem2_nstd0.1_kcov0.1,qr_alt_refresh10_self_cem2_nstd0.1_qrprior_kcov0.1,qr_alt_refresh10_self_cem2_nstd0.1_qrmeanonly_kcov0.1,qr_alt_refresh10_self_cem2_nstd0.1_qrshuf_kcov0.1,self_cem2_nstd0.1_kcov0.1 ^
        --max_test 200 --m_inducing 40 --qr_steps 40 --qr_outer_cycles 4 ^
        --out_dir experiments/synthetic/uncertain_w_cem_qr_controls_seed0

    # Low-rank residual q(R): W(x)=W0+C(x)V^T, anchor-conditioned.
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants qr_alt_lr1_refresh10_self_cem2_nstd0.1_kcov0.1,qr_alt_lr2_refresh10_self_cem2_nstd0.1_kcov0.1,qr_alt_lr4_refresh10_self_cem2_nstd0.1_kcov0.1 ^
        --max_test 200 --m_inducing 40 --qr_steps 40 --qr_outer_cycles 4 ^
        --lowrank_basis pca_residual --lowrank_qr_w_signal_var 0.1 ^
        --out_dir experiments/synthetic/uncertain_w_cem_qr_lowrank_seed0

    # Structured low-rank residual variants from docs/w_learning_ablation_plan_cn.md:
    # overcomplete V, optional residual-column group penalty (_rg...), optional
    # learned basis scales (_scale), and optional pairwise W0.
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings lh_q5_train1000 ^
        --projection_init pairwise --lowrank_basis overcomplete ^
        --variants qr_alt_lr4_rg1e-3_refresh10_self_cem2_nstd0.1_kcov0.1,qr_alt_lr8_scale_rg1e-3_refresh10_self_cem2_nstd0.1_kcov0.1 ^
        --max_test 200 --m_inducing 40 --qr_steps 40 --qr_outer_cycles 4 ^
        --lowrank_qr_w_signal_var 0.1 --lowrank_basis_scale_l1 1e-3 ^
        --out_dir experiments/synthetic/uncertain_w_cem_qr_wstruct_lh_seed0

    # Pairwise-augmented q(R) M-step:
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings l2_q2_train1000,lh_q5_train1000 ^
        --variants qr_alt_refresh10_self_cem2_nstd0.1_pair0.05_kcov0.1 ^
        --max_test 200 --m_inducing 40 --qr_steps 40 --qr_outer_cycles 4 ^
        --out_dir experiments/synthetic/uncertain_w_cem_qr_pairwise_seed0

    # Retrieval-aware train-anchor q(R): fixed raw candidate pools with
    # anchor-W, pointwise-W, distance-hybrid, or rank-hybrid raw/W top-k
    # neighborhoods; leave-anchor-out q(R) predictive loss; and optional
    # anchor-centric ranking loss. Add _ens<N> after _rank... to replace final
    # uncertain-W moment prediction with an N-sample deterministic-W self-CEM
    # ensemble, e.g. _rank0.1_ens5_kcov0.1.  Add _ortho<lambda> before
    # _kcov to penalize ||R_l R_l^T - I||_F^2 in the q(R) M-step.
    # Add _svd before _kcov to use the structured shared-SVD mean
    # R_l = U diag(s_l) V^T for q(R).
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings lh_q5_train1000 ^
        --variants qr_alt_wknn_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_pwwknn_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hya0.5_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1,qr_alt_hyra0.5_refresh10_self_cem2_nstd0.1_rank0.1_kcov0.1 ^
        --max_test 200 --m_inducing 40 --qr_outer_cycles 4 ^
        --qr_train_anchor_count 100 --qr_train_anchor_strategy kmeans_yvar ^
        --qr_retrieval_pool 200 --out_dir experiments/synthetic/uncertain_w_cem_qr_retrieval_seed0

    # q(R) orthogonality penalty smoke on LH:
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings lh_q5_train1000 ^
        --variants qr_alt_hya0.25_refresh10_self_cem2_nstd0.1_rank0.1_ortho0.1_kcov0.1,self_cem2_nstd0.1_kcov0.1 ^
        --seed_start 0 --num_exp 1 --max_test 40 --m_inducing 40 ^
        --qr_outer_cycles 4 --qr_train_anchor_count 100 ^
        --qr_train_anchor_strategy kmeans_yvar --qr_train_n_neighbors 35 ^
        --qr_retrieval_pool 200 ^
        --out_dir experiments/synthetic/uncertain_w_cem_qr_ortho_lh_smoke

    # Inductive train-anchor q(R): learn q(R) on train medoid anchors, then
    # induce q(W) at test anchors for the final uncertain-W CEM head.
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings lh_q5_train1000 ^
        --variants qr_alt_lr2_refresh10_self_cem2_nstd0.1_kcov0.1 ^
        --max_test 200 --n_neighbors 100 --m_inducing 100 ^
        --qr_train_anchor_count 100 --qr_train_n_neighbors 100 ^
        --qr_train_anchor_strategy kmeans_yvar ^
        --qr_steps 40 --qr_outer_cycles 4 ^
        --out_dir experiments/synthetic/uncertain_w_cem_qr_trainanchors_seed0

    # q(W) moment forms for ``self_cem`` / ``qr_mstep_self_cem`` (see
    # docs/w_learning_ablation_plan_cn.md §4--8): optional ``_wm<kind>`` after
    # ``self_cem<u>``, optional ``_lr<rank>`` before ``_wm`` when using
    # ``_wmlrprior``.  Default homogeneous diagonal matches omitting ``_wm``.
    #
    #   _wmdiag       diagonal homoscedastic q(W) per input dimension (baseline)
    #   _wmdet        deterministic W (zero variance): isolates uncertain-W term
    #   _wmhetero     diagonal q(W) variance scaled by train-X marginal spread
    #   _wmlrprior    low-rank row covariance for W=W0+C V^T with prior C-variance
    #
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings lh_q5_train1000 ^
        --variants self_cem2_nstd0.1_kcov0.1,self_cem2_wmdet_nstd0.1_kcov0.1,self_cem2_wmhetero_nstd0.1_kcov0.1,self_cem2_lr8_wmlrprior_nstd0.1_kcov0.1 ^
        --num_exp 1 --max_test 200 --lowrank_basis pca_residual ^
        --out_dir experiments/synthetic/uncertain_w_cem_wm_ablate_lh_seed0

    # Soft self-VEM counterpart to non-MC uncertain-W self-CEM:
    python experiments/synthetic/compare_uncertain_w_cem_l2_lh.py ^
        --settings lh_q5_train1000 ^
        --variants self_cem2_nstd0.1_kcov0.1,self_vem2_nstd0.1_kcov0.1,qr_mstep_self_vem2_nstd0.1_kcov0.1 ^
        --max_test 40 --hyper_steps 3 --gate_steps 4 ^
        --out_dir experiments/synthetic/self_vem_lh_seed0_40_20260518

This runner can be used in two modes.  The base variants are prediction-head
tests with a fixed mean projection (PLS/PCA) and controlled uncertain-W
covariance.  Variants prefixed by ``qr_mstep_`` add a sparse-GP q(R) M-step
after the local CEM state is learned, then evaluate prediction with the q(R)
induced q(W) moments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from djgp.diagnostics import sanitize_for_json  # noqa: E402
from djgp.evaluation.calibration import RESULT_ROW_KEYS, pack_gaussian_uq_list  # noqa: E402
from djgp.evaluation.crps import mean_gaussian_crps_torch  # noqa: E402
from djgp.jumpgp_bridge import center_linear_gate  # noqa: E402
from djgp.projections import (  # noqa: E402
    FixedLabelGateConfig,
    FixedLabelHyperoptConfig,
    SelfCEMConfig,
    SelfVEMConfig,
    UncertainWQRMstepConfig,
    compute_pls_W0,
    fit_uncertain_kernel_hyperparams_from_labels,
    run_uncertain_w_self_cem,
    run_uncertain_w_self_vem,
    sparse_gp_lowrank_residual_w_moments_for_mode,
    sparse_gp_uncertain_w_moments_for_mode,
    train_qr_uncertain_w_cem_mstep,
    uncertain_w_cem_predict_from_labels,
)
from experiments.synthetic.compare_paper_synthetic_baselines import (  # noqa: E402
    SETTING_PRESETS,
    _generate_paper_data,
    _set_seed,
)
from shared.utils1 import jumpgp_ld_wrapper  # noqa: E402


DEFAULT_VARIANTS = (
    "fixed_label_hyperopt_kcov0",
    "self_cem1_kcov0.05",
    "self_cem1_kcov0.1",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    mode: str
    kind: str
    init: str
    cem_updates: int
    kcov: float
    local_kind: str = ""
    refresh_every: int = 0
    local_noise_std_floor: float | None = None
    refresh_noise_std_floor: float | None = None
    refresh_damping: float | None = None
    qr_control: str = "learned"
    qr_pairwise_weight: float | None = None
    qr_rank_weight: float | None = None
    qr_ortho_weight: float | None = None
    qr_mean_parameterization: str | None = None
    qr_residual_group_weight: float | None = None
    neighbor_mode: str = "raw"
    retrieval_lambda: float = 1.0
    lowrank_rank: int = 0
    lowrank_train_basis_scale: bool = False
    w_moment_kind: str = "diag"
    ensemble_samples: int = 0


def _parse_variant(name: str) -> VariantSpec:
    mode = "anchor"
    base = name
    if base.startswith("anchor_"):
        base = base[len("anchor_") :]
    elif base.startswith("pointwise_"):
        mode = "pointwise"
        base = base[len("pointwise_") :]

    fixed = re.fullmatch(
        r"fixed_label_hyperopt(?:_nstd(?P<nstd>[0-9.]+))?(?:_(?P<init>cem|response|pairwise))?_kcov(?P<kcov>[0-9.]+)",
        base,
    )
    if fixed:
        return VariantSpec(
            name=name,
            mode=mode,
            kind="fixed_label_hyperopt",
            init=fixed.group("init") or "cem",
            cem_updates=0,
            kcov=float(fixed.group("kcov")),
            local_kind="fixed_label_hyperopt",
            local_noise_std_floor=float(fixed.group("nstd")) if fixed.group("nstd") is not None else None,
        )

    self_cem = re.fullmatch(
        r"self_(?P<soft>cem|vem)(?P<updates>[0-9]+)"
        r"(?:_lr(?P<lr>[0-9]+))?"
        r"(?:_wm(?P<wm>diag|det|lrprior|hetero))?"
        r"(?:_nstd(?P<nstd>[0-9.]+))?"
        r"(?:_(?P<init>cem|response|pairwise))?"
        r"_kcov(?P<kcov>[0-9.]+)",
        base,
    )
    if self_cem:
        wm_raw = self_cem.group("wm")
        wm = wm_raw if wm_raw is not None else "diag"
        lr_raw = self_cem.group("lr")
        lr_rank = int(lr_raw) if lr_raw is not None else 0
        if wm == "lrprior" and lr_rank <= 0:
            raise ValueError(
                f"Variant {name!r}: _wmlrprior requires an embedded low-rank basis, "
                "e.g. self_cem2_lr8_wmlrprior_kcov0.1 (add _lr<rank> before _wm...)."
            )
        local_kind = "self_vem" if self_cem.group("soft") == "vem" else "self_cem"
        return VariantSpec(
            name=name,
            mode=mode,
            kind=local_kind,
            init=self_cem.group("init") or "cem",
            cem_updates=int(self_cem.group("updates")),
            kcov=float(self_cem.group("kcov")),
            local_kind=local_kind,
            local_noise_std_floor=float(self_cem.group("nstd")) if self_cem.group("nstd") is not None else None,
            lowrank_rank=int(lr_rank),
            w_moment_kind=str(wm),
        )

    qr_fixed = re.fullmatch(
        r"qr_mstep_fixed_label(?:_pair(?P<pair>[0-9.]+))?(?:_ortho(?P<ortho>[0-9.eE+-]+))?(?P<svd>_svd)?(?:_(?P<init>cem|response|pairwise))?_kcov(?P<kcov>[0-9.]+)",
        base,
    )
    if qr_fixed:
        return VariantSpec(
            name=name,
            mode=mode,
            kind="qr_mstep",
            init=qr_fixed.group("init") or "cem",
            cem_updates=0,
            kcov=float(qr_fixed.group("kcov")),
            local_kind="fixed_label_hyperopt",
            qr_pairwise_weight=float(qr_fixed.group("pair")) if qr_fixed.group("pair") is not None else None,
            qr_ortho_weight=float(qr_fixed.group("ortho")) if qr_fixed.group("ortho") is not None else None,
            qr_mean_parameterization="shared_svd" if qr_fixed.group("svd") else None,
        )

    qr_self = re.fullmatch(
        r"qr_mstep_self_(?P<soft>cem|vem)(?P<updates>[0-9]+)"
        r"(?:_lr(?P<lr>[0-9]+))?"
        r"(?P<scale>_scale)?"
        r"(?:_wm(?P<wm>diag|det|lrprior|hetero))?"
        r"(?:_rg(?P<rg>[0-9.eE+-]+))?"
        r"(?:_pair(?P<pair>[0-9.]+))?"
        r"(?:_ortho(?P<ortho>[0-9.eE+-]+))?"
        r"(?P<svd>_svd)?"
        r"(?:_nstd(?P<nstd>[0-9.]+))?"
        r"(?:_(?P<init>cem|response|pairwise))?"
        r"_kcov(?P<kcov>[0-9.]+)",
        base,
    )
    if qr_self:
        wm_raw = qr_self.group("wm")
        wm = wm_raw if wm_raw is not None else "diag"
        lr_raw = qr_self.group("lr")
        lr_rank = int(lr_raw) if lr_raw is not None else 0
        if qr_self.group("scale") and lr_rank <= 0:
            raise ValueError(f"Variant {name!r}: _scale requires _lr<rank>.")
        if wm == "lrprior" and lr_rank <= 0:
            raise ValueError(
                f"Variant {name!r}: _wmlrprior requires _lr<rank>, "
                "e.g. qr_mstep_self_cem2_lr8_wmlrprior_kcov0.1."
            )
        return VariantSpec(
            name=name,
            mode=mode,
            kind="qr_mstep",
            init=qr_self.group("init") or "cem",
            cem_updates=int(qr_self.group("updates")),
            kcov=float(qr_self.group("kcov")),
            local_kind="self_vem" if qr_self.group("soft") == "vem" else "self_cem",
            qr_pairwise_weight=float(qr_self.group("pair")) if qr_self.group("pair") is not None else None,
            qr_ortho_weight=float(qr_self.group("ortho")) if qr_self.group("ortho") is not None else None,
            qr_mean_parameterization="shared_svd" if qr_self.group("svd") else None,
            qr_residual_group_weight=float(qr_self.group("rg")) if qr_self.group("rg") is not None else None,
            local_noise_std_floor=float(qr_self.group("nstd")) if qr_self.group("nstd") is not None else None,
            lowrank_rank=int(lr_rank),
            lowrank_train_basis_scale=bool(qr_self.group("scale")),
            w_moment_kind=str(wm),
        )

    qr_alt = re.fullmatch(
        r"qr_alt(?:_lr(?P<lr>[0-9]+))?"
        r"(?P<scale>_scale)?"
        r"(?:_rg(?P<rg>[0-9.eE+-]+))?"
        r"(?:(?:_(?P<neighbor>wknn|pwwknn))|(?:_hyra(?P<hyra>[0-9.]+))|(?:_hyrpw(?P<hyrpw>[0-9.]+))|(?:_hya(?P<hya>[0-9.]+))|(?:_hypw(?P<hypw>[0-9.]+)))?"
        r"_refresh(?P<refresh>[0-9]+)_self_cem(?P<updates>[0-9]+)"
        r"(?:_nstd(?P<nstd>[0-9.]+))?"
        r"(?:_damp(?P<damp>[0-9.]+))?"
        r"(?:_(?P<qrctl>qrprior|qrmeanonly|qrshuf))?"
        r"(?:_pair(?P<pair>[0-9.]+))?"
        r"(?:_rank(?P<rank>[0-9.]+))?"
        r"(?:_ens(?P<ens>[0-9]+))?"
        r"(?:_ortho(?P<ortho>[0-9.eE+-]+))?"
        r"(?P<svd>_svd)?"
        r"(?:_(?P<init>cem|response|pairwise))?"
        r"_kcov(?P<kcov>[0-9.]+)",
        base,
    )
    if qr_alt:
        lr_rank = int(qr_alt.group("lr")) if qr_alt.group("lr") is not None else 0
        if qr_alt.group("scale") and lr_rank <= 0:
            raise ValueError(f"Variant {name!r}: _scale requires _lr<rank>.")
        neighbor = qr_alt.group("neighbor")
        retrieval_lambda = 1.0
        if qr_alt.group("hyra") is not None:
            neighbor_mode = "hybrid_rank_anchor"
            retrieval_lambda = float(qr_alt.group("hyra"))
        elif qr_alt.group("hyrpw") is not None:
            neighbor_mode = "hybrid_rank_pointwise"
            retrieval_lambda = float(qr_alt.group("hyrpw"))
        elif qr_alt.group("hya") is not None:
            neighbor_mode = "hybrid_anchor"
            retrieval_lambda = float(qr_alt.group("hya"))
        elif qr_alt.group("hypw") is not None:
            neighbor_mode = "hybrid_pointwise"
            retrieval_lambda = float(qr_alt.group("hypw"))
        elif neighbor == "pwwknn":
            neighbor_mode = "pointwise_wknn"
        elif neighbor == "wknn":
            neighbor_mode = "anchor_wknn"
        else:
            neighbor_mode = "raw"
        return VariantSpec(
            name=name,
            mode=mode,
            kind="qr_alt",
            init=qr_alt.group("init") or "cem",
            cem_updates=int(qr_alt.group("updates")),
            kcov=float(qr_alt.group("kcov")),
            local_kind="meanw_cem",
            refresh_every=int(qr_alt.group("refresh")),
            local_noise_std_floor=float(qr_alt.group("nstd")) if qr_alt.group("nstd") is not None else None,
            refresh_noise_std_floor=float(qr_alt.group("nstd")) if qr_alt.group("nstd") is not None else None,
            refresh_damping=float(qr_alt.group("damp")) if qr_alt.group("damp") is not None else None,
            qr_control=(qr_alt.group("qrctl") or "learned").replace("qr", "", 1),
            qr_pairwise_weight=float(qr_alt.group("pair")) if qr_alt.group("pair") is not None else None,
            qr_rank_weight=float(qr_alt.group("rank")) if qr_alt.group("rank") is not None else None,
            qr_ortho_weight=float(qr_alt.group("ortho")) if qr_alt.group("ortho") is not None else None,
            qr_mean_parameterization="shared_svd" if qr_alt.group("svd") else None,
            qr_residual_group_weight=float(qr_alt.group("rg")) if qr_alt.group("rg") is not None else None,
            neighbor_mode=neighbor_mode,
            retrieval_lambda=float(retrieval_lambda),
            lowrank_rank=int(lr_rank),
            lowrank_train_basis_scale=bool(qr_alt.group("scale")),
            ensemble_samples=int(qr_alt.group("ens")) if qr_alt.group("ens") is not None else 0,
        )
    raise ValueError(f"Unknown uncertain-W CEM variant {name!r}.")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    preferred = [
        "setting",
        "family",
        "seed",
        "variant",
        "status",
        "mode",
        "kind",
        "init",
        "projection_init",
        "kcov_lambda",
        "cem_updates",
        "local_noise_std_floor",
        "local_noise_var_floor",
        "refresh_every",
        "refresh_noise_std_floor",
        "refresh_noise_var_floor",
        "refresh_damping",
        "refresh_gate_l2",
        "refresh_gate_max_norm",
        "qr_control",
        "ensemble_samples",
        "neighbor_mode",
        "retrieval_lambda",
        "qr_retrieval_pool",
        "qr_anchor_pred_weight",
        "qr_retrieval_composite_temperature",
        "qr_rank_weight",
        "qr_rank_temperature",
        "qr_outer_cycles",
        "N_train",
        "N_test",
        "n_test_eval",
        "Q",
        "observed_dim",
        "latent_dim",
        "n_neighbors",
        "m_inducing",
        "w_var",
        "w_cov_lengthscale",
        "qr_steps",
        "qr_beta_kl",
        "qr_train_log_std",
        "qr_pairwise_weight",
        "qr_pairwise_margin",
        "qr_ortho_weight",
        "qr_ortho_target_scale",
        "qr_ortho_normalize",
        "qr_mean_parameterization",
        "qr_residual_group_weight",
        "qr_residual_group_eps",
        "lowrank_rank",
        "lowrank_train_basis_scale",
        "w_moment_kind",
        "lowrank_basis",
        "lowrank_qr_w_signal_var",
        "lowrank_basis_scale_l1",
        "standardize_x",
        "standardize_y_for_head",
        *RESULT_ROW_KEYS,
        "init_sec",
        "train_sec",
        "pred_sec",
        "init_inlier_rate",
        "final_inlier_rate",
        "label_change_rate",
        "mean_n_inliers",
        "min_n_inliers",
        "median_n_inliers",
        "max_n_inliers",
        "gate_norm_mean",
        "mean_lengthscale",
        "mean_signal_var",
        "mean_noise_var",
        "mean_kernel_vector_correction",
        "ensemble_cem_sec",
        "ensemble_predict_sec",
        "ensemble_total_sec",
        "ensemble_within_sigma_mean",
        "ensemble_between_sigma_mean",
        "ensemble_total_sigma_mean",
        "ensemble_mean_n_inliers",
        "ensemble_fallback_count",
        "qr_train_sec",
        "refresh_train_sec",
        "qr_best_loss",
        "qr_kl_per_anchor",
        "qr_R_std_median",
        "qr_grad_data_over_scaled_KL_R_mu_final",
        "qr_grad_anchor_pred_R_mu_final",
        "qr_grad_anchor_pred_over_scaled_KL_R_mu_final",
        "qr_anchor_pred_nll_final",
        "qr_anchor_pred_var_mean_final",
        "qr_grad_pairwise_R_mu_final",
        "qr_grad_pairwise_over_data_R_mu_final",
        "qr_grad_pairwise_over_scaled_KL_R_mu_final",
        "qr_grad_rank_R_mu_final",
        "qr_grad_rank_over_anchor_pred_R_mu_final",
        "qr_grad_rank_over_scaled_KL_R_mu_final",
        "qr_grad_data_over_scaled_KL_R_log_std_final",
        "qr_pairwise_loss_final",
        "qr_pairwise_same_dist_mean_final",
        "qr_pairwise_cross_dist_mean_final",
        "qr_pairwise_cross_violation_rate_final",
        "qr_rank_loss_final",
        "qr_rank_pos_dist_mean_final",
        "qr_rank_neg_dist_mean_final",
        "qr_rank_pair_accuracy_final",
        "qr_rank_pair_count_final",
        "qr_residual_group_penalty_final",
        "qr_residual_group_usage_mean_final",
        "qr_residual_group_usage_max_final",
        "qr_residual_group_top_feature_frac_final",
        "qr_grad_residual_group_R_mu_final",
        "qr_grad_residual_group_over_data_R_mu_final",
        "qr_grad_residual_group_over_scaled_KL_R_mu_final",
        "qr_basis_scale_mean_final",
        "qr_basis_scale_min_final",
        "qr_basis_scale_max_final",
        "qr_train_wknn_metric_lambda_final",
        "qr_train_wknn_rank_mixing_final",
        "qr_train_wknn_raw_topk_overlap_final",
        "qr_train_wknn_selected_inlier_frac_final",
        "test_wknn_rank_mixing",
        "test_wknn_raw_topk_overlap",
        "test_wknn_selected_inlier_frac",
        "refresh_count",
        "refresh_label_change_mean",
        "refresh_label_change_final",
        "refresh_label_change_max",
        "refresh_inlier_rate_final",
        "refresh_noise_var_floor_active",
        "refresh_damping_active",
        "refresh_gate_norm_final",
        "error",
    ]
    keys = [k for k in preferred if k in all_keys]
    keys.extend(sorted(all_keys - set(keys)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_or_nan(x: Any) -> float:
    try:
        if x == "":
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["setting"]), str(row["variant"])), []).append(row)
    out: list[dict[str, Any]] = []
    value_keys = list(RESULT_ROW_KEYS) + [
        "init_sec",
        "train_sec",
        "pred_sec",
        "init_inlier_rate",
        "final_inlier_rate",
        "label_change_rate",
        "mean_n_inliers",
        "min_n_inliers",
        "median_n_inliers",
        "max_n_inliers",
        "gate_norm_mean",
        "mean_lengthscale",
        "mean_signal_var",
        "mean_noise_var",
        "mean_kernel_vector_correction",
        "ensemble_cem_sec",
        "ensemble_predict_sec",
        "ensemble_total_sec",
        "ensemble_within_sigma_mean",
        "ensemble_between_sigma_mean",
        "ensemble_total_sigma_mean",
        "ensemble_mean_n_inliers",
        "ensemble_fallback_count",
        "qr_train_sec",
        "refresh_train_sec",
        "qr_best_loss",
        "qr_kl_per_anchor",
        "qr_R_std_median",
        "qr_grad_data_over_scaled_KL_R_mu_final",
        "qr_grad_anchor_pred_R_mu_final",
        "qr_grad_anchor_pred_over_scaled_KL_R_mu_final",
        "qr_anchor_pred_nll_final",
        "qr_anchor_pred_var_mean_final",
        "qr_grad_pairwise_R_mu_final",
        "qr_grad_pairwise_over_data_R_mu_final",
        "qr_grad_pairwise_over_scaled_KL_R_mu_final",
        "qr_grad_rank_R_mu_final",
        "qr_grad_rank_over_anchor_pred_R_mu_final",
        "qr_grad_rank_over_scaled_KL_R_mu_final",
        "qr_grad_data_over_scaled_KL_R_log_std_final",
        "qr_pairwise_loss_final",
        "qr_pairwise_same_dist_mean_final",
        "qr_pairwise_cross_dist_mean_final",
        "qr_pairwise_cross_violation_rate_final",
        "qr_rank_loss_final",
        "qr_rank_pos_dist_mean_final",
        "qr_rank_neg_dist_mean_final",
        "qr_rank_pair_accuracy_final",
        "qr_rank_pair_count_final",
        "qr_residual_group_penalty_final",
        "qr_residual_group_usage_mean_final",
        "qr_residual_group_usage_max_final",
        "qr_residual_group_top_feature_frac_final",
        "qr_grad_residual_group_R_mu_final",
        "qr_grad_residual_group_over_data_R_mu_final",
        "qr_grad_residual_group_over_scaled_KL_R_mu_final",
        "qr_basis_scale_mean_final",
        "qr_basis_scale_min_final",
        "qr_basis_scale_max_final",
        "qr_train_wknn_metric_lambda_final",
        "qr_train_wknn_rank_mixing_final",
        "qr_train_wknn_raw_topk_overlap_final",
        "qr_train_wknn_selected_inlier_frac_final",
        "test_wknn_rank_mixing",
        "test_wknn_raw_topk_overlap",
        "test_wknn_selected_inlier_frac",
        "refresh_count",
        "refresh_label_change_mean",
        "refresh_label_change_final",
        "refresh_label_change_max",
        "refresh_inlier_rate_final",
        "refresh_noise_var_floor_active",
        "refresh_damping_active",
        "refresh_gate_norm_final",
    ]
    for (setting, variant), group in sorted(groups.items()):
        agg: dict[str, Any] = {"setting": setting, "variant": variant, "n_runs": len(group)}
        for meta in (
            "family",
            "mode",
            "kind",
            "init",
            "projection_init",
            "kcov_lambda",
            "cem_updates",
            "local_noise_std_floor",
            "local_noise_var_floor",
            "refresh_every",
            "refresh_noise_std_floor",
            "refresh_noise_var_floor",
            "refresh_damping",
            "refresh_gate_l2",
            "refresh_gate_max_norm",
            "qr_control",
            "ensemble_samples",
            "neighbor_mode",
            "retrieval_lambda",
            "qr_retrieval_pool",
            "qr_anchor_pred_weight",
            "qr_retrieval_composite_temperature",
            "qr_rank_weight",
            "qr_rank_temperature",
            "qr_outer_cycles",
            "N_train",
            "N_test",
            "n_test_eval",
            "Q",
            "observed_dim",
            "latent_dim",
            "n_neighbors",
            "m_inducing",
            "w_var",
            "w_cov_lengthscale",
            "qr_steps",
            "qr_beta_kl",
            "qr_train_log_std",
            "qr_pairwise_weight",
            "qr_pairwise_margin",
            "qr_ortho_weight",
            "qr_ortho_target_scale",
            "qr_ortho_normalize",
            "qr_mean_parameterization",
            "qr_residual_group_weight",
            "qr_residual_group_eps",
            "lowrank_rank",
            "lowrank_train_basis_scale",
            "lowrank_basis",
            "lowrank_qr_w_signal_var",
            "lowrank_basis_scale_l1",
            "standardize_x",
            "standardize_y_for_head",
        ):
            agg[meta] = group[0].get(meta, "")
        for key in value_keys:
            vals = np.asarray([_float_or_nan(r.get(key)) for r in group], dtype=np.float64)
            if np.isfinite(vals).any():
                agg[f"{key}_mean"] = float(np.nanmean(vals))
                agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def _metrics_pack(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, wall_sec: float) -> list[float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64).reshape(-1), 1e-12)
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    crps = float(
        mean_gaussian_crps_torch(
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(mu, dtype=torch.float32),
            torch.as_tensor(sigma, dtype=torch.float32),
        ).item()
    )
    return pack_gaussian_uq_list(rmse, crps, wall_sec, y, mu, sigma)


def _neighbor_stack(
    X_query: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    n_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        d = torch.cdist(X_query, X_train)
        idx = torch.topk(d, k=min(int(n_neighbors), X_train.shape[0]), largest=False, dim=1).indices
    return X_train[idx].contiguous(), y_train[idx].contiguous(), idx.contiguous()


def _is_wknn_variant(spec: VariantSpec) -> bool:
    return str(spec.neighbor_mode) != "raw"


def _uses_pointwise_retrieval(spec: VariantSpec) -> bool:
    return str(spec.neighbor_mode) in {"pointwise_wknn", "hybrid_pointwise", "hybrid_rank_pointwise"}


def _uses_hybrid_retrieval(spec: VariantSpec) -> bool:
    return str(spec.neighbor_mode) in {
        "hybrid_anchor",
        "hybrid_pointwise",
        "hybrid_rank_anchor",
        "hybrid_rank_pointwise",
    }


def _uses_rank_hybrid_retrieval(spec: VariantSpec) -> bool:
    return str(spec.neighbor_mode) in {"hybrid_rank_anchor", "hybrid_rank_pointwise"}


def _expected_anchor_pool_sqdist(
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    *,
    mode: str,
    w_moments: dict[str, torch.Tensor],
) -> torch.Tensor:
    if mode != "anchor":
        raise NotImplementedError("W-KNN retrieval currently supports anchor-conditioned W only.")
    W_mu = w_moments["W_mu_anchor"]
    delta = X_pool - X_anchor.unsqueeze(1)
    mean_proj = torch.einsum("tqd,tpd->tpq", W_mu, delta)
    mean_sq = mean_proj.pow(2).sum(dim=-1)
    if "W_cov_anchor" in w_moments:
        var_sq = torch.einsum("tqde,tpd,tpe->tpq", w_moments["W_cov_anchor"], delta, delta).sum(dim=-1)
    else:
        W_var = w_moments["W_var_anchor"]
        var_sq = (W_var[:, None, :, :] * delta[:, :, None, :].pow(2)).sum(dim=(-1, -2))
    return (mean_sq + var_sq).clamp_min(0.0)


def _raw_anchor_pool_sqdist(X_anchor: torch.Tensor, X_pool: torch.Tensor) -> torch.Tensor:
    return (X_pool - X_anchor.unsqueeze(1)).pow(2).sum(dim=-1).clamp_min(0.0)


def _fixed_pointwise_pool_sqdist(
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    *,
    W0: torch.Tensor,
) -> torch.Tensor:
    z_anchor = torch.einsum("qd,td->tq", W0, X_anchor)
    z_pool = torch.einsum("qd,tpd->tpq", W0, X_pool)
    return (z_pool - z_anchor.unsqueeze(1)).pow(2).sum(dim=-1).clamp_min(0.0)


def _normalize_pool_dist(dist: torch.Tensor) -> torch.Tensor:
    scale = torch.median(dist.detach(), dim=1, keepdim=True).values.clamp_min(1e-12)
    return dist / scale


def _pool_fractional_rank(dist: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(dist.detach(), dim=1, stable=True)
    P = dist.shape[1]
    ranks = torch.empty_like(dist)
    values = torch.arange(P, device=dist.device, dtype=dist.dtype).reshape(1, P).expand_as(dist)
    ranks.scatter_(1, order, values)
    if P > 1:
        ranks = ranks / float(P - 1)
    return ranks


def _gather_pool(
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    idx_pool: torch.Tensor,
    top: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    D = X_pool.shape[-1]
    X_sel = torch.gather(X_pool, 1, top.unsqueeze(-1).expand(-1, -1, D)).contiguous()
    y_sel = torch.gather(y_pool, 1, top).contiguous()
    idx_sel = torch.gather(idx_pool, 1, top).contiguous()
    return X_sel, y_sel, idx_sel


def _select_wknn_neighbors(
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    idx_pool: torch.Tensor,
    *,
    n_neighbors: int,
    mode: str,
    w_moments: dict[str, torch.Tensor],
    spec: VariantSpec,
    W0: torch.Tensor,
    pointwise_dist: torch.Tensor | None = None,
    label_pool: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    k = min(int(n_neighbors), X_pool.shape[1])
    with torch.no_grad():
        raw_dist = _raw_anchor_pool_sqdist(X_anchor, X_pool)
        if _uses_pointwise_retrieval(spec):
            w_dist = pointwise_dist
            if w_dist is None:
                w_dist = _fixed_pointwise_pool_sqdist(X_anchor, X_pool, W0=W0)
        else:
            w_dist = _expected_anchor_pool_sqdist(X_anchor, X_pool, mode=mode, w_moments=w_moments)
        if _uses_rank_hybrid_retrieval(spec):
            lam = float(min(max(float(spec.retrieval_lambda), 0.0), 1.0))
            dist = lam * _pool_fractional_rank(w_dist) + (1.0 - lam) * _pool_fractional_rank(raw_dist)
            rank_mixing = True
        elif _uses_hybrid_retrieval(spec):
            lam = float(min(max(float(spec.retrieval_lambda), 0.0), 1.0))
            dist = lam * _normalize_pool_dist(w_dist) + (1.0 - lam) * _normalize_pool_dist(raw_dist)
            rank_mixing = False
        else:
            lam = 1.0
            dist = w_dist
            rank_mixing = False
        top = torch.topk(dist, k=k, largest=False, dim=1).indices
    X_sel, y_sel, idx_sel = _gather_pool(X_pool, y_pool, idx_pool, top)
    raw_idx = idx_pool[:, :k]
    overlap = (idx_sel.unsqueeze(2) == raw_idx.unsqueeze(1)).any(dim=2).to(dtype=torch.float64).mean()
    selected_d = torch.gather(dist.detach(), 1, top)
    selected_w_d = torch.gather(w_dist.detach(), 1, top)
    selected_raw_d = torch.gather(raw_dist.detach(), 1, top)
    raw_top_metric_d = dist.detach()[:, :k]
    raw_top_w_d = w_dist.detach()[:, :k]
    raw_top_raw_d = raw_dist.detach()[:, :k]
    diag = {
        "wknn_pool_size": float(X_pool.shape[1]),
        "wknn_selected_k": float(k),
        "wknn_metric_lambda": float(lam),
        "wknn_rank_mixing": float(1.0 if rank_mixing else 0.0),
        "wknn_raw_topk_overlap": float(overlap.cpu().item()),
        "wknn_selected_dist_mean": float(selected_d.mean().cpu().item()),
        "wknn_selected_w_dist_mean": float(selected_w_d.mean().cpu().item()),
        "wknn_selected_raw_dist_mean": float(selected_raw_d.mean().cpu().item()),
        "wknn_raw_topk_dist_mean": float(raw_top_metric_d.mean().cpu().item()),
        "wknn_raw_topk_w_dist_mean": float(raw_top_w_d.mean().cpu().item()),
        "wknn_raw_topk_raw_dist_mean": float(raw_top_raw_d.mean().cpu().item()),
    }
    if label_pool is not None:
        selected_labels = torch.gather(label_pool.bool(), 1, top)
        raw_labels = label_pool.bool()[:, :k]
        diag["wknn_selected_inlier_frac"] = float(selected_labels.to(dtype=torch.float64).mean().cpu().item())
        diag["wknn_raw_topk_inlier_frac"] = float(raw_labels.to(dtype=torch.float64).mean().cpu().item())
    return X_sel, y_sel, idx_sel, diag


def _retrieval_pool_size(args: argparse.Namespace, *, n_neighbors: int, n_train: int) -> int:
    pool = int(args.qr_retrieval_pool)
    if pool <= 0:
        pool = 200
    return min(max(int(n_neighbors), pool), int(n_train))


def _kmeans_medoids(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = X.shape[0]
    k = max(1, min(int(k), n))
    if k >= n:
        return np.arange(n, dtype=np.int64)
    km = KMeans(n_clusters=k, random_state=int(seed), n_init=5, max_iter=200)
    labels = km.fit_predict(np.asarray(X, dtype=np.float64))
    out: list[int] = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        d2 = np.sum((X[idx] - km.cluster_centers_[c]) ** 2, axis=1)
        out.append(int(idx[int(np.argmin(d2))]))
    return np.asarray(sorted(set(out)), dtype=np.int64)


def _select_train_anchor_indices(
    X: np.ndarray,
    y: np.ndarray,
    count: int,
    *,
    strategy: str,
    seed: int,
    local_var_neighbors: int = 30,
) -> np.ndarray:
    n = int(X.shape[0])
    count = max(1, min(int(count), n))
    if count >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    strategy = str(strategy)
    if strategy == "kmeans":
        return _kmeans_medoids(X, count, seed)
    if strategy == "random":
        return np.sort(rng.choice(n, size=count, replace=False)).astype(np.int64)
    if strategy in {"kmeans_random", "kmeans_yvar"}:
        n_km = max(1, count // 2)
        selected = set(int(i) for i in _kmeans_medoids(X, n_km, seed))
        remaining = count - len(selected)
        if remaining <= 0:
            return np.asarray(sorted(selected), dtype=np.int64)
        if strategy == "kmeans_random":
            pool = np.asarray([i for i in range(n) if i not in selected], dtype=np.int64)
            extra = rng.choice(pool, size=min(remaining, pool.size), replace=False)
            selected.update(int(i) for i in extra)
        else:
            k = max(2, min(int(local_var_neighbors), n))
            nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(np.asarray(X, dtype=np.float64))
            idx = nn.kneighbors(return_distance=False)
            scores = np.var(np.asarray(y, dtype=np.float64).reshape(-1)[idx], axis=1)
            order = np.argsort(-scores)
            for i in order:
                if int(i) not in selected:
                    selected.add(int(i))
                    if len(selected) >= count:
                        break
        return np.asarray(sorted(selected), dtype=np.int64)
    raise ValueError(f"Unknown train-anchor strategy {strategy!r}.")


def _neighbor_coverage_diag(idx: torch.Tensor, n_train: int, prefix: str) -> dict[str, float]:
    idx_cpu = idx.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    counts = torch.bincount(idx_cpu, minlength=int(n_train)).to(dtype=torch.float64)
    visited = counts > 0
    if visited.any():
        mean_overlap = float(counts[visited].mean().item())
        max_overlap = float(counts.max().item())
    else:
        mean_overlap = 0.0
        max_overlap = 0.0
    return {
        f"{prefix}_visited_frac": float(visited.to(dtype=torch.float64).mean().item()),
        f"{prefix}_mean_overlap_visited": mean_overlap,
        f"{prefix}_max_overlap": max_overlap,
    }


def _pca_W0(X: np.ndarray, Q: int, *, scaled: bool = False) -> np.ndarray:
    W = PCA(n_components=int(Q), random_state=0).fit(np.asarray(X, dtype=np.float64)).components_.astype(np.float64)
    if not scaled:
        return W
    z = np.asarray(X, dtype=np.float64) @ W.T
    scale = (10.0 / np.maximum(np.ptp(z, axis=0), 1e-6) ** 2).reshape(-1, 1)
    return W * scale


def _pairwise_direction_candidates(
    X: np.ndarray,
    y: np.ndarray,
    count: int,
    *,
    seed: int,
) -> list[np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, D = X.shape
    if n < 2 or count <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    max_pairs = min(max(2000, 200 * int(count)), n * max(n - 1, 1) // 2)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n - 1, size=max_pairs)
    j = j + (j >= i)
    delta = X[i] - X[j]
    norm = np.linalg.norm(delta, axis=1)
    valid = norm > 1e-10
    if not np.any(valid):
        return []
    score = np.abs(y[i] - y[j]) / (norm + 1e-6)
    order = np.argsort(score[valid])[::-1]
    delta_v = delta[valid][order]
    norm_v = norm[valid][order]
    out: list[np.ndarray] = []
    for vec, vec_norm in zip(delta_v, norm_v):
        out.append((vec / vec_norm).astype(np.float64))
        if len(out) >= int(count):
            break
    if len(out) < int(count):
        out.extend([np.eye(D, dtype=np.float64)[d] for d in range(D)])
    return out[: int(count)]


def _orthonormal_rows_from_candidates(candidates: list[np.ndarray], Q: int, D: int) -> np.ndarray:
    basis: list[np.ndarray] = []
    for cand in candidates + [np.eye(D, dtype=np.float64)[d] for d in range(D)]:
        v = np.asarray(cand, dtype=np.float64).reshape(D).copy()
        for b in basis:
            v = v - np.dot(v, b) * b
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            basis.append(v / norm)
        if len(basis) >= int(Q):
            break
    if len(basis) < int(Q):
        raise RuntimeError(f"Could only build {len(basis)} projection rows, requested {Q}.")
    return np.stack(basis, axis=0).astype(np.float64)


def _pairwise_W0(X: np.ndarray, y: np.ndarray, Q: int, *, seed: int) -> np.ndarray:
    D = int(np.asarray(X).shape[1])
    candidates = _pairwise_direction_candidates(X, y, max(8 * int(Q), int(Q) + 8), seed=seed)
    return _orthonormal_rows_from_candidates(candidates, Q, D)


def _projection_W0(X: np.ndarray, y: np.ndarray, Q: int, mode: str, *, seed: int = 0) -> np.ndarray:
    if mode == "pls":
        return compute_pls_W0(X, y, Q=Q).astype(np.float64)
    if mode == "pca":
        return _pca_W0(X, Q, scaled=False)
    if mode == "pca_scaled":
        return _pca_W0(X, Q, scaled=True)
    if mode == "pairwise":
        return _pairwise_W0(X, y, Q, seed=seed).astype(np.float64)
    raise ValueError(f"Unknown projection init {mode!r}.")


def _residual_basis_V(
    X: np.ndarray,
    y: np.ndarray,
    W0: np.ndarray,
    r: int,
    *,
    mode: str,
    seed: int,
) -> np.ndarray:
    D = int(W0.shape[1])
    r = max(0, min(int(r), D))
    if r == 0:
        return np.zeros((D, 0), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    candidates: list[np.ndarray] = []
    if mode in {"pca", "pca_residual", "pls_pca", "overcomplete"}:
        n_comp = min(D, max(r + int(W0.shape[0]) + 5, r))
        pcs = PCA(n_components=n_comp, random_state=int(seed)).fit(np.asarray(X, dtype=np.float64)).components_
        if mode in {"pls_pca", "overcomplete"}:
            candidates.extend([row.astype(np.float64) for row in np.asarray(W0, dtype=np.float64)])
        candidates.extend([pc.astype(np.float64) for pc in pcs])
        if mode == "overcomplete":
            candidates.extend(
                _pairwise_direction_candidates(
                    X,
                    y,
                    max(2 * r, r + int(W0.shape[0])),
                    seed=int(seed) + 17,
                )
            )
            candidates.extend([rng.normal(size=D).astype(np.float64) for _ in range(max(r, 4))])
    elif mode == "random":
        candidates.extend([rng.normal(size=D).astype(np.float64) for _ in range(max(4 * r, r + 4))])
    else:
        raise ValueError(f"Unknown low-rank basis mode {mode!r}.")
    candidates.extend([np.eye(D, dtype=np.float64)[j] for j in range(D)])

    basis: list[np.ndarray] = []
    if mode == "pca_residual":
        row_basis = []
        for row in np.asarray(W0, dtype=np.float64):
            v = row.copy()
            for b in row_basis:
                v = v - np.dot(v, b) * b
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                row_basis.append(v / norm)
    else:
        row_basis = []
    for cand in candidates:
        v = cand.copy()
        for b in row_basis:
            v = v - np.dot(v, b) * b
        for b in basis:
            v = v - np.dot(v, b) * b
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            basis.append(v / norm)
        if len(basis) >= r:
            break
    if len(basis) < r:
        raise RuntimeError(f"Could only build {len(basis)} low-rank residual basis vectors, requested {r}.")
    return np.stack(basis, axis=1).astype(np.float64)


def _median_distance(X: torch.Tensor) -> float:
    with torch.no_grad():
        d = torch.cdist(X, X)
        vals = d[d > 0]
        if vals.numel() == 0:
            return 1.0
        return float(vals.median().clamp_min(1e-3).item())


def _make_w_moments(
    *,
    mode: str,
    W0: torch.Tensor,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    w_var: float,
    w_cov_lengthscale: float,
) -> dict[str, torch.Tensor]:
    T, n, D = X_neighbors.shape
    Q = W0.shape[0]
    if mode == "anchor":
        return {
            "W_mu_anchor": W0.reshape(1, Q, D).expand(T, Q, D).contiguous(),
            "W_var_anchor": torch.full((T, Q, D), float(w_var), device=X_anchor.device, dtype=X_anchor.dtype),
        }
    if mode == "pointwise":
        X_all = torch.cat([X_anchor.unsqueeze(1), X_neighbors], dim=1)
        m = n + 1
        W_mu_all = W0.reshape(1, 1, Q, D).expand(T, m, Q, D).contiguous()
        with torch.no_grad():
            diff = X_all.unsqueeze(2) - X_all.unsqueeze(1)
            Kpos = torch.exp(-0.5 * diff.pow(2).sum(dim=-1) / max(float(w_cov_lengthscale) ** 2, 1e-12))
        W_cov_all = Kpos[:, None, None, :, :].expand(T, Q, D, m, m).contiguous() * float(w_var)
        return {"W_mu_all": W_mu_all, "W_cov_all": W_cov_all}
    raise ValueError(f"Unknown W mode {mode!r}.")


def _build_initial_w_moments(
    *,
    args: argparse.Namespace,
    spec: VariantSpec,
    W0: torch.Tensor,
    V_basis: torch.Tensor | None,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    X_train_t: torch.Tensor,
    w_cov_lengthscale: float,
) -> dict[str, torch.Tensor]:
    """Initial uncertain-W moments before any sparse-GP q(R) training."""
    kind = str(spec.w_moment_kind)
    if kind == "diag":
        return _make_w_moments(
            mode=spec.mode,
            W0=W0,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            w_var=float(args.w_var),
            w_cov_lengthscale=w_cov_lengthscale,
        )
    if kind == "det":
        base = _make_w_moments(
            mode=spec.mode,
            W0=W0,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            w_var=float(args.w_var),
            w_cov_lengthscale=w_cov_lengthscale,
        )
        if spec.mode == "anchor":
            return {
                "W_mu_anchor": base["W_mu_anchor"],
                "W_var_anchor": torch.zeros_like(base["W_var_anchor"]),
            }
        base["W_cov_all"].zero_()
        return base
    if kind == "hetero":
        if spec.mode != "anchor":
            raise ValueError(
                f"Variant {spec.name!r}: _wmhetero is only implemented for anchor-conditioned W "
                "(remove the pointwise_ prefix)."
            )
        base = _make_w_moments(
            mode="anchor",
            W0=W0,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            w_var=float(args.w_var),
            w_cov_lengthscale=w_cov_lengthscale,
        )
        std = X_train_t.std(dim=0).clamp_min(1e-6)
        scale = (std / std.mean()).pow(2)
        base["W_var_anchor"] = base["W_var_anchor"] * scale.reshape(1, 1, -1)
        return base
    if kind == "lrprior":
        if spec.mode != "anchor":
            raise ValueError(
                f"Variant {spec.name!r}: _wmlrprior is only implemented for anchor-conditioned W "
                "(remove the pointwise_ prefix)."
            )
        if V_basis is None:
            raise ValueError(
                f"Variant {spec.name!r}: _wmlrprior requires a residual basis; add _lr<rank> to the variant name."
            )
        return _lowrank_prior_w_moments(
            mode="anchor",
            W0=W0,
            V_basis=V_basis,
            X_anchor=X_anchor,
            qr_w_signal_var=float(args.w_var),
        )
    raise ValueError(f"Unknown w_moment_kind {kind!r} for variant {spec.name!r}.")


def _project_neighbors(X_anchor: torch.Tensor, X_neighbors: torch.Tensor, W0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Z_neighbors = torch.einsum("tnd,qd->tnq", X_neighbors, W0)
    Z_anchor = torch.einsum("td,qd->tq", X_anchor, W0)
    return Z_anchor, Z_neighbors


def _parse_jumpgp_logtheta(logtheta: torch.Tensor, Q: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lt = logtheta.detach().to(device=device, dtype=dtype).reshape(-1)
    if lt.numel() < Q + 2:
        pad = torch.zeros(Q + 2 - lt.numel(), device=device, dtype=dtype)
        lt = torch.cat([lt, pad], dim=0)
        lt[-1] = -1.15
    lengthscale = torch.exp(lt[:Q]).clamp(1e-3, 20.0)
    signal_var = torch.exp(2.0 * lt[Q]).clamp(1e-6, 100.0)
    noise_var = torch.exp(2.0 * lt[Q + 1]).clamp(1e-6, 10.0)
    return lengthscale, signal_var, noise_var


def _response_seed_labels(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    *,
    seed_frac: float,
    inlier_frac: float,
    min_inliers: int,
) -> torch.Tensor:
    T, n, _D = X_neighbors.shape
    labels = torch.zeros(T, n, device=X_neighbors.device, dtype=torch.bool)
    seed_k = min(max(int(round(float(seed_frac) * n)), 1), n)
    keep_k = min(max(int(round(float(inlier_frac) * n)), int(min_inliers)), n)
    with torch.no_grad():
        d = torch.sum((X_neighbors - X_anchor.unsqueeze(1)).pow(2), dim=-1)
        seed_idx = torch.topk(d, k=seed_k, largest=False, dim=1).indices
        seed_y = torch.gather(y_neighbors, 1, seed_idx)
        center = seed_y.median(dim=1).values
        dy = (y_neighbors - center.unsqueeze(1)).abs()
        keep = torch.topk(dy, k=keep_k, largest=False, dim=1).indices
        labels.scatter_(1, keep, True)
    return labels


def _pairwise_seed_labels(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    *,
    seed_frac: float,
    inlier_frac: float,
    min_inliers: int,
) -> torch.Tensor:
    T, n, _D = X_neighbors.shape
    labels = torch.zeros(T, n, device=X_neighbors.device, dtype=torch.bool)
    seed_k = min(max(int(round(float(seed_frac) * n)), 1), n)
    keep_k = min(max(int(round(float(inlier_frac) * n)), int(min_inliers)), n)
    with torch.no_grad():
        d_anchor = torch.sum((X_neighbors - X_anchor.unsqueeze(1)).pow(2), dim=-1)
        seed_idx = torch.topk(d_anchor, k=seed_k, largest=False, dim=1).indices
        seed_y = torch.gather(y_neighbors, 1, seed_idx)
        center = seed_y.median(dim=1).values
        dy = (y_neighbors - center.unsqueeze(1)).abs()
        dy_scale = dy.median(dim=1).values.clamp_min(1e-6).unsqueeze(1)
        d_scale = d_anchor.median(dim=1).values.clamp_min(1e-6).unsqueeze(1)
        score = d_anchor / d_scale + 1.5 * dy / dy_scale
        keep = torch.topk(score, k=keep_k, largest=False, dim=1).indices
        labels.scatter_(1, keep, True)
    return labels


def _default_hypers_from_labels(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    labels: torch.Tensor,
    W0: torch.Tensor,
    *,
    min_inliers: int,
) -> dict[str, torch.Tensor]:
    T, n, _D = X_neighbors.shape
    Q = W0.shape[0]
    Z_anchor, Z_neighbors = _project_neighbors(X_anchor, X_neighbors, W0)
    ell = torch.empty(T, Q, device=X_neighbors.device, dtype=X_neighbors.dtype)
    signal = torch.empty(T, device=X_neighbors.device, dtype=X_neighbors.dtype)
    noise = torch.empty(T, device=X_neighbors.device, dtype=X_neighbors.dtype)
    mean = torch.empty(T, device=X_neighbors.device, dtype=X_neighbors.dtype)
    for t in range(T):
        mask = labels[t].bool()
        if int(mask.sum().item()) < int(min_inliers):
            k = min(max(int(min_inliers), 1), n)
            d = torch.sum((X_neighbors[t] - X_anchor[t]).pow(2), dim=-1)
            idx = torch.topk(d, k=k, largest=False).indices
            mask = torch.zeros(n, device=X_neighbors.device, dtype=torch.bool)
            mask[idx] = True
        z = Z_neighbors[t, mask]
        if z.shape[0] > 1:
            pd = torch.cdist(z, z)
            vals = pd[pd > 0]
            med = vals.median().clamp(0.05, 10.0) if vals.numel() else torch.tensor(1.0, device=z.device, dtype=z.dtype)
        else:
            med = torch.linalg.norm(Z_neighbors[t] - Z_anchor[t], dim=1).median().clamp(0.05, 10.0)
        y_i = y_neighbors[t, mask]
        mean[t] = y_i.mean()
        var = y_i.var(unbiased=False).clamp_min(0.05)
        signal[t] = var.clamp(1e-3, 20.0)
        noise[t] = (0.10 * var).clamp(1e-4, 2.0)
        ell[t] = med
    return {"lengthscale": ell, "signal_var": signal, "noise_var": noise, "mean_const": mean}


def _run_meanw_cem_init(
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    W0: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    T, n, _D = X_neighbors.shape
    Q = W0.shape[0]
    dtype = X_neighbors.dtype
    device = X_neighbors.device
    labels = torch.zeros(T, n, device=device, dtype=torch.bool)
    mean_const = torch.empty(T, device=device, dtype=dtype)
    lengthscale = torch.empty(T, Q, device=device, dtype=dtype)
    signal_var = torch.empty(T, device=device, dtype=dtype)
    noise_var = torch.empty(T, device=device, dtype=dtype)
    gate = torch.zeros(T, Q + 1, device=device, dtype=dtype)
    mu = torch.empty(T, device=device, dtype=dtype)
    sigma = torch.empty(T, device=device, dtype=dtype)
    failed = 0
    Z_anchor, Z_neighbors = _project_neighbors(X_anchor, X_neighbors, W0)
    fallback = _default_hypers_from_labels(
        X_anchor,
        X_neighbors,
        y_neighbors,
        _response_seed_labels(X_anchor, X_neighbors, y_neighbors, seed_frac=0.25, inlier_frac=0.70, min_inliers=2),
        W0,
        min_inliers=2,
    )
    t0 = time.perf_counter()
    for t in range(T):
        try:
            mu_t, sig2_t, model, _h = jumpgp_ld_wrapper(
                Z_neighbors[t].float(),
                y_neighbors[t].reshape(-1, 1).float(),
                Z_anchor[t : t + 1].float(),
                mode="CEM",
                flag=False,
                device=torch.device("cpu"),
            )
            labels[t] = model["r"].detach().reshape(-1).to(device=device).bool()
            mean_const[t] = model["ms"].detach().reshape(-1)[0].to(device=device, dtype=dtype)
            ell_t, sig_t, noise_t = _parse_jumpgp_logtheta(model["logtheta"], Q, dtype, device)
            lengthscale[t] = ell_t
            signal_var[t] = sig_t
            noise_var[t] = noise_t
            w = model.get("w")
            if isinstance(w, torch.Tensor) and w.numel() == Q + 1:
                gate[t] = center_linear_gate(
                    w.detach().reshape(-1).to(device=device, dtype=dtype),
                    Z_anchor[t].to(device=device, dtype=dtype),
                )
            mu[t] = mu_t.detach().reshape(-1)[0].to(device=device, dtype=dtype)
            sigma[t] = torch.sqrt(torch.clamp(sig2_t.detach().reshape(-1)[0].to(device=device, dtype=dtype), min=1e-12))
        except Exception:
            failed += 1
            labels[t] = _response_seed_labels(
                X_anchor[t : t + 1],
                X_neighbors[t : t + 1],
                y_neighbors[t : t + 1],
                seed_frac=0.25,
                inlier_frac=0.70,
                min_inliers=2,
            )[0]
            mean_const[t] = fallback["mean_const"][t]
            lengthscale[t] = fallback["lengthscale"][t]
            signal_var[t] = fallback["signal_var"][t]
            noise_var[t] = fallback["noise_var"][t]
            mu[t] = mean_const[t]
            sigma[t] = torch.sqrt(noise_var[t].clamp_min(1e-12))
    state = {
        "labels": labels,
        "mean_const": mean_const,
        "lengthscale": lengthscale,
        "signal_var": signal_var,
        "noise_var": noise_var,
        "gate": gate,
        "mu": mu,
        "sigma": sigma,
    }
    diag = {
        "init_sec": float(time.perf_counter() - t0),
        "meanw_cem_failed": int(failed),
        "meanw_cem_inlier_rate": float(labels.to(dtype=dtype).mean().detach().cpu().item()),
    }
    return state, diag


def _init_state_for_variant(
    spec: VariantSpec,
    *,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors: torch.Tensor,
    W0: torch.Tensor,
    cem_state: dict[str, torch.Tensor],
    min_inliers: int,
    response_seed_frac: float,
    response_inlier_frac: float,
) -> dict[str, torch.Tensor]:
    if spec.init == "cem":
        return {k: v.detach().clone() for k, v in cem_state.items() if isinstance(v, torch.Tensor)}
    if spec.init == "response":
        labels = _response_seed_labels(
            X_anchor,
            X_neighbors,
            y_neighbors,
            seed_frac=response_seed_frac,
            inlier_frac=response_inlier_frac,
            min_inliers=min_inliers,
        )
    elif spec.init == "pairwise":
        labels = _pairwise_seed_labels(
            X_anchor,
            X_neighbors,
            y_neighbors,
            seed_frac=response_seed_frac,
            inlier_frac=response_inlier_frac,
            min_inliers=min_inliers,
        )
    else:
        raise ValueError(f"Unknown label init {spec.init!r}.")
    hp = _default_hypers_from_labels(X_anchor, X_neighbors, y_neighbors, labels, W0, min_inliers=min_inliers)
    T, Q = X_anchor.shape[0], W0.shape[0]
    gate = torch.zeros(T, Q + 1, device=X_anchor.device, dtype=X_anchor.dtype)
    return {"labels": labels, "gate": gate, **hp}


def _label_diag(init_labels: torch.Tensor, final_labels: torch.Tensor, n_total: int) -> dict[str, float]:
    init_f = init_labels.to(dtype=torch.float64)
    final_f = final_labels.to(dtype=torch.float64)
    n_in = final_f.sum(dim=1)
    return {
        "init_inlier_rate": float(init_f.mean().detach().cpu().item()),
        "final_inlier_rate": float(final_f.mean().detach().cpu().item()),
        "label_change_rate": float((init_labels.bool() != final_labels.bool()).to(dtype=torch.float64).mean().detach().cpu().item()),
        "mean_n_inliers": float(n_in.mean().detach().cpu().item()),
        "min_n_inliers": float(n_in.min().detach().cpu().item()),
        "median_n_inliers": float(n_in.median().detach().cpu().item()),
        "max_n_inliers": float(n_in.max().detach().cpu().item()),
        "n_total_neighbors": int(n_total),
    }


def _local_noise_var_floor(args: argparse.Namespace, spec: VariantSpec) -> float:
    if spec.local_noise_std_floor is None:
        return float(args.min_noise_var)
    return max(float(args.min_noise_var), float(spec.local_noise_std_floor) ** 2)


def _refresh_noise_var_floor(args: argparse.Namespace, spec: VariantSpec) -> float:
    noise_std = spec.refresh_noise_std_floor if spec.refresh_noise_std_floor is not None else args.refresh_noise_std_floor
    if noise_std is None:
        return float(args.min_noise_var)
    return max(float(args.min_noise_var), float(noise_std) ** 2)


def _refresh_damping(args: argparse.Namespace, spec: VariantSpec) -> float:
    alpha = float(spec.refresh_damping) if spec.refresh_damping is not None else float(args.refresh_damping)
    return float(min(max(alpha, 0.0), 1.0))


def _refresh_gate_l2(args: argparse.Namespace) -> float:
    return float(args.refresh_gate_l2) if args.refresh_gate_l2 is not None else float(args.gate_l2)


def _refresh_gate_max_norm(args: argparse.Namespace) -> float:
    return float(args.refresh_gate_max_norm) if args.refresh_gate_max_norm is not None else float(args.gate_max_norm)


def _qr_include_residual(args: argparse.Namespace, spec: VariantSpec) -> bool:
    return False if spec.qr_control == "meanonly" else bool(args.qr_include_residual)


def _qr_init_log_std(args: argparse.Namespace, spec: VariantSpec) -> float:
    return -10.0 if spec.qr_control == "meanonly" else float(args.qr_init_log_std)


def _qr_pairwise_weight(args: argparse.Namespace, spec: VariantSpec) -> float:
    return float(spec.qr_pairwise_weight) if spec.qr_pairwise_weight is not None else float(args.qr_pairwise_weight)


def _qr_rank_weight(args: argparse.Namespace, spec: VariantSpec) -> float:
    return float(spec.qr_rank_weight) if spec.qr_rank_weight is not None else float(args.qr_rank_weight)


def _qr_ortho_weight(args: argparse.Namespace, spec: VariantSpec) -> float:
    return float(spec.qr_ortho_weight) if spec.qr_ortho_weight is not None else float(args.qr_ortho_weight)


def _qr_mean_parameterization(args: argparse.Namespace, spec: VariantSpec) -> str:
    return str(spec.qr_mean_parameterization or args.qr_mean_parameterization)


def _qr_residual_group_weight(args: argparse.Namespace, spec: VariantSpec) -> float:
    return (
        float(spec.qr_residual_group_weight)
        if spec.qr_residual_group_weight is not None
        else float(args.qr_residual_group_weight)
    )


def _lowrank_train_basis_scale(spec: VariantSpec) -> bool:
    return bool(spec.lowrank_train_basis_scale) and int(spec.lowrank_rank) > 0


def _qr_anchor_pred_weight(args: argparse.Namespace, spec: VariantSpec) -> float:
    if _is_wknn_variant(spec):
        return float(args.qr_anchor_pred_weight)
    return 0.0


def _qr_composite_temperature(args: argparse.Namespace, spec: VariantSpec) -> float:
    if _is_wknn_variant(spec):
        return float(args.qr_retrieval_composite_temperature)
    return float(args.qr_composite_temperature)


def _qr_w_signal_var(args: argparse.Namespace, spec: VariantSpec) -> float:
    if int(spec.lowrank_rank) > 0:
        return float(args.lowrank_qr_w_signal_var)
    return float(args.qr_w_signal_var)


def _maybe_meanonly_log_std(R_log_std: torch.Tensor, spec: VariantSpec) -> torch.Tensor:
    if spec.qr_control == "meanonly":
        return torch.full_like(R_log_std, -10.0)
    return R_log_std


def _shuffle_w_moments_by_anchor(w_moments: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    first = next(iter(w_moments.values()))
    perm = torch.randperm(first.shape[0], device=first.device)
    return {key: value[perm].contiguous() for key, value in w_moments.items()}


def _prior_w_moments_for_mode(
    *,
    mode: str,
    W0: torch.Tensor,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    qr_w_lengthscale: float,
    qr_w_signal_var: float,
) -> dict[str, torch.Tensor]:
    T, n, D = X_neighbors.shape
    Q = W0.shape[0]
    dtype = X_neighbors.dtype
    device = X_neighbors.device
    signal = float(qr_w_signal_var)
    if mode == "anchor":
        return {
            "W_mu_anchor": torch.zeros(T, Q, D, device=device, dtype=dtype),
            "W_var_anchor": torch.full((T, Q, D), signal, device=device, dtype=dtype),
        }
    if mode == "pointwise":
        X_all = torch.cat([X_anchor.unsqueeze(1), X_neighbors], dim=1)
        diff = X_all.unsqueeze(2) - X_all.unsqueeze(1)
        Kpos = torch.exp(-0.5 * diff.pow(2).sum(dim=-1) / max(float(qr_w_lengthscale) ** 2, 1e-12)) * signal
        return {
            "W_mu_all": torch.zeros(T, n + 1, Q, D, device=device, dtype=dtype),
            "W_cov_all": Kpos[:, None, None, :, :].expand(T, Q, D, n + 1, n + 1).contiguous(),
        }
    raise ValueError(f"Unknown W mode {mode!r}.")


def _lowrank_prior_w_moments(
    *,
    mode: str,
    W0: torch.Tensor,
    V_basis: torch.Tensor,
    X_anchor: torch.Tensor,
    qr_w_signal_var: float,
) -> dict[str, torch.Tensor]:
    if mode != "anchor":
        raise NotImplementedError("Low-rank prior control currently supports anchor mode only.")
    T = X_anchor.shape[0]
    Q, D = W0.shape
    r = V_basis.shape[1]
    dtype = X_anchor.dtype
    device = X_anchor.device
    C_var = torch.full((T, Q, r), float(qr_w_signal_var), device=device, dtype=dtype)
    W_cov = torch.einsum("tqr,dr,er->tqde", C_var, V_basis, V_basis)
    return {
        "W_mu_anchor": W0.reshape(1, Q, D).expand(T, Q, D).contiguous(),
        "W_var_anchor": torch.diagonal(W_cov, dim1=-2, dim2=-1).contiguous(),
        "W_cov_anchor": W_cov,
    }


def _w_moments_from_qr_result(
    *,
    args: argparse.Namespace,
    spec: VariantSpec,
    qr: Any,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    W0: torch.Tensor,
    V_basis: torch.Tensor | None,
    qr_w_lengthscale: float,
) -> dict[str, torch.Tensor]:
    if int(spec.lowrank_rank) > 0:
        if V_basis is None:
            raise ValueError("Low-rank qR moments require V_basis.")
        V_eff = V_basis
        if getattr(qr, "basis_scale", None) is not None:
            scale = qr.basis_scale.to(device=V_basis.device, dtype=V_basis.dtype).reshape(1, -1)
            V_eff = V_basis * scale
        moments = sparse_gp_lowrank_residual_w_moments_for_mode(
            mode=spec.mode,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            inducing_X=qr.inducing_X,
            C_mu=qr.R_mu,
            C_log_std=_maybe_meanonly_log_std(qr.R_log_std, spec),
            base_W=W0,
            basis_V=V_eff,
            lengthscale=float(qr_w_lengthscale),
            signal_var=_qr_w_signal_var(args, spec),
            include_conditional_residual=_qr_include_residual(args, spec),
        )
    else:
        moments = sparse_gp_uncertain_w_moments_for_mode(
            mode=spec.mode,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            inducing_X=qr.inducing_X,
            R_mu=qr.R_mu,
            R_log_std=_maybe_meanonly_log_std(qr.R_log_std, spec),
            lengthscale=float(qr_w_lengthscale),
            signal_var=_qr_w_signal_var(args, spec),
            include_conditional_residual=_qr_include_residual(args, spec),
        )
    if spec.qr_control == "shuf":
        return _shuffle_w_moments_by_anchor(moments)
    return moments


def _repeat_first_dim(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    return (
        x.unsqueeze(0)
        .expand(int(n_repeat), *tuple(x.shape))
        .reshape(int(n_repeat) * int(x.shape[0]), *tuple(x.shape[1:]))
        .contiguous()
    )


def _sample_anchor_w_from_moments(
    w_moments: dict[str, torch.Tensor],
    *,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    if "W_mu_anchor" not in w_moments:
        raise ValueError("Deterministic-W ensemble currently requires anchor-mode W_mu_anchor moments.")
    W_mu = w_moments["W_mu_anchor"].detach()
    if int(n_samples) <= 1:
        return W_mu.unsqueeze(0).contiguous()
    if "W_var_anchor" in w_moments:
        W_var = w_moments["W_var_anchor"].detach().to(device=W_mu.device, dtype=W_mu.dtype)
    elif "W_cov_anchor" in w_moments:
        W_var = torch.diagonal(w_moments["W_cov_anchor"].detach(), dim1=-2, dim2=-1).to(
            device=W_mu.device,
            dtype=W_mu.dtype,
        )
    else:
        W_var = torch.zeros_like(W_mu)
    W_std = W_var.clamp_min(0.0).sqrt()
    try:
        gen = torch.Generator(device=W_mu.device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(seed))
    eps = torch.randn(
        (int(n_samples), *tuple(W_mu.shape)),
        device=W_mu.device,
        dtype=W_mu.dtype,
        generator=gen,
    )
    return (W_mu.unsqueeze(0) + eps * W_std.unsqueeze(0)).contiguous()


def _predict_deterministic_w_self_cem_ensemble(
    *,
    args: argparse.Namespace,
    spec: VariantSpec,
    seed: int,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors_std: torch.Tensor,
    test_init_state: dict[str, torch.Tensor],
    w_moments: dict[str, torch.Tensor],
    y_test: np.ndarray,
    y_mean: float,
    y_std: float,
) -> tuple[list[float], dict[str, Any], dict[str, torch.Tensor | list[dict[str, float]]], Any]:
    if spec.mode != "anchor":
        raise NotImplementedError("Deterministic-W CEM ensemble is implemented for anchor-mode variants.")
    n_samples = int(spec.ensemble_samples)
    if n_samples <= 0:
        raise ValueError("ensemble_samples must be positive for deterministic-W ensemble prediction.")
    T = int(X_anchor.shape[0])
    W_samples = _sample_anchor_w_from_moments(
        w_moments,
        n_samples=n_samples,
        seed=int(seed) + 977 * max(1, n_samples),
    )
    W_flat = W_samples.reshape(n_samples * T, *tuple(W_samples.shape[2:])).contiguous()
    zero_var = torch.zeros_like(W_flat)
    X_anchor_flat = _repeat_first_dim(X_anchor, n_samples)
    X_neighbors_flat = _repeat_first_dim(X_neighbors, n_samples)
    y_neighbors_flat = _repeat_first_dim(y_neighbors_std, n_samples)
    labels0 = _repeat_first_dim(test_init_state["labels"].bool(), n_samples)
    ell0 = _repeat_first_dim(test_init_state["lengthscale"], n_samples)
    sig0 = _repeat_first_dim(test_init_state["signal_var"], n_samples)
    noise0 = _repeat_first_dim(test_init_state["noise_var"], n_samples)
    gate0 = _repeat_first_dim(test_init_state["gate"], n_samples)

    t_cem = time.perf_counter()
    sample_state = run_uncertain_w_self_cem(
        X_anchor_flat,
        X_neighbors_flat,
        y_neighbors_flat,
        labels0,
        mode="anchor",
        init_lengthscale=ell0,
        init_signal_var=sig0,
        init_noise_var=noise0,
        gate_init=gate0,
        config=SelfCEMConfig(
            cem_updates=int(spec.cem_updates),
            hyperopt=FixedLabelHyperoptConfig(
                steps=int(args.refresh_hyper_steps),
                lr=float(args.hyper_lr),
                min_noise_var=float(_refresh_noise_var_floor(args, spec)),
                max_noise_var=float(args.max_noise_var),
                min_signal_var=float(args.min_signal_var),
                max_signal_var=float(args.max_signal_var),
                min_lengthscale=float(args.min_lengthscale),
                max_lengthscale=float(args.max_lengthscale),
            ),
            gate=FixedLabelGateConfig(
                steps=int(args.refresh_gate_steps),
                lr=float(args.gate_lr),
                l2=float(_refresh_gate_l2(args)),
                max_norm=float(_refresh_gate_max_norm(args)),
                gh_points=int(args.gh_points),
            ),
            outlier_mode="background_normal",
            outlier_sigma_mult=float(args.outlier_sigma_mult),
            background_scale_mult=float(args.background_scale_mult),
            min_inliers=int(args.min_inliers),
            max_inlier_frac=float(args.max_inlier_frac),
            refit_after_update=True,
        ),
        W_mu_anchor=W_flat,
        W_var_anchor=zero_var,
    )
    cem_sec = time.perf_counter() - t_cem

    t_pred = time.perf_counter()
    out = uncertain_w_cem_predict_from_labels(
        X_anchor_flat,
        X_neighbors_flat,
        y_neighbors_flat,
        sample_state["labels"],
        mode="anchor",
        lengthscale=sample_state["lengthscale"],
        signal_var=sample_state["signal_var"],
        noise_var=sample_state["noise_var"],
        mean_const=sample_state["mean_const"],
        correction_weight=0.0,
        min_inliers=int(args.min_inliers),
        W_mu_anchor=W_flat,
        W_var_anchor=zero_var,
    )
    predict_sec = time.perf_counter() - t_pred

    mu_samples_std = out.mu.detach().reshape(n_samples, T)
    var_samples_std = out.var.detach().reshape(n_samples, T).clamp_min(1e-12)
    mu_std = mu_samples_std.mean(dim=0)
    total_var_std = (var_samples_std + mu_samples_std.pow(2)).mean(dim=0) - mu_std.pow(2)
    total_var_std = total_var_std.clamp_min(1e-12)
    scale = abs(float(y_std))
    mu = mu_std.detach().cpu().numpy().reshape(-1) * float(y_std) + float(y_mean)
    sigma = total_var_std.sqrt().detach().cpu().numpy().reshape(-1) * scale
    pack = _metrics_pack(y_test, mu, sigma, cem_sec + predict_sec)
    within_sigma = var_samples_std.mean(dim=0).sqrt() * scale
    between_sigma = mu_samples_std.var(dim=0, unbiased=False).clamp_min(0.0).sqrt() * scale
    total_sigma = total_var_std.sqrt() * scale
    diag = {
        "ensemble_samples": int(n_samples),
        "ensemble_cem_sec": float(cem_sec),
        "ensemble_predict_sec": float(predict_sec),
        "ensemble_total_sec": float(cem_sec + predict_sec),
        "ensemble_within_sigma_mean": float(within_sigma.mean().detach().cpu().item()),
        "ensemble_between_sigma_mean": float(between_sigma.mean().detach().cpu().item()),
        "ensemble_total_sigma_mean": float(total_sigma.mean().detach().cpu().item()),
        "ensemble_mean_n_inliers": float(out.diagnostics["mean_n_inliers"]),
        "ensemble_fallback_count": int(out.diagnostics["fallback_count"]),
        "mean_kernel_vector_correction": 0.0,
        "fallback_count": int(out.diagnostics["fallback_count"]),
    }
    return pack, diag, sample_state, out


def _rbf_cross(
    X1: torch.Tensor,
    X2: torch.Tensor,
    *,
    lengthscale: float,
    signal_var: float,
) -> torch.Tensor:
    diff = X1.unsqueeze(-2) - X2.unsqueeze(-3)
    d2 = diff.pow(2).sum(dim=-1)
    return float(signal_var) * torch.exp(-0.5 * d2 / max(float(lengthscale) ** 2, 1e-12))


def _pointwise_pool_sqdist_from_qr_result(
    *,
    args: argparse.Namespace,
    spec: VariantSpec,
    qr: Any,
    X_anchor: torch.Tensor,
    X_pool: torch.Tensor,
    qr_w_lengthscale: float,
) -> torch.Tensor:
    """Compute E[||W(x_i)x_i - W(x_a)x_a||^2] without materializing full pool covariance."""
    if int(spec.lowrank_rank) > 0:
        raise NotImplementedError("Pointwise retrieval distance is implemented for the full q(R) W-field only.")
    if spec.qr_control == "shuf":
        raise NotImplementedError("Pointwise retrieval is not defined for shuffled-qR controls.")
    Xa = X_anchor
    Xp = X_pool
    U = qr.inducing_X.to(device=Xa.device, dtype=Xa.dtype)
    R_mu = qr.R_mu.to(device=Xa.device, dtype=Xa.dtype)
    R_var = (2.0 * _maybe_meanonly_log_std(qr.R_log_std, spec).to(device=Xa.device, dtype=Xa.dtype)).exp()
    T, P, Dx = Xp.shape
    M, Q, D = R_mu.shape
    if D != Dx:
        raise ValueError("q(R) output dimension is inconsistent with retrieval X dimension.")
    signal = _qr_w_signal_var(args, spec)
    jitter = float(getattr(args, "qr_jitter", 1e-5))
    Kuu = _rbf_cross(U, U, lengthscale=float(qr_w_lengthscale), signal_var=signal)
    Kuu = Kuu + jitter * torch.eye(M, device=Xa.device, dtype=Xa.dtype)
    Luu = torch.linalg.cholesky(Kuu)

    KaU = _rbf_cross(Xa, U, lengthscale=float(qr_w_lengthscale), signal_var=signal)
    Aa = torch.cholesky_solve(KaU.T, Luu).T
    Xflat = Xp.reshape(T * P, Dx)
    KpU_flat = _rbf_cross(Xflat, U, lengthscale=float(qr_w_lengthscale), signal_var=signal)
    Ap_flat = torch.cholesky_solve(KpU_flat.T, Luu).T
    KpU = KpU_flat.reshape(T, P, M)
    Ap = Ap_flat.reshape(T, P, M)

    mean_a = torch.einsum("tm,mqd->tqd", Aa, R_mu)
    mean_p = torch.einsum("tpm,mqd->tpqd", Ap, R_mu)
    var_a = torch.einsum("tm,mqd,tm->tqd", Aa, R_var, Aa)
    var_p = torch.einsum("tpm,mqd,tpm->tpqd", Ap, R_var, Ap)
    cov_ap = torch.einsum("tm,mqd,tpm->tpqd", Aa, R_var, Ap)

    if _qr_include_residual(args, spec):
        cond_aa = (float(signal) - (KaU * Aa).sum(dim=1)).clamp_min(0.0)
        cond_pp = (float(signal) - (KpU * Ap).sum(dim=2)).clamp_min(0.0)
        Kap = _rbf_cross(Xa.unsqueeze(1), Xp, lengthscale=float(qr_w_lengthscale), signal_var=signal).squeeze(1)
        cond_ap = Kap - torch.einsum("tm,tpm->tp", Aa, KpU)
        var_a = var_a + cond_aa[:, None, None]
        var_p = var_p + cond_pp[:, :, None, None]
        cov_ap = cov_ap + cond_ap[:, :, None, None]

    z_mean_a = torch.einsum("tqd,td->tq", mean_a, Xa)
    z_mean_p = torch.einsum("tpqd,tpd->tpq", mean_p, Xp)
    mean_sq = (z_mean_p - z_mean_a.unsqueeze(1)).pow(2).sum(dim=-1)
    var_diff = (
        Xp[:, :, None, :].pow(2) * var_p
        + Xa[:, None, None, :].pow(2) * var_a[:, None, :, :]
        - 2.0 * Xp[:, :, None, :] * Xa[:, None, None, :] * cov_ap
    ).sum(dim=(-1, -2))
    return (mean_sq + var_diff).clamp_min(0.0)


def _positive_damped(
    old: torch.Tensor,
    new: torch.Tensor,
    *,
    alpha: float,
    min_value: float,
    max_value: float,
) -> torch.Tensor:
    old = old.detach().clamp_min(1e-12)
    new = new.detach().clamp_min(1e-12)
    if alpha < 1.0:
        out = torch.exp((1.0 - alpha) * torch.log(old) + alpha * torch.log(new))
    else:
        out = new.clone()
    return out.clamp(float(min_value), float(max_value))


def _damp_refresh_state(
    *,
    old_state: dict[str, torch.Tensor],
    refreshed: dict[str, torch.Tensor],
    new_labels: torch.Tensor,
    args: argparse.Namespace,
    spec: VariantSpec,
) -> dict[str, torch.Tensor]:
    alpha = _refresh_damping(args, spec)
    noise_floor = _refresh_noise_var_floor(args, spec)
    lengthscale = _positive_damped(
        old_state["lengthscale"],
        refreshed["lengthscale"],
        alpha=alpha,
        min_value=float(args.min_lengthscale),
        max_value=float(args.max_lengthscale),
    )
    signal_var = _positive_damped(
        old_state["signal_var"],
        refreshed["signal_var"],
        alpha=alpha,
        min_value=float(args.min_signal_var),
        max_value=float(args.max_signal_var),
    )
    noise_var = _positive_damped(
        old_state["noise_var"],
        refreshed["noise_var"],
        alpha=alpha,
        min_value=noise_floor,
        max_value=float(args.max_noise_var),
    )
    if alpha < 1.0:
        mean_const = (1.0 - alpha) * old_state["mean_const"].detach() + alpha * refreshed["mean_const"].detach()
        gate = (1.0 - alpha) * old_state["gate"].detach() + alpha * refreshed["gate"].detach()
    else:
        mean_const = refreshed["mean_const"].detach().clone()
        gate = refreshed["gate"].detach().clone()
    max_norm = _refresh_gate_max_norm(args)
    if max_norm > 0:
        norm = torch.linalg.norm(gate, dim=1, keepdim=True).clamp_min(1e-12)
        gate = gate * torch.clamp(float(max_norm) / norm, max=1.0)
    return {
        "labels": new_labels,
        "lengthscale": lengthscale,
        "signal_var": signal_var,
        "noise_var": noise_var,
        "mean_const": mean_const,
        "gate": gate,
    }


def _base_row(args: argparse.Namespace, cfg: dict[str, Any], setting: str, seed: int, spec: VariantSpec) -> dict[str, Any]:
    local_noise_std_floor = (
        float(spec.local_noise_std_floor) if spec.local_noise_std_floor is not None else ""
    )
    local_noise_var_floor = (
        float(local_noise_std_floor) ** 2
        if local_noise_std_floor != ""
        else float(args.min_noise_var)
    )
    refresh_noise_std_floor = (
        float(spec.refresh_noise_std_floor)
        if spec.refresh_noise_std_floor is not None
        else (float(args.refresh_noise_std_floor) if args.refresh_noise_std_floor is not None else "")
    )
    refresh_noise_var_floor = (
        float(refresh_noise_std_floor) ** 2
        if refresh_noise_std_floor != ""
        else float(args.min_noise_var)
    )
    refresh_damping = (
        float(spec.refresh_damping)
        if spec.refresh_damping is not None
        else float(args.refresh_damping)
    )
    refresh_gate_l2 = float(args.refresh_gate_l2) if args.refresh_gate_l2 is not None else float(args.gate_l2)
    refresh_gate_max_norm = (
        float(args.refresh_gate_max_norm) if args.refresh_gate_max_norm is not None else float(args.gate_max_norm)
    )
    return {
        "setting": setting,
        "family": cfg["family"],
        "seed": int(seed),
        "variant": spec.name,
        "mode": spec.mode,
        "kind": spec.kind,
        "init": spec.init,
        "projection_init": str(args.projection_init),
        "kcov_lambda": float(spec.kcov),
        "cem_updates": int(spec.cem_updates),
        "local_noise_std_floor": local_noise_std_floor,
        "local_noise_var_floor": local_noise_var_floor,
        "refresh_every": int(spec.refresh_every),
        "refresh_noise_std_floor": refresh_noise_std_floor,
        "refresh_noise_var_floor": refresh_noise_var_floor,
        "refresh_damping": refresh_damping,
        "refresh_gate_l2": refresh_gate_l2,
        "refresh_gate_max_norm": refresh_gate_max_norm,
        "qr_control": str(spec.qr_control),
        "ensemble_samples": int(spec.ensemble_samples),
        "neighbor_mode": str(spec.neighbor_mode),
        "retrieval_lambda": float(spec.retrieval_lambda),
        "qr_retrieval_pool": int(args.qr_retrieval_pool),
        "qr_anchor_pred_weight": _qr_anchor_pred_weight(args, spec),
        "qr_retrieval_composite_temperature": _qr_composite_temperature(args, spec),
        "qr_rank_weight": _qr_rank_weight(args, spec),
        "qr_rank_temperature": float(args.qr_rank_temperature),
        "qr_outer_cycles": int(args.qr_outer_cycles),
        "N_train": int(cfg["N_train"]),
        "N_test": int(cfg["N_test"]),
        "n_test_eval": int(args.max_test if args.max_test is not None else cfg["N_test"]),
        "Q": int(cfg["Q"]),
        "observed_dim": int(cfg["H"]),
        "latent_dim": int(cfg["d"]),
        "Q_true": int(cfg["Q_true"]),
        "n_neighbors": int(args.n_neighbors or cfg["n"]),
        "m_inducing": int(args.m_inducing),
        "qr_train_anchor_count": int(args.qr_train_anchor_count),
        "qr_train_anchor_strategy": str(args.qr_train_anchor_strategy),
        "qr_train_n_neighbors": int(args.qr_train_n_neighbors or args.n_neighbors or cfg["n"]),
        "w_var": float(args.w_var),
        "w_cov_lengthscale": float(args.w_cov_lengthscale) if args.w_cov_lengthscale is not None else "",
        "qr_steps": int(args.qr_steps),
        "qr_beta_kl": float(args.qr_beta_kl),
        "qr_train_log_std": bool(args.qr_train_log_std),
        "qr_pairwise_weight": _qr_pairwise_weight(args, spec),
        "qr_pairwise_margin": float(args.qr_pairwise_margin),
        "qr_ortho_weight": _qr_ortho_weight(args, spec),
        "qr_ortho_target_scale": float(args.qr_ortho_target_scale),
        "qr_ortho_normalize": bool(args.qr_ortho_normalize),
        "qr_mean_parameterization": _qr_mean_parameterization(args, spec),
        "qr_residual_group_weight": _qr_residual_group_weight(args, spec),
        "qr_residual_group_eps": float(args.qr_residual_group_eps),
        "lowrank_rank": int(spec.lowrank_rank),
        "lowrank_train_basis_scale": _lowrank_train_basis_scale(spec),
        "w_moment_kind": str(spec.w_moment_kind),
        "lowrank_basis": str(args.lowrank_basis),
        "lowrank_qr_w_signal_var": float(args.lowrank_qr_w_signal_var),
        "lowrank_basis_scale_l1": float(args.lowrank_basis_scale_l1),
        "standardize_x": True,
        "standardize_y_for_head": True,
        "expansion": str(cfg.get("expansion", "rff")),
        "noise_std": float(cfg["noise_std"]),
        "noise_var": float(cfg["noise_var"]),
    }


def _fit_local_state_for_variant(
    args: argparse.Namespace,
    spec: VariantSpec,
    *,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors_std: torch.Tensor,
    W0: torch.Tensor,
    V_basis: torch.Tensor | None,
    w_moments: dict[str, torch.Tensor],
    cem_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[dict[str, float]], float]:
    init_state = _init_state_for_variant(
        spec,
        X_anchor=X_anchor,
        X_neighbors=X_neighbors,
        y_neighbors=y_neighbors_std,
        W0=W0,
        cem_state=cem_state,
        min_inliers=int(args.min_inliers),
        response_seed_frac=float(args.response_seed_frac),
        response_inlier_frac=float(args.response_inlier_frac),
    )
    init_labels = init_state["labels"].bool()
    t0 = time.perf_counter()
    local_kind = spec.local_kind or spec.kind
    local_noise_floor = _local_noise_var_floor(args, spec)
    if local_kind == "meanw_cem":
        final_state = {
            "labels": init_state["labels"].bool(),
            "lengthscale": init_state["lengthscale"],
            "signal_var": init_state["signal_var"],
            "noise_var": init_state["noise_var"].clamp_min(local_noise_floor),
            "mean_const": init_state["mean_const"],
            "gate": init_state["gate"],
        }
        history: list[dict[str, float]] = []
    elif local_kind == "fixed_label_hyperopt":
        hp = fit_uncertain_kernel_hyperparams_from_labels(
            X_anchor,
            X_neighbors,
            y_neighbors_std,
            init_labels,
            mode=spec.mode,
            init_lengthscale=init_state["lengthscale"],
            init_signal_var=init_state["signal_var"],
            init_noise_var=init_state["noise_var"],
            config=FixedLabelHyperoptConfig(
                steps=int(args.hyper_steps),
                lr=float(args.hyper_lr),
                min_noise_var=float(local_noise_floor),
                max_noise_var=float(args.max_noise_var),
                min_signal_var=float(args.min_signal_var),
                max_signal_var=float(args.max_signal_var),
                min_lengthscale=float(args.min_lengthscale),
                max_lengthscale=float(args.max_lengthscale),
            ),
            min_inliers=int(args.min_inliers),
            **w_moments,
        )
        final_state: dict[str, torch.Tensor] = {
            "labels": init_labels,
            "lengthscale": hp["lengthscale"],
            "signal_var": hp["signal_var"],
            "noise_var": hp["noise_var"],
            "mean_const": hp["mean_const"],
            "gate": init_state["gate"],
        }
        history: list[dict[str, float]] = []
    elif local_kind == "self_cem":
        final_state = run_uncertain_w_self_cem(
            X_anchor,
            X_neighbors,
            y_neighbors_std,
            init_labels,
            mode=spec.mode,
            init_lengthscale=init_state["lengthscale"],
            init_signal_var=init_state["signal_var"],
            init_noise_var=init_state["noise_var"],
            gate_init=init_state["gate"],
            config=SelfCEMConfig(
                cem_updates=int(spec.cem_updates),
                hyperopt=FixedLabelHyperoptConfig(
                    steps=int(args.hyper_steps),
                    lr=float(args.hyper_lr),
                    min_noise_var=float(local_noise_floor),
                    max_noise_var=float(args.max_noise_var),
                    min_signal_var=float(args.min_signal_var),
                    max_signal_var=float(args.max_signal_var),
                    min_lengthscale=float(args.min_lengthscale),
                    max_lengthscale=float(args.max_lengthscale),
                ),
                gate=FixedLabelGateConfig(
                    steps=int(args.gate_steps),
                    lr=float(args.gate_lr),
                    l2=float(args.gate_l2),
                    max_norm=float(args.gate_max_norm),
                    gh_points=int(args.gh_points),
                ),
                outlier_mode="background_normal",
                outlier_sigma_mult=float(args.outlier_sigma_mult),
                background_scale_mult=float(args.background_scale_mult),
                min_inliers=int(args.min_inliers),
                max_inlier_frac=float(args.max_inlier_frac),
                refit_after_update=True,
            ),
            **w_moments,
        )
        history = final_state.get("history", [])  # type: ignore[assignment]
    elif local_kind == "self_vem":
        final_state = run_uncertain_w_self_vem(
            X_anchor,
            X_neighbors,
            y_neighbors_std,
            init_labels.to(dtype=y_neighbors_std.dtype).clamp(1e-4, 1.0 - 1e-4),
            mode=spec.mode,
            init_lengthscale=init_state["lengthscale"],
            init_signal_var=init_state["signal_var"],
            init_noise_var=init_state["noise_var"],
            gate_init=init_state["gate"],
            config=SelfVEMConfig(
                vem_updates=int(spec.cem_updates),
                hyperopt=FixedLabelHyperoptConfig(
                    steps=int(args.hyper_steps),
                    lr=float(args.hyper_lr),
                    min_noise_var=float(local_noise_floor),
                    max_noise_var=float(args.max_noise_var),
                    min_signal_var=float(args.min_signal_var),
                    max_signal_var=float(args.max_signal_var),
                    min_lengthscale=float(args.min_lengthscale),
                    max_lengthscale=float(args.max_lengthscale),
                ),
                gate=FixedLabelGateConfig(
                    steps=int(args.gate_steps),
                    lr=float(args.gate_lr),
                    l2=float(args.gate_l2),
                    max_norm=float(args.gate_max_norm),
                    gh_points=int(args.gh_points),
                ),
                outlier_mode="background_normal",
                outlier_sigma_mult=float(args.outlier_sigma_mult),
                background_scale_mult=float(args.background_scale_mult),
                min_inliers=int(args.min_inliers),
                max_inlier_frac=float(args.max_inlier_frac),
                refit_after_update=True,
            ),
            **w_moments,
        )
        history = final_state.get("history", [])  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported local variant kind {local_kind!r}.")
    return final_state, init_labels, history, time.perf_counter() - t0


def _run_qr_segment(
    args: argparse.Namespace,
    spec: VariantSpec,
    *,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors_std: torch.Tensor,
    y_anchor_std: torch.Tensor | None = None,
    W0: torch.Tensor,
    V_basis: torch.Tensor | None,
    X_inducing: torch.Tensor,
    qr_w_lengthscale: float,
    final_state: dict[str, torch.Tensor],
    steps: int,
    init_R_mu: torch.Tensor | None,
    init_basis_scale: torch.Tensor | None = None,
) -> Any:
    return train_qr_uncertain_w_cem_mstep(
        X_anchor,
        X_neighbors,
        y_neighbors_std,
        final_state["labels"],
        y_anchor=y_anchor_std,
        mode=spec.mode,
        inducing_X=X_inducing,
        init_W=W0 if int(spec.lowrank_rank) > 0 else (None if init_R_mu is not None else W0),
        init_R_mu=init_R_mu,
        init_basis_scale=init_basis_scale,
        residual_basis=V_basis if int(spec.lowrank_rank) > 0 else None,
        gate=final_state["gate"],
        lengthscale=final_state["lengthscale"],
        signal_var=final_state["signal_var"],
        noise_var=final_state["noise_var"],
        mean_const=final_state["mean_const"],
        config=UncertainWQRMstepConfig(
            steps=int(steps),
            lr=float(args.qr_lr),
            beta_kl=float(args.qr_beta_kl),
            kl_warmup_steps=int(args.qr_kl_warmup_steps),
            composite_temperature=_qr_composite_temperature(args, spec),
            init_log_std=_qr_init_log_std(args, spec),
            train_log_std=False if spec.qr_control == "meanonly" else bool(args.qr_train_log_std),
            w_lengthscale=float(qr_w_lengthscale),
            w_signal_var=_qr_w_signal_var(args, spec),
            include_conditional_residual=_qr_include_residual(args, spec),
            outlier_mode="background_normal",
            outlier_sigma_mult=float(args.outlier_sigma_mult),
            background_scale_mult=float(args.background_scale_mult),
            gh_points=int(args.gh_points),
            min_inliers=int(args.min_inliers),
            log_interval=int(args.qr_log_interval),
            grad_clip_norm=float(args.qr_grad_clip_norm),
            pairwise_weight=_qr_pairwise_weight(args, spec),
            pairwise_margin=float(args.qr_pairwise_margin),
            pairwise_same_weight=float(args.qr_pairwise_same_weight),
            pairwise_cross_weight=float(args.qr_pairwise_cross_weight),
            pairwise_normalize=bool(args.qr_pairwise_normalize),
            rank_weight=_qr_rank_weight(args, spec),
            rank_temperature=float(args.qr_rank_temperature),
            rank_normalize=bool(args.qr_rank_normalize),
            anchor_pred_weight=_qr_anchor_pred_weight(args, spec),
            anchor_pred_var_floor=float(args.qr_anchor_pred_var_floor),
            anchor_pred_exclude_self=bool(args.qr_anchor_pred_exclude_self),
            anchor_self_tol=float(args.qr_anchor_self_tol),
            residual_group_weight=_qr_residual_group_weight(args, spec),
            residual_group_eps=float(args.qr_residual_group_eps),
            train_basis_scale=_lowrank_train_basis_scale(spec),
            basis_scale_l1_weight=float(args.lowrank_basis_scale_l1),
            ortho_weight=_qr_ortho_weight(args, spec),
            ortho_target_scale=float(args.qr_ortho_target_scale),
            ortho_normalize=bool(args.qr_ortho_normalize),
            mean_parameterization=_qr_mean_parameterization(args, spec),
        ),
    )


def _run_variant(
    args: argparse.Namespace,
    spec: VariantSpec,
    *,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors_std: torch.Tensor,
    y_test: np.ndarray,
    y_mean: float,
    y_std: float,
    W0: torch.Tensor,
    V_basis: torch.Tensor | None,
    w_moments: dict[str, torch.Tensor],
    cem_state: dict[str, torch.Tensor],
    X_inducing: torch.Tensor,
    qr_w_lengthscale: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    final_state, init_labels, history, local_train_sec = _fit_local_state_for_variant(
        args,
        spec,
        X_anchor=X_anchor,
        X_neighbors=X_neighbors,
        y_neighbors_std=y_neighbors_std,
        W0=W0,
        V_basis=V_basis,
        w_moments=w_moments,
        cem_state=cem_state,
    )
    train_sec = float(local_train_sec)
    pred_w_moments = w_moments
    qr_diag: dict[str, Any] = {}
    qr_history: list[dict[str, float]] = []
    refresh_history: list[dict[str, float]] = []
    if spec.kind == "qr_mstep":
        t_qr = time.perf_counter()
        qr = _run_qr_segment(
            args,
            spec,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            y_neighbors_std=y_neighbors_std,
            W0=W0,
            V_basis=V_basis,
            X_inducing=X_inducing,
            qr_w_lengthscale=qr_w_lengthscale,
            final_state=final_state,
            steps=int(args.qr_steps),
            init_R_mu=None,
        )
        qr_train_sec = time.perf_counter() - t_qr
        train_sec += float(qr_train_sec)
        pred_w_moments = _w_moments_from_qr_result(
            args=args,
            spec=spec,
            qr=qr,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            W0=W0,
            V_basis=V_basis,
            qr_w_lengthscale=float(qr_w_lengthscale),
        )
        qr_history = qr.history
        qr_diag = {
            "qr_train_sec": float(qr.train_sec),
            "qr_best_loss": float(qr.diagnostics["best_loss"]),
            "qr_kl_per_anchor": float(qr.diagnostics["kl_per_anchor"]),
            "qr_R_std_median": float(qr.diagnostics["R_std_median"]),
            "qr_w_lengthscale": float(qr_w_lengthscale),
            "qr_control": str(spec.qr_control),
        }
        for key, value in qr.diagnostics.items():
            if key.startswith(("grad_", "pairwise_", "rank_", "anchor_pred_", "residual_group_", "basis_scale_", "ortho_", "svd_")):
                qr_diag[f"qr_{key}"] = value
    elif spec.kind == "qr_alt":
        current_R_mu: torch.Tensor | None = None
        current_basis_scale: torch.Tensor | None = None
        qr_train_total = 0.0
        refresh_train_total = 0.0
        cycles = max(1, int(args.qr_outer_cycles))
        segment_steps = max(1, int(spec.refresh_every))
        qr = None
        for outer in range(cycles):
            if spec.qr_control == "prior":
                if int(spec.lowrank_rank) > 0:
                    if V_basis is None:
                        raise ValueError("Low-rank prior control requires V_basis.")
                    pred_w_moments = _lowrank_prior_w_moments(
                        mode=spec.mode,
                        W0=W0,
                        V_basis=V_basis,
                        X_anchor=X_anchor,
                        qr_w_signal_var=_qr_w_signal_var(args, spec),
                    )
                else:
                    pred_w_moments = _prior_w_moments_for_mode(
                        mode=spec.mode,
                        W0=W0,
                        X_anchor=X_anchor,
                        X_neighbors=X_neighbors,
                        qr_w_lengthscale=float(qr_w_lengthscale),
                        qr_w_signal_var=_qr_w_signal_var(args, spec),
                    )
            else:
                t_qr = time.perf_counter()
                qr = _run_qr_segment(
                    args,
                    spec,
                    X_anchor=X_anchor,
                    X_neighbors=X_neighbors,
                    y_neighbors_std=y_neighbors_std,
                    W0=W0,
                    V_basis=V_basis,
                    X_inducing=X_inducing,
                    qr_w_lengthscale=qr_w_lengthscale,
                    final_state=final_state,
                    steps=segment_steps,
                    init_R_mu=current_R_mu,
                    init_basis_scale=current_basis_scale,
                )
                qr_train_total += time.perf_counter() - t_qr
                current_R_mu = qr.R_mu.detach()
                current_basis_scale = None if qr.basis_scale is None else qr.basis_scale.detach()
                for h in qr.history:
                    hh = dict(h)
                    hh["outer"] = float(outer + 1)
                    qr_history.append(hh)
                pred_w_moments = _w_moments_from_qr_result(
                    args=args,
                    spec=spec,
                    qr=qr,
                    X_anchor=X_anchor,
                    X_neighbors=X_neighbors,
                    W0=W0,
                    V_basis=V_basis,
                    qr_w_lengthscale=float(qr_w_lengthscale),
                )
            if outer < cycles - 1:
                prev_labels = final_state["labels"].bool()
                refresh_noise_floor = _refresh_noise_var_floor(args, spec)
                refresh_alpha = _refresh_damping(args, spec)
                refresh_l2 = _refresh_gate_l2(args)
                refresh_max_norm = _refresh_gate_max_norm(args)
                t_refresh = time.perf_counter()
                refreshed = run_uncertain_w_self_cem(
                    X_anchor,
                    X_neighbors,
                    y_neighbors_std,
                    prev_labels,
                    mode=spec.mode,
                    init_lengthscale=final_state["lengthscale"],
                    init_signal_var=final_state["signal_var"],
                    init_noise_var=final_state["noise_var"],
                    gate_init=final_state["gate"],
                    config=SelfCEMConfig(
                        cem_updates=int(spec.cem_updates),
                        hyperopt=FixedLabelHyperoptConfig(
                            steps=int(args.refresh_hyper_steps),
                            lr=float(args.hyper_lr),
                            min_noise_var=float(refresh_noise_floor),
                            max_noise_var=float(args.max_noise_var),
                            min_signal_var=float(args.min_signal_var),
                            max_signal_var=float(args.max_signal_var),
                            min_lengthscale=float(args.min_lengthscale),
                            max_lengthscale=float(args.max_lengthscale),
                        ),
                        gate=FixedLabelGateConfig(
                            steps=int(args.refresh_gate_steps),
                            lr=float(args.gate_lr),
                            l2=float(refresh_l2),
                            max_norm=float(refresh_max_norm),
                            gh_points=int(args.gh_points),
                        ),
                        outlier_mode="background_normal",
                        outlier_sigma_mult=float(args.outlier_sigma_mult),
                        background_scale_mult=float(args.background_scale_mult),
                        min_inliers=int(args.min_inliers),
                        max_inlier_frac=float(args.max_inlier_frac),
                        refit_after_update=True,
                    ),
                    **pred_w_moments,
                )
                refresh_train_total += time.perf_counter() - t_refresh
                new_labels = refreshed["labels"].bool()
                change = float((new_labels != prev_labels).to(dtype=torch.float64).mean().cpu().item())
                raw_noise = float(refreshed["noise_var"].detach().mean().cpu().item())
                final_state = _damp_refresh_state(
                    old_state=final_state,
                    refreshed=refreshed,
                    new_labels=new_labels,
                    args=args,
                    spec=spec,
                )
                refresh_history.append(
                    {
                        "outer": float(outer + 1),
                        "label_change_rate": change,
                        "inlier_rate": float(new_labels.to(dtype=torch.float64).mean().cpu().item()),
                        "refresh_noise_var_floor": float(refresh_noise_floor),
                        "refresh_damping": float(refresh_alpha),
                        "raw_mean_noise_var": raw_noise,
                        "mean_lengthscale": float(final_state["lengthscale"].detach().mean().cpu().item()),
                        "mean_signal_var": float(final_state["signal_var"].detach().mean().cpu().item()),
                        "mean_noise_var": float(final_state["noise_var"].detach().mean().cpu().item()),
                        "gate_norm_mean": float(torch.linalg.norm(final_state["gate"].detach(), dim=1).mean().cpu().item()),
                    }
                )
        train_sec += float(qr_train_total + refresh_train_total)
        if qr is not None:
            qr_diag = {
                "qr_train_sec": float(qr_train_total),
                "refresh_train_sec": float(refresh_train_total),
                "qr_best_loss": float(qr.diagnostics["best_loss"]),
                "qr_kl_per_anchor": float(qr.diagnostics["kl_per_anchor"]),
                "qr_R_std_median": float(qr.diagnostics["R_std_median"]),
                "qr_w_lengthscale": float(qr_w_lengthscale),
                "qr_control": str(spec.qr_control),
            }
            for key, value in qr.diagnostics.items():
                if key.startswith(("grad_", "pairwise_", "rank_", "anchor_pred_", "residual_group_", "basis_scale_", "ortho_", "svd_")):
                    qr_diag[f"qr_{key}"] = value
        else:
            qr_diag = {
                "qr_train_sec": float(qr_train_total),
                "refresh_train_sec": float(refresh_train_total),
                "qr_w_lengthscale": float(qr_w_lengthscale),
                "qr_control": str(spec.qr_control),
            }
        if refresh_history:
            changes = [r["label_change_rate"] for r in refresh_history]
            qr_diag.update(
                {
                    "refresh_count": float(len(refresh_history)),
                    "refresh_label_change_mean": float(np.mean(changes)),
                    "refresh_label_change_final": float(changes[-1]),
                    "refresh_label_change_max": float(np.max(changes)),
                    "refresh_inlier_rate_final": float(refresh_history[-1]["inlier_rate"]),
                    "refresh_noise_var_floor_active": float(refresh_history[-1]["refresh_noise_var_floor"]),
                    "refresh_damping_active": float(refresh_history[-1]["refresh_damping"]),
                    "refresh_gate_norm_final": float(refresh_history[-1]["gate_norm_mean"]),
                }
            )

    t1 = time.perf_counter()
    out = uncertain_w_cem_predict_from_labels(
        X_anchor,
        X_neighbors,
        y_neighbors_std,
        final_state["labels"],
        mode=spec.mode,
        lengthscale=final_state["lengthscale"],
        signal_var=final_state["signal_var"],
        noise_var=final_state["noise_var"],
        mean_const=final_state["mean_const"],
        correction_weight=float(spec.kcov),
        min_inliers=int(args.min_inliers),
        **pred_w_moments,
    )
    pred_sec = time.perf_counter() - t1
    mu = out.mu.detach().cpu().numpy().reshape(-1) * float(y_std) + float(y_mean)
    sigma = out.sigma.detach().cpu().numpy().reshape(-1) * abs(float(y_std))
    pack = _metrics_pack(y_test, mu, sigma, pred_sec)
    final_label_tensor = final_state["labels"].detach()
    final_labels = final_label_tensor > 0.5
    diag = _label_diag(init_labels, final_labels, X_neighbors.shape[1])
    if bool(((final_label_tensor > 1e-6) & (final_label_tensor < 1.0 - 1e-6)).any().item()):
        resp = final_label_tensor.clamp(1e-12, 1.0 - 1e-12)
        diag.update(
            {
                "soft_inlier_frac": float(resp.mean().cpu().item()),
                "soft_entropy_mean": float((-(resp * resp.log() + (1.0 - resp) * (1.0 - resp).log())).mean().cpu().item()),
            }
        )
    gate = final_state.get("gate")
    gate_norm = float("nan")
    if isinstance(gate, torch.Tensor):
        gate_norm = float(torch.linalg.norm(gate.detach(), dim=1).mean().cpu().item())
    diag.update(
        {
            "train_sec": float(train_sec),
            "pred_sec": float(pred_sec),
            "gate_norm_mean": gate_norm,
            "mean_lengthscale": float(final_state["lengthscale"].detach().mean().cpu().item()),
            "mean_signal_var": float(final_state["signal_var"].detach().mean().cpu().item()),
            "mean_noise_var": float(final_state["noise_var"].detach().mean().cpu().item()),
            "mean_kernel_vector_correction": float(out.diagnostics["mean_kernel_vector_correction"]),
            "fallback_count": int(out.diagnostics["fallback_count"]),
        }
    )
    diag.update({k: sanitize_for_json(v) for k, v in qr_diag.items()})
    row = {key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)}
    row.update(diag)
    row["history_json"] = json.dumps(sanitize_for_json(history), sort_keys=True)
    row["qr_history_json"] = json.dumps(sanitize_for_json(qr_history), sort_keys=True)
    row["refresh_history_json"] = json.dumps(sanitize_for_json(refresh_history), sort_keys=True)
    return row, {"history": history, "qr_history": qr_history, "refresh_history": refresh_history}


def _run_train_anchor_qr_alt_variant(
    args: argparse.Namespace,
    spec: VariantSpec,
    *,
    seed: int,
    X_train_np: np.ndarray,
    y_train_std_np: np.ndarray,
    X_train_t: torch.Tensor,
    y_train_std_t: torch.Tensor,
    X_anchor: torch.Tensor,
    X_neighbors: torch.Tensor,
    y_neighbors_std: torch.Tensor,
    test_neighbor_idx: torch.Tensor,
    y_test: np.ndarray,
    y_mean: float,
    y_std: float,
    W0: torch.Tensor,
    V_basis: torch.Tensor | None,
    test_w_moments: dict[str, torch.Tensor],
    test_cem_state: dict[str, torch.Tensor],
    X_inducing: torch.Tensor,
    qr_w_lengthscale: float,
    w_cov_lengthscale: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if spec.kind != "qr_alt":
        raise ValueError("Train-anchor qR path is currently implemented for qr_alt variants only.")
    if spec.mode != "anchor":
        raise NotImplementedError("Train-anchor qR path currently supports anchor-conditioned W only.")
    A = int(args.qr_train_anchor_count)
    if A <= 0:
        raise ValueError("qr_train_anchor_count must be positive for train-anchor qR.")
    train_anchor_idx_np = _select_train_anchor_indices(
        X_train_np,
        y_train_std_np,
        A,
        strategy=str(args.qr_train_anchor_strategy),
        seed=int(seed) + 31337,
    )
    train_anchor_idx = torch.as_tensor(train_anchor_idx_np, device=X_train_t.device, dtype=torch.long)
    X_qr_anchor = X_train_t[train_anchor_idx].contiguous()
    y_qr_anchor_std = y_train_std_t[train_anchor_idx].contiguous()
    qr_n = int(args.qr_train_n_neighbors or X_neighbors.shape[1])
    retrieval_history: list[dict[str, float]] = []
    qr_pool_idx: torch.Tensor | None = None
    if _is_wknn_variant(spec):
        qr_pool_n = _retrieval_pool_size(args, n_neighbors=qr_n, n_train=X_train_t.shape[0])
        X_qr_pool, y_qr_pool_std, qr_pool_idx = _neighbor_stack(
            X_qr_anchor,
            X_train_t,
            y_train_std_t,
            n_neighbors=qr_pool_n,
        )
        qr_pool_w_moments0 = _make_w_moments(
            mode=spec.mode,
            W0=W0,
            X_anchor=X_qr_anchor,
            X_neighbors=X_qr_pool,
            w_var=float(args.w_var),
            w_cov_lengthscale=float(w_cov_lengthscale),
        )
        X_qr_neighbors, y_qr_neighbors_std, qr_neighbor_idx, init_wknn_diag = _select_wknn_neighbors(
            X_qr_anchor,
            X_qr_pool,
            y_qr_pool_std,
            qr_pool_idx,
            n_neighbors=qr_n,
            mode=spec.mode,
            w_moments=qr_pool_w_moments0,
            spec=spec,
            W0=W0,
        )
        init_wknn_diag["outer"] = 0.0
        retrieval_history.append(init_wknn_diag)
    else:
        X_qr_neighbors, y_qr_neighbors_std, qr_neighbor_idx = _neighbor_stack(
            X_qr_anchor,
            X_train_t,
            y_train_std_t,
            n_neighbors=qr_n,
        )
    qr_w_moments0 = _make_w_moments(
        mode=spec.mode,
        W0=W0,
        X_anchor=X_qr_anchor,
        X_neighbors=X_qr_neighbors,
        w_var=float(args.w_var),
        w_cov_lengthscale=float(w_cov_lengthscale),
    )
    qr_cem_state, qr_cem_diag = _run_meanw_cem_init(X_qr_anchor, X_qr_neighbors, y_qr_neighbors_std, W0)
    final_state, init_labels, history, local_train_sec = _fit_local_state_for_variant(
        args,
        spec,
        X_anchor=X_qr_anchor,
        X_neighbors=X_qr_neighbors,
        y_neighbors_std=y_qr_neighbors_std,
        W0=W0,
        V_basis=V_basis,
        w_moments=qr_w_moments0,
        cem_state=qr_cem_state,
    )
    train_sec = float(local_train_sec + qr_cem_diag.get("init_sec", 0.0))
    pred_w_moments_qr = qr_w_moments0
    current_R_mu: torch.Tensor | None = None
    current_basis_scale: torch.Tensor | None = None
    qr_train_total = 0.0
    refresh_train_total = 0.0
    qr_history: list[dict[str, float]] = []
    refresh_history: list[dict[str, float]] = []
    qr = None
    cycles = max(1, int(args.qr_outer_cycles))
    segment_steps = max(1, int(spec.refresh_every))
    for outer in range(cycles):
        if spec.qr_control == "prior":
            if int(spec.lowrank_rank) > 0:
                if V_basis is None:
                    raise ValueError("Low-rank prior control requires V_basis.")
                pred_w_moments_qr = _lowrank_prior_w_moments(
                    mode=spec.mode,
                    W0=W0,
                    V_basis=V_basis,
                    X_anchor=X_qr_anchor,
                    qr_w_signal_var=_qr_w_signal_var(args, spec),
                )
            else:
                pred_w_moments_qr = _prior_w_moments_for_mode(
                    mode=spec.mode,
                    W0=W0,
                    X_anchor=X_qr_anchor,
                    X_neighbors=X_qr_neighbors,
                    qr_w_lengthscale=float(qr_w_lengthscale),
                    qr_w_signal_var=_qr_w_signal_var(args, spec),
                )
        else:
            t_qr = time.perf_counter()
            qr = _run_qr_segment(
                args,
                spec,
                X_anchor=X_qr_anchor,
                X_neighbors=X_qr_neighbors,
                y_neighbors_std=y_qr_neighbors_std,
                y_anchor_std=y_qr_anchor_std,
                W0=W0,
                V_basis=V_basis,
                X_inducing=X_inducing,
                qr_w_lengthscale=qr_w_lengthscale,
                final_state=final_state,
                steps=segment_steps,
                init_R_mu=current_R_mu,
                init_basis_scale=current_basis_scale,
            )
            qr_train_total += time.perf_counter() - t_qr
            current_R_mu = qr.R_mu.detach()
            current_basis_scale = None if qr.basis_scale is None else qr.basis_scale.detach()
            for h in qr.history:
                hh = dict(h)
                hh["outer"] = float(outer + 1)
                qr_history.append(hh)
            pred_w_moments_qr = _w_moments_from_qr_result(
                args=args,
                spec=spec,
                qr=qr,
                X_anchor=X_qr_anchor,
                X_neighbors=X_qr_neighbors,
                W0=W0,
                V_basis=V_basis,
                qr_w_lengthscale=float(qr_w_lengthscale),
            )
        if outer < cycles - 1:
            refresh_noise_floor = _refresh_noise_var_floor(args, spec)
            refresh_alpha = _refresh_damping(args, spec)
            refresh_l2 = _refresh_gate_l2(args)
            refresh_max_norm = _refresh_gate_max_norm(args)
            wknn_diag: dict[str, float] = {}
            if _is_wknn_variant(spec):
                if qr_pool_idx is None:
                    raise RuntimeError("W-KNN retrieval requires a fixed qR candidate pool.")
                prev_neighbor_idx = qr_neighbor_idx
                pointwise_dist = None
                if _uses_pointwise_retrieval(spec) and qr is not None:
                    pointwise_dist = _pointwise_pool_sqdist_from_qr_result(
                        args=args,
                        spec=spec,
                        qr=qr,
                        X_anchor=X_qr_anchor,
                        X_pool=X_qr_pool,
                        qr_w_lengthscale=float(qr_w_lengthscale),
                    )
                X_qr_neighbors, y_qr_neighbors_std, qr_neighbor_idx, wknn_diag = _select_wknn_neighbors(
                    X_qr_anchor,
                    X_qr_pool,
                    y_qr_pool_std,
                    qr_pool_idx,
                    n_neighbors=qr_n,
                    mode=spec.mode,
                    w_moments=pred_w_moments_qr,
                    spec=spec,
                    W0=W0,
                    pointwise_dist=pointwise_dist,
                )
                neighbor_overlap = (qr_neighbor_idx.unsqueeze(2) == prev_neighbor_idx.unsqueeze(1)).any(dim=2)
                wknn_diag["wknn_prev_selected_overlap"] = float(neighbor_overlap.to(dtype=torch.float64).mean().cpu().item())
                if spec.qr_control == "prior":
                    if int(spec.lowrank_rank) > 0:
                        if V_basis is None:
                            raise ValueError("Low-rank prior control requires V_basis.")
                        pred_w_moments_qr = _lowrank_prior_w_moments(
                            mode=spec.mode,
                            W0=W0,
                            V_basis=V_basis,
                            X_anchor=X_qr_anchor,
                            qr_w_signal_var=_qr_w_signal_var(args, spec),
                        )
                    else:
                        pred_w_moments_qr = _prior_w_moments_for_mode(
                            mode=spec.mode,
                            W0=W0,
                            X_anchor=X_qr_anchor,
                            X_neighbors=X_qr_neighbors,
                            qr_w_lengthscale=float(qr_w_lengthscale),
                            qr_w_signal_var=_qr_w_signal_var(args, spec),
                        )
                elif qr is not None:
                    pred_w_moments_qr = _w_moments_from_qr_result(
                        args=args,
                        spec=spec,
                        qr=qr,
                        X_anchor=X_qr_anchor,
                        X_neighbors=X_qr_neighbors,
                        W0=W0,
                        V_basis=V_basis,
                        qr_w_lengthscale=float(qr_w_lengthscale),
                    )
                else:
                    pred_w_moments_qr = _make_w_moments(
                        mode=spec.mode,
                        W0=W0,
                        X_anchor=X_qr_anchor,
                        X_neighbors=X_qr_neighbors,
                        w_var=float(args.w_var),
                        w_cov_lengthscale=float(w_cov_lengthscale),
                    )
                refresh_seed_state, refresh_seed_diag = _run_meanw_cem_init(X_qr_anchor, X_qr_neighbors, y_qr_neighbors_std, W0)
                refresh_train_total += float(refresh_seed_diag.get("init_sec", 0.0))
                prev_labels = refresh_seed_state["labels"].bool()
                init_lengthscale = refresh_seed_state["lengthscale"]
                init_signal_var = refresh_seed_state["signal_var"]
                init_noise_var = refresh_seed_state["noise_var"].clamp_min(refresh_noise_floor)
                init_gate = refresh_seed_state["gate"]
            else:
                prev_labels = final_state["labels"].bool()
                init_lengthscale = final_state["lengthscale"]
                init_signal_var = final_state["signal_var"]
                init_noise_var = final_state["noise_var"]
                init_gate = final_state["gate"]
            t_refresh = time.perf_counter()
            refreshed = run_uncertain_w_self_cem(
                X_qr_anchor,
                X_qr_neighbors,
                y_qr_neighbors_std,
                prev_labels,
                mode=spec.mode,
                init_lengthscale=init_lengthscale,
                init_signal_var=init_signal_var,
                init_noise_var=init_noise_var,
                gate_init=init_gate,
                config=SelfCEMConfig(
                    cem_updates=int(spec.cem_updates),
                    hyperopt=FixedLabelHyperoptConfig(
                        steps=int(args.refresh_hyper_steps),
                        lr=float(args.hyper_lr),
                        min_noise_var=float(refresh_noise_floor),
                        max_noise_var=float(args.max_noise_var),
                        min_signal_var=float(args.min_signal_var),
                        max_signal_var=float(args.max_signal_var),
                        min_lengthscale=float(args.min_lengthscale),
                        max_lengthscale=float(args.max_lengthscale),
                    ),
                    gate=FixedLabelGateConfig(
                        steps=int(args.refresh_gate_steps),
                        lr=float(args.gate_lr),
                        l2=float(refresh_l2),
                        max_norm=float(refresh_max_norm),
                        gh_points=int(args.gh_points),
                    ),
                    outlier_mode="background_normal",
                    outlier_sigma_mult=float(args.outlier_sigma_mult),
                    background_scale_mult=float(args.background_scale_mult),
                    min_inliers=int(args.min_inliers),
                    max_inlier_frac=float(args.max_inlier_frac),
                    refit_after_update=True,
                ),
                **pred_w_moments_qr,
            )
            refresh_train_total += time.perf_counter() - t_refresh
            new_labels = refreshed["labels"].bool()
            if wknn_diag:
                wknn_diag["wknn_selected_inlier_frac"] = float(new_labels.to(dtype=torch.float64).mean().cpu().item())
            change = float((new_labels != prev_labels).to(dtype=torch.float64).mean().cpu().item())
            raw_noise = float(refreshed["noise_var"].detach().mean().cpu().item())
            final_state = _damp_refresh_state(
                old_state=final_state,
                refreshed=refreshed,
                new_labels=new_labels,
                args=args,
                spec=spec,
            )
            refresh_history.append(
                {
                    "outer": float(outer + 1),
                    "label_change_rate": change,
                    "inlier_rate": float(new_labels.to(dtype=torch.float64).mean().cpu().item()),
                    "refresh_noise_var_floor": float(refresh_noise_floor),
                    "refresh_damping": float(refresh_alpha),
                    "raw_mean_noise_var": raw_noise,
                    "mean_lengthscale": float(final_state["lengthscale"].detach().mean().cpu().item()),
                    "mean_signal_var": float(final_state["signal_var"].detach().mean().cpu().item()),
                    "mean_noise_var": float(final_state["noise_var"].detach().mean().cpu().item()),
                    "gate_norm_mean": float(torch.linalg.norm(final_state["gate"].detach(), dim=1).mean().cpu().item()),
                    **wknn_diag,
                }
            )
            if wknn_diag:
                rec = dict(wknn_diag)
                rec["outer"] = float(outer + 1)
                retrieval_history.append(rec)

    train_sec += float(qr_train_total + refresh_train_total)
    if spec.qr_control == "prior":
        if int(spec.lowrank_rank) > 0:
            if V_basis is None:
                raise ValueError("Low-rank prior control requires V_basis.")
            pred_w_moments_test = _lowrank_prior_w_moments(
                mode=spec.mode,
                W0=W0,
                V_basis=V_basis,
                X_anchor=X_anchor,
                qr_w_signal_var=_qr_w_signal_var(args, spec),
            )
        else:
            pred_w_moments_test = _prior_w_moments_for_mode(
                mode=spec.mode,
                W0=W0,
                X_anchor=X_anchor,
                X_neighbors=X_neighbors,
                qr_w_lengthscale=float(qr_w_lengthscale),
                qr_w_signal_var=_qr_w_signal_var(args, spec),
            )
    elif qr is None:
        pred_w_moments_test = test_w_moments
    else:
        pred_w_moments_test = _w_moments_from_qr_result(
            args=args,
            spec=spec,
            qr=qr,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            W0=W0,
            V_basis=V_basis,
            qr_w_lengthscale=float(qr_w_lengthscale),
        )

    test_retrieval_diag: dict[str, float] = {}
    test_pool_idx: torch.Tensor | None = None
    if _is_wknn_variant(spec):
        test_pool_n = _retrieval_pool_size(args, n_neighbors=X_neighbors.shape[1], n_train=X_train_t.shape[0])
        X_test_pool, y_test_pool_std, test_pool_idx = _neighbor_stack(
            X_anchor,
            X_train_t,
            y_train_std_t,
            n_neighbors=test_pool_n,
        )
        test_pointwise_dist = None
        if _uses_pointwise_retrieval(spec) and qr is not None:
            test_pointwise_dist = _pointwise_pool_sqdist_from_qr_result(
                args=args,
                spec=spec,
                qr=qr,
                X_anchor=X_anchor,
                X_pool=X_test_pool,
                qr_w_lengthscale=float(qr_w_lengthscale),
            )
        X_neighbors, y_neighbors_std, test_neighbor_idx, test_retrieval_diag = _select_wknn_neighbors(
            X_anchor,
            X_test_pool,
            y_test_pool_std,
            test_pool_idx,
            n_neighbors=int(args.n_neighbors or X_neighbors.shape[1]),
            mode=spec.mode,
            w_moments=pred_w_moments_test,
            spec=spec,
            W0=W0,
            pointwise_dist=test_pointwise_dist,
        )
        if spec.qr_control == "prior":
            if int(spec.lowrank_rank) > 0:
                if V_basis is None:
                    raise ValueError("Low-rank prior control requires V_basis.")
                pred_w_moments_test = _lowrank_prior_w_moments(
                    mode=spec.mode,
                    W0=W0,
                    V_basis=V_basis,
                    X_anchor=X_anchor,
                    qr_w_signal_var=_qr_w_signal_var(args, spec),
                )
            else:
                pred_w_moments_test = _prior_w_moments_for_mode(
                    mode=spec.mode,
                    W0=W0,
                    X_anchor=X_anchor,
                    X_neighbors=X_neighbors,
                    qr_w_lengthscale=float(qr_w_lengthscale),
                    qr_w_signal_var=_qr_w_signal_var(args, spec),
                )
        elif qr is not None:
            pred_w_moments_test = _w_moments_from_qr_result(
                args=args,
                spec=spec,
                qr=qr,
                X_anchor=X_anchor,
                X_neighbors=X_neighbors,
                W0=W0,
                V_basis=V_basis,
                qr_w_lengthscale=float(qr_w_lengthscale),
            )
        else:
            pred_w_moments_test = _make_w_moments(
                mode=spec.mode,
                W0=W0,
                X_anchor=X_anchor,
                X_neighbors=X_neighbors,
                w_var=float(args.w_var),
                w_cov_lengthscale=float(w_cov_lengthscale),
            )
        test_cem_state, test_cem_diag = _run_meanw_cem_init(X_anchor, X_neighbors, y_neighbors_std, W0)
        train_sec += float(test_cem_diag.get("init_sec", 0.0))

    test_init_state = {
        "labels": test_cem_state["labels"].bool(),
        "lengthscale": test_cem_state["lengthscale"],
        "signal_var": test_cem_state["signal_var"],
        "noise_var": test_cem_state["noise_var"].clamp_min(_local_noise_var_floor(args, spec)),
        "mean_const": test_cem_state["mean_const"],
        "gate": test_cem_state["gate"],
    }
    t_test = time.perf_counter()
    test_state = run_uncertain_w_self_cem(
        X_anchor,
        X_neighbors,
        y_neighbors_std,
        test_init_state["labels"].bool(),
        mode=spec.mode,
        init_lengthscale=test_init_state["lengthscale"],
        init_signal_var=test_init_state["signal_var"],
        init_noise_var=test_init_state["noise_var"],
        gate_init=test_init_state["gate"],
        config=SelfCEMConfig(
            cem_updates=int(spec.cem_updates),
            hyperopt=FixedLabelHyperoptConfig(
                steps=int(args.refresh_hyper_steps),
                lr=float(args.hyper_lr),
                min_noise_var=float(_refresh_noise_var_floor(args, spec)),
                max_noise_var=float(args.max_noise_var),
                min_signal_var=float(args.min_signal_var),
                max_signal_var=float(args.max_signal_var),
                min_lengthscale=float(args.min_lengthscale),
                max_lengthscale=float(args.max_lengthscale),
            ),
            gate=FixedLabelGateConfig(
                steps=int(args.refresh_gate_steps),
                lr=float(args.gate_lr),
                l2=float(_refresh_gate_l2(args)),
                max_norm=float(_refresh_gate_max_norm(args)),
                gh_points=int(args.gh_points),
            ),
            outlier_mode="background_normal",
            outlier_sigma_mult=float(args.outlier_sigma_mult),
            background_scale_mult=float(args.background_scale_mult),
            min_inliers=int(args.min_inliers),
            max_inlier_frac=float(args.max_inlier_frac),
            refit_after_update=True,
        ),
        **pred_w_moments_test,
    )
    test_self_cem_sec = time.perf_counter() - t_test
    train_sec += float(test_self_cem_sec)
    if test_retrieval_diag:
        test_retrieval_diag["wknn_selected_inlier_frac"] = float(
            test_state["labels"].bool().to(dtype=torch.float64).mean().cpu().item()
        )

    prediction_diag: dict[str, Any]
    if int(spec.ensemble_samples) > 0:
        pack, prediction_diag, prediction_state, _ensemble_out = _predict_deterministic_w_self_cem_ensemble(
            args=args,
            spec=spec,
            seed=seed,
            X_anchor=X_anchor,
            X_neighbors=X_neighbors,
            y_neighbors_std=y_neighbors_std,
            test_init_state=test_init_state,
            w_moments=pred_w_moments_test,
            y_test=y_test,
            y_mean=y_mean,
            y_std=y_std,
        )
        pred_sec = float(prediction_diag["ensemble_total_sec"])
        init_labels_for_diag = _repeat_first_dim(test_init_state["labels"].bool(), int(spec.ensemble_samples))
        final_labels_for_diag = prediction_state["labels"].bool()
    else:
        t_pred = time.perf_counter()
        out = uncertain_w_cem_predict_from_labels(
            X_anchor,
            X_neighbors,
            y_neighbors_std,
            test_state["labels"],
            mode=spec.mode,
            lengthscale=test_state["lengthscale"],
            signal_var=test_state["signal_var"],
            noise_var=test_state["noise_var"],
            mean_const=test_state["mean_const"],
            correction_weight=float(spec.kcov),
            min_inliers=int(args.min_inliers),
            **pred_w_moments_test,
        )
        pred_sec = time.perf_counter() - t_pred
        mu = out.mu.detach().cpu().numpy().reshape(-1) * float(y_std) + float(y_mean)
        sigma = out.sigma.detach().cpu().numpy().reshape(-1) * abs(float(y_std))
        pack = _metrics_pack(y_test, mu, sigma, pred_sec)
        prediction_state = test_state
        init_labels_for_diag = test_init_state["labels"].bool()
        final_labels_for_diag = test_state["labels"].bool()
        prediction_diag = {
            "mean_kernel_vector_correction": float(out.diagnostics["mean_kernel_vector_correction"]),
            "fallback_count": int(out.diagnostics["fallback_count"]),
        }
    if test_retrieval_diag and int(spec.ensemble_samples) > 0:
        test_retrieval_diag["wknn_selected_inlier_frac"] = float(
            final_labels_for_diag.to(dtype=torch.float64).mean().cpu().item()
        )
    row = {key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)}
    row.update(_label_diag(init_labels_for_diag, final_labels_for_diag, X_neighbors.shape[1]))
    train_label_diag = _label_diag(init_labels.bool(), final_state["labels"].bool(), X_qr_neighbors.shape[1])
    row.update({f"qr_train_{k}": v for k, v in train_label_diag.items()})
    row.update(_neighbor_coverage_diag(qr_neighbor_idx, X_train_t.shape[0], "qr_train_neighbor"))
    row.update(_neighbor_coverage_diag(test_neighbor_idx, X_train_t.shape[0], "test_neighbor"))
    if qr_pool_idx is not None:
        row.update(_neighbor_coverage_diag(qr_pool_idx, X_train_t.shape[0], "qr_train_pool"))
    if test_pool_idx is not None:
        row.update(_neighbor_coverage_diag(test_pool_idx, X_train_t.shape[0], "test_pool"))
    row.update(
        {
            "train_sec": float(train_sec),
            "pred_sec": float(pred_sec),
            "test_self_cem_sec": float(test_self_cem_sec),
            "qr_train_anchor_count_actual": int(X_qr_anchor.shape[0]),
            "qr_train_n_neighbors_actual": int(X_qr_neighbors.shape[1]),
            "qr_train_init_sec": float(qr_cem_diag.get("init_sec", 0.0)),
            "gate_norm_mean": float(torch.linalg.norm(prediction_state["gate"].detach(), dim=1).mean().cpu().item()),
            "mean_lengthscale": float(prediction_state["lengthscale"].detach().mean().cpu().item()),
            "mean_signal_var": float(prediction_state["signal_var"].detach().mean().cpu().item()),
            "mean_noise_var": float(prediction_state["noise_var"].detach().mean().cpu().item()),
            "mean_kernel_vector_correction": float(prediction_diag.get("mean_kernel_vector_correction", 0.0)),
            "fallback_count": int(prediction_diag.get("fallback_count", 0)),
        }
    )
    row.update({k: sanitize_for_json(v) for k, v in prediction_diag.items()})
    qr_diag: dict[str, Any] = {
        "qr_train_sec": float(qr_train_total),
        "refresh_train_sec": float(refresh_train_total),
        "qr_w_lengthscale": float(qr_w_lengthscale),
        "qr_control": str(spec.qr_control),
        "qr_train_anchor_strategy": str(args.qr_train_anchor_strategy),
    }
    if retrieval_history:
        last_retrieval = retrieval_history[-1]
        qr_diag.update({f"qr_train_{k}_final": v for k, v in last_retrieval.items() if k != "outer"})
        qr_diag["qr_train_wknn_refresh_count"] = float(max(0, len(retrieval_history) - 1))
    if test_retrieval_diag:
        qr_diag.update({f"test_{k}": v for k, v in test_retrieval_diag.items()})
    if qr is not None:
        qr_diag.update(
            {
                "qr_best_loss": float(qr.diagnostics["best_loss"]),
                "qr_kl_per_anchor": float(qr.diagnostics["kl_per_anchor"]),
                "qr_R_std_median": float(qr.diagnostics["R_std_median"]),
            }
        )
        for key, value in qr.diagnostics.items():
            if key.startswith(("grad_", "pairwise_", "rank_", "anchor_pred_", "residual_group_", "basis_scale_", "ortho_", "svd_")):
                qr_diag[f"qr_{key}"] = value
    if refresh_history:
        changes = [r["label_change_rate"] for r in refresh_history]
        qr_diag.update(
            {
                "refresh_count": float(len(refresh_history)),
                "refresh_label_change_mean": float(np.mean(changes)),
                "refresh_label_change_final": float(changes[-1]),
                "refresh_label_change_max": float(np.max(changes)),
                "refresh_inlier_rate_final": float(refresh_history[-1]["inlier_rate"]),
                "refresh_noise_var_floor_active": float(refresh_history[-1]["refresh_noise_var_floor"]),
                "refresh_damping_active": float(refresh_history[-1]["refresh_damping"]),
                "refresh_gate_norm_final": float(refresh_history[-1]["gate_norm_mean"]),
            }
        )
    row.update({k: sanitize_for_json(v) for k, v in qr_diag.items()})
    row["history_json"] = json.dumps(sanitize_for_json(history), sort_keys=True)
    row["qr_history_json"] = json.dumps(sanitize_for_json(qr_history), sort_keys=True)
    row["refresh_history_json"] = json.dumps(sanitize_for_json(refresh_history), sort_keys=True)
    row["retrieval_history_json"] = json.dumps(sanitize_for_json(retrieval_history), sort_keys=True)
    row["test_history_json"] = json.dumps(sanitize_for_json(prediction_state.get("history", [])), sort_keys=True)
    return row, {"history": history, "qr_history": qr_history, "refresh_history": refresh_history, "retrieval_history": retrieval_history}


def _run_original_row(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    setting: str,
    seed: int,
    cem_state: dict[str, torch.Tensor],
    cem_diag: dict[str, Any],
    y_test: np.ndarray,
    y_mean: float,
    y_std: float,
) -> dict[str, Any]:
    mu = cem_state["mu"].detach().cpu().numpy().reshape(-1) * float(y_std) + float(y_mean)
    sigma = cem_state["sigma"].detach().cpu().numpy().reshape(-1) * abs(float(y_std))
    pack = _metrics_pack(y_test, mu, sigma, float(cem_diag["init_sec"]))
    fake = VariantSpec("original_meanw_jumpgp_cem", "anchor", "original", "cem", 0, 0.0)
    row = _base_row(args, cfg, setting, seed, fake)
    row["variant"] = "original_meanw_jumpgp_cem"
    row["status"] = "ok"
    row.update({key: float(pack[i]) for i, key in enumerate(RESULT_ROW_KEYS)})
    labels = cem_state["labels"].bool()
    row.update(_label_diag(labels, labels, labels.shape[1]))
    row.update({k: sanitize_for_json(v) for k, v in cem_diag.items()})
    row["train_sec"] = float(cem_diag["init_sec"])
    row["pred_sec"] = 0.0
    return row


def _run_one(args: argparse.Namespace, setting: str, seed: int, variant: str, device: torch.device) -> dict[str, Any]:
    cfg = dict(SETTING_PRESETS[setting])
    spec = (
        VariantSpec("original_meanw_jumpgp_cem", "anchor", "original", "cem", 0, 0.0)
        if variant == "__original__"
        else _parse_variant(variant)
    )
    _set_seed(seed)
    row = _base_row(args, cfg, setting, seed, spec)
    try:
        if _is_wknn_variant(spec) and int(args.qr_train_anchor_count) <= 0:
            raise ValueError("W-dependent retrieval variants require --qr_train_anchor_count > 0 for supervised train-anchor qR.")
        data = _generate_paper_data(cfg, seed, device)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(data["X_train"]).astype(np.float64)
        X_test = scaler.transform(data["X_test"]).astype(np.float64)
        y_train = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
        y_test = np.asarray(data["y_test"], dtype=np.float64).reshape(-1)
        if args.max_test is not None:
            X_test = X_test[: int(args.max_test)]
            y_test = y_test[: int(args.max_test)]
        y_mean = float(y_train.mean())
        y_std = float(y_train.std() if y_train.std() > 1e-8 else 1.0)
        y_train_std = ((y_train - y_mean) / y_std).astype(np.float64)

        Q = int(cfg["Q"])
        W0_np = _projection_W0(X_train, y_train_std, Q, str(args.projection_init), seed=seed + 513)
        W0 = torch.as_tensor(W0_np, device=device, dtype=torch.float64)
        V_basis = None
        if int(spec.lowrank_rank) > 0:
            V_np = _residual_basis_V(
                X_train,
                y_train_std,
                W0_np,
                int(spec.lowrank_rank),
                mode=str(args.lowrank_basis),
                seed=seed + 104729,
            )
            V_basis = torch.as_tensor(V_np, device=device, dtype=torch.float64)
        X_train_t = torch.as_tensor(X_train, device=device, dtype=torch.float64)
        X_test_t = torch.as_tensor(X_test, device=device, dtype=torch.float64)
        y_train_std_t = torch.as_tensor(y_train_std, device=device, dtype=torch.float64)
        n_neighbors = int(args.n_neighbors or cfg["n"])
        X_neighbors, y_neighbors_std, test_neighbor_idx = _neighbor_stack(
            X_test_t,
            X_train_t,
            y_train_std_t,
            n_neighbors=n_neighbors,
        )
        w_cov_lengthscale = float(args.w_cov_lengthscale) if args.w_cov_lengthscale is not None else _median_distance(X_train_t)
        row["w_cov_lengthscale"] = float(w_cov_lengthscale)
        inducing_idx = _kmeans_medoids(X_train, int(args.m_inducing), seed + 7919)
        X_inducing = X_train_t[torch.as_tensor(inducing_idx, device=device, dtype=torch.long)]
        qr_w_lengthscale = float(args.qr_w_lengthscale) if args.qr_w_lengthscale is not None else _median_distance(X_inducing)
        w_moments = _build_initial_w_moments(
            args=args,
            spec=spec,
            W0=W0,
            V_basis=V_basis,
            X_anchor=X_test_t,
            X_neighbors=X_neighbors,
            X_train_t=X_train_t,
            w_cov_lengthscale=w_cov_lengthscale,
        )
        cem_state, cem_diag = _run_meanw_cem_init(X_test_t, X_neighbors, y_neighbors_std, W0)
        if bool(args.include_original) and variant == "__original__":
            return _run_original_row(
                args=args,
                cfg=cfg,
                setting=setting,
                seed=seed,
                cem_state=cem_state,
                cem_diag=cem_diag,
                y_test=y_test,
                y_mean=y_mean,
                y_std=y_std,
            )
        if int(args.qr_train_anchor_count) > 0 and spec.kind == "qr_alt":
            out_row, _extra = _run_train_anchor_qr_alt_variant(
                args,
                spec,
                seed=seed,
                X_train_np=X_train,
                y_train_std_np=y_train_std,
                X_train_t=X_train_t,
                y_train_std_t=y_train_std_t,
                X_anchor=X_test_t,
                X_neighbors=X_neighbors,
                y_neighbors_std=y_neighbors_std,
                test_neighbor_idx=test_neighbor_idx,
                y_test=y_test,
                y_mean=y_mean,
                y_std=y_std,
                W0=W0,
                V_basis=V_basis,
                test_w_moments=w_moments,
                test_cem_state=cem_state,
                X_inducing=X_inducing,
                qr_w_lengthscale=qr_w_lengthscale,
                w_cov_lengthscale=w_cov_lengthscale,
            )
        else:
            out_row, _extra = _run_variant(
                args,
                spec,
                X_anchor=X_test_t,
                X_neighbors=X_neighbors,
                y_neighbors_std=y_neighbors_std,
                y_test=y_test,
                y_mean=y_mean,
                y_std=y_std,
                W0=W0,
                V_basis=V_basis,
                w_moments=w_moments,
                cem_state=cem_state,
                X_inducing=X_inducing,
                qr_w_lengthscale=qr_w_lengthscale,
            )
        row["status"] = "ok"
        row.update(out_row)
        row.update({k: sanitize_for_json(v) for k, v in cem_diag.items()})
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED setting={setting} seed={seed} variant={variant}: {row['error']}", flush=True)
    return row


def _write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# Uncertain-W CEM L2/LH Prediction-Head Ablation

This experiment evaluates the uncertain-W CEM prediction head on paper-style
L2/LH synthetic data.  Base variants use a fixed mean projection from PLS/PCA
and controlled W uncertainty; `qr_mstep_*` and `qr_alt_*` variants train a
sparse-GP `q(R)` M-step through the moment-matched uncertain-W kernel.

Files:

- `metrics.csv`: one row per setting/seed/variant.
- `aggregate.csv`: mean/std over successful rows.
- `summary.json`: exact runner arguments and setting metadata.

```json
{json.dumps(summary, indent=2)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uncertain-W CEM prediction-head ablations on paper L2/LH data.")
    p.add_argument("--settings", default="l2_q2_train1000,lh_q5_train1000")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--num_exp", type=int, default=1)
    p.add_argument("--projection_init", choices=["pls", "pca", "pca_scaled", "pairwise"], default="pls")
    p.add_argument("--max_test", type=int, default=200)
    p.add_argument("--n_neighbors", type=int, default=None)
    p.add_argument("--m_inducing", type=int, default=40)
    p.add_argument(
        "--qr_train_anchor_count",
        type=int,
        default=0,
        help="If positive, train q(R) on this many train-set anchors and induce test-anchor W for prediction.",
    )
    p.add_argument(
        "--qr_train_anchor_strategy",
        choices=["kmeans", "random", "kmeans_random", "kmeans_yvar"],
        default="kmeans",
    )
    p.add_argument(
        "--qr_train_n_neighbors",
        type=int,
        default=None,
        help="Neighborhood size for train anchors used by the inductive q(R) M-step; defaults to --n_neighbors.",
    )
    p.add_argument("--w_var", type=float, default=0.02)
    p.add_argument("--w_cov_lengthscale", type=float, default=None)
    p.add_argument("--hyper_steps", type=int, default=30)
    p.add_argument("--hyper_lr", type=float, default=0.04)
    p.add_argument("--gate_steps", type=int, default=60)
    p.add_argument("--gate_lr", type=float, default=0.05)
    p.add_argument("--gate_l2", type=float, default=1e-3)
    p.add_argument("--gate_max_norm", type=float, default=8.0)
    p.add_argument("--gh_points", type=int, default=20)
    p.add_argument("--min_inliers", type=int, default=2)
    p.add_argument("--max_inlier_frac", type=float, default=0.95)
    p.add_argument("--outlier_sigma_mult", type=float, default=2.5)
    p.add_argument("--background_scale_mult", type=float, default=3.0)
    p.add_argument("--min_noise_var", type=float, default=1e-5)
    p.add_argument("--max_noise_var", type=float, default=10.0)
    p.add_argument("--min_signal_var", type=float, default=1e-4)
    p.add_argument("--max_signal_var", type=float, default=100.0)
    p.add_argument("--min_lengthscale", type=float, default=1e-3)
    p.add_argument("--max_lengthscale", type=float, default=20.0)
    p.add_argument("--response_seed_frac", type=float, default=0.25)
    p.add_argument("--response_inlier_frac", type=float, default=0.70)
    p.add_argument("--qr_steps", type=int, default=30)
    p.add_argument("--qr_lr", type=float, default=0.01)
    p.add_argument("--qr_beta_kl", type=float, default=0.05)
    p.add_argument("--qr_kl_warmup_steps", type=int, default=20)
    p.add_argument("--qr_composite_temperature", type=float, default=1.0)
    p.add_argument("--qr_init_log_std", type=float, default=-3.0)
    p.add_argument("--qr_train_log_std", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--qr_w_lengthscale", type=float, default=None)
    p.add_argument("--qr_w_signal_var", type=float, default=1.0)
    p.add_argument("--lowrank_basis", choices=["pca_residual", "pca", "random", "pls_pca", "overcomplete"], default="pca_residual")
    p.add_argument("--lowrank_qr_w_signal_var", type=float, default=0.1)
    p.add_argument("--lowrank_basis_scale_l1", type=float, default=0.0)
    p.add_argument("--qr_residual_group_weight", type=float, default=0.0)
    p.add_argument("--qr_residual_group_eps", type=float, default=1e-8)
    p.add_argument("--qr_include_residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qr_grad_clip_norm", type=float, default=10.0)
    p.add_argument("--qr_log_interval", type=int, default=10)
    p.add_argument("--qr_outer_cycles", type=int, default=4)
    p.add_argument("--qr_pairwise_weight", type=float, default=0.0)
    p.add_argument("--qr_pairwise_margin", type=float, default=1.0)
    p.add_argument("--qr_pairwise_same_weight", type=float, default=1.0)
    p.add_argument("--qr_pairwise_cross_weight", type=float, default=1.0)
    p.add_argument("--qr_pairwise_normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qr_ortho_weight", type=float, default=0.0)
    p.add_argument("--qr_ortho_target_scale", type=float, default=1.0)
    p.add_argument("--qr_ortho_normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qr_mean_parameterization", choices=["free", "shared_svd"], default="free")
    p.add_argument(
        "--qr_retrieval_pool",
        type=int,
        default=200,
        help="Raw-X candidate-pool size for _wknn variants; W top-k is selected inside this fixed pool.",
    )
    p.add_argument(
        "--qr_retrieval_composite_temperature",
        type=float,
        default=0.0,
        help="Selected-neighborhood CEM-bound weight for _wknn qR; default 0 avoids comparing W-selected marginal likelihoods.",
    )
    p.add_argument("--qr_anchor_pred_weight", type=float, default=1.0)
    p.add_argument("--qr_anchor_pred_var_floor", type=float, default=1e-6)
    p.add_argument("--qr_anchor_pred_exclude_self", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qr_anchor_self_tol", type=float, default=1e-10)
    p.add_argument("--qr_rank_weight", type=float, default=0.0)
    p.add_argument("--qr_rank_temperature", type=float, default=0.25)
    p.add_argument("--qr_rank_normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--refresh_hyper_steps", type=int, default=10)
    p.add_argument("--refresh_gate_steps", type=int, default=20)
    p.add_argument(
        "--refresh_noise_std_floor",
        type=float,
        default=None,
        help="Optional noise standard-deviation floor for alternating C-step refreshes on standardized y scale.",
    )
    p.add_argument(
        "--refresh_damping",
        type=float,
        default=1.0,
        help="Blend factor for refreshed local hyperparameters/gate; 1.0 fully accepts the refresh.",
    )
    p.add_argument(
        "--refresh_gate_l2",
        type=float,
        default=None,
        help="Optional gate L2 penalty for alternating C-step refreshes; defaults to --gate_l2.",
    )
    p.add_argument(
        "--refresh_gate_max_norm",
        type=float,
        default=None,
        help="Optional gate norm cap for alternating C-step refreshes; defaults to --gate_max_norm.",
    )
    p.add_argument("--include_original", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out_dir", default="experiments/synthetic/uncertain_w_cem_l2_lh_20260514")
    return p.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    settings = [s.strip() for s in str(args.settings).split(",") if s.strip()]
    variants = [s.strip() for s in str(args.variants).split(",") if s.strip()]
    unknown_settings = sorted(set(settings) - set(SETTING_PRESETS))
    if unknown_settings:
        raise ValueError(f"Unknown settings: {unknown_settings}")
    for variant in variants:
        _parse_variant(variant)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(out_dir / "metrics.csv") if args.resume else []
    done = {(r.get("setting"), int(r.get("seed", -1)), r.get("variant")) for r in rows if r.get("status") == "ok"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.num_exp)):
        for setting in settings:
            if bool(args.include_original):
                key = (setting, seed, "original_meanw_jumpgp_cem")
                if key not in done:
                    print(f"run setting={setting} seed={seed} variant=original_meanw_jumpgp_cem", flush=True)
                    rows.append(_run_one(args, setting, seed, "__original__", device))
                    _write_csv(out_dir / "metrics.csv", rows)
                    _write_csv(out_dir / "aggregate.csv", _aggregate(rows))
            for variant in variants:
                key = (setting, seed, variant)
                if key in done:
                    print(f"skip completed setting={setting} seed={seed} variant={variant}", flush=True)
                    continue
                print(f"run setting={setting} seed={seed} variant={variant}", flush=True)
                rows.append(_run_one(args, setting, seed, variant, device))
                _write_csv(out_dir / "metrics.csv", rows)
                _write_csv(out_dir / "aggregate.csv", _aggregate(rows))

    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "args": vars(args),
        "settings": {name: SETTING_PRESETS[name] for name in settings},
        "variants": variants,
        "n_rows": len(rows),
        "n_ok": int(sum(1 for r in rows if r.get("status") == "ok")),
        "n_failed": int(sum(1 for r in rows if r.get("status") != "ok")),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    _write_readme(out_dir, sanitize_for_json(summary))
    print(f"Saved results to {out_dir}", flush=True)
    return {"rows": rows, "aggregate": _aggregate(rows), "summary": summary}


if __name__ == "__main__":
    main()
