"""Global sparse-GP backends for the residual self-CEM additive model.

This module exposes a small, uniform API used by the
``djgp.projections.global_residual_selfcem`` driver to fit a global smooth
mean component ``g(x)`` on training data and produce both pointwise
``(mu_g, v_g)`` predictions and full posterior covariance blocks
``Sigma_g(X, X)`` for arbitrary query points.

Three backends are supported, all built on top of GPyTorch's
``ApproximateGP`` so they scale to ``N_train ~ 1000-10000`` and observed
dimensionality ``D ~ 30`` without the slow ``O(N^3)`` cost of a full GP:

1. ``sparse_svgp`` -- raw-X SVGP with an RBF/ARD kernel and learned
   inducing locations (default, recommended for LH/L2 where ``D=30``).
2. ``dkl`` -- deep kernel learning: small MLP feature map followed by an
   SVGP on the latent features.  Useful when the data has a low-rank
   structure that benefits from a learned nonlinear projection (matches
   the LH/RFF generator pattern with ``d=5 -> H=30``).
3. ``deepgp`` -- a 2-layer ``gpytorch.models.deep_gps.DeepGP`` with
   ``DeepApproximateMLL``.  Heavier and easier to overfit; offered as an
   ablation option requested by the user but not used by default.

The module also offers a K-fold OOB-style helper (``predict_oob_kfold``)
used by Variant C of the residual ablation, and supports heteroscedastic
training noise (``FixedNoiseGaussianLikelihood``) used by Variant D when
the global GP is refit with local-residual uncertainty.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

try:
    from gpytorch.mlls import DeepApproximateMLL
    from gpytorch.models.deep_gps import DeepGP, DeepGPLayer

    _DEEPGP_AVAILABLE = True
except Exception:  # pragma: no cover - depends on gpytorch version
    _DEEPGP_AVAILABLE = False


BACKENDS = ("sparse_svgp", "dkl", "deepgp")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalGPConfig:
    """Hyperparameters shared by every global GP backend."""

    backend: str = "sparse_svgp"
    num_inducing: int = 128
    epochs: int = 200
    batch_size: int = 1024
    lr: float = 0.01
    weight_decay: float = 0.0
    init_noise_var: float = 0.05
    standardize_y: bool = True
    seed: int = 0
    # DKL-only options.
    dkl_hidden: int = 64
    dkl_latent: int = 8
    # Deep-GP-only options.
    deepgp_hidden_dim: int = 4
    deepgp_num_inducing_inner: int = 64
    deepgp_num_samples: int = 8
    # Numerical safety.
    jitter: float = 1e-4
    min_noise_var: float = 1e-6
    verbose: bool = False


@dataclass(frozen=True)
class GlobalGPTrainResult:
    """Diagnostics from a single ``train`` call."""

    final_loss: float
    train_sec: float
    num_data: int
    backend: str
    history: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal SVGP modules
# ---------------------------------------------------------------------------


class _RawSVGP(ApproximateGP):
    """Standard SVGP with ARD-RBF kernel and learned inducing locations."""

    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=int(inducing_points.size(-1))),
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class _FeatureMap(nn.Module):
    """MLP feature extractor used by the DKL backend."""

    def __init__(self, d_in: int, d_hidden: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LatentSVGP(ApproximateGP):
    """SVGP on a learned latent feature space (used by DKL)."""

    def __init__(self, inducing_points_latent: torch.Tensor):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points_latent.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points_latent,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=int(inducing_points_latent.size(-1))),
        )

    def forward(self, h: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(h), self.covar_module(h))


if _DEEPGP_AVAILABLE:

    class _ToyDeepGPHiddenLayer(DeepGPLayer):
        def __init__(
            self,
            input_dims: int,
            output_dims: int | None,
            num_inducing: int = 64,
            mean_type: str = "constant",
        ):
            inducing_points = torch.randn(
                output_dims if output_dims is not None else 1,
                num_inducing,
                input_dims,
            ) if output_dims is not None else torch.randn(num_inducing, input_dims)
            batch_shape = (
                torch.Size([output_dims]) if output_dims is not None else torch.Size([])
            )
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing,
                batch_shape=batch_shape,
            )
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )
            super().__init__(variational_strategy, input_dims, output_dims)
            self.mean_module = (
                ConstantMean(batch_shape=batch_shape)
                if mean_type == "constant"
                else gpytorch.means.LinearMean(input_dims, batch_shape=batch_shape)
            )
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape,
            )

        def forward(self, x):  # type: ignore[override]
            return MultivariateNormal(self.mean_module(x), self.covar_module(x))


    class _TwoLayerDeepGP(DeepGP):
        def __init__(self, d_in: int, d_hidden: int, num_inducing_inner: int):
            hidden_layer = _ToyDeepGPHiddenLayer(
                input_dims=d_in,
                output_dims=d_hidden,
                num_inducing=num_inducing_inner,
                mean_type="linear",
            )
            last_layer = _ToyDeepGPHiddenLayer(
                input_dims=d_hidden,
                output_dims=None,
                num_inducing=num_inducing_inner,
                mean_type="constant",
            )
            super().__init__()
            self.hidden_layer = hidden_layer
            self.last_layer = last_layer
            self.likelihood = GaussianLikelihood()

        def forward(self, x):  # type: ignore[override]
            hidden = self.hidden_layer(x)
            output = self.last_layer(hidden)
            return output


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _to_2d(X: torch.Tensor | np.ndarray) -> torch.Tensor:
    out = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    if not torch.is_floating_point(out):
        out = out.float()
    return out


def _to_1d(y: torch.Tensor | np.ndarray) -> torch.Tensor:
    out = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
    if not torch.is_floating_point(out):
        out = out.float()
    return out.reshape(-1)


def _pick_inducing(X: torch.Tensor, M: int, seed: int) -> torch.Tensor:
    """Pick ``M`` inducing points by random subsample; works for any backend."""
    N = X.shape[0]
    M = max(1, min(int(M), int(N)))
    g = torch.Generator(device=X.device).manual_seed(int(seed) + 1)
    idx = torch.randperm(N, generator=g, device=X.device)[:M]
    return X[idx].clone().contiguous()


# ---------------------------------------------------------------------------
# Public model wrapper
# ---------------------------------------------------------------------------


class GlobalGPModel:
    """Uniform wrapper over the three global GP backends.

    All predict methods expose values on the **original y scale**, i.e. the
    stored ``y_mean`` / ``y_std`` are unrolled before returning predictive
    mean/variance/covariance.  ``include_noise`` controls whether the
    likelihood noise variance is included in the returned variances; the
    residual-self-CEM driver always asks for noise-free latent variances
    because the local CEM step then adds its own ``sigma_a^2`` floor.
    """

    def __init__(
        self,
        cfg: GlobalGPConfig,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        if cfg.backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend {cfg.backend!r}; choose from {BACKENDS}."
            )
        if cfg.backend == "deepgp" and not _DEEPGP_AVAILABLE:
            raise RuntimeError(
                "deepgp backend requires gpytorch.models.deep_gps; install a "
                "matching gpytorch version or fall back to sparse_svgp / dkl."
            )
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self._features: nn.Module | None = None
        self._svgp: ApproximateGP | None = None
        self._deepgp: "DeepGP | None" = None
        self._likelihood: GaussianLikelihood | None = None
        self._fixed_likelihood: FixedNoiseGaussianLikelihood | None = None
        self._trained: bool = False
        self._heteroscedastic: bool = False
        self._num_data: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        *,
        noise_var_per_point: torch.Tensor | np.ndarray | None = None,
    ) -> GlobalGPTrainResult:
        """Fit the global GP.  ``noise_var_per_point`` enables heteroscedastic noise.

        When ``noise_var_per_point`` is provided, ``GaussianLikelihood`` is
        replaced by a ``FixedNoiseGaussianLikelihood`` so that the
        per-training-point noise variance can encode the local residual
        uncertainty ``v_h_i`` used by Variant D.
        """
        _seed_all(self.cfg.seed)
        device = self.device
        dtype = self.dtype

        X = _to_2d(X_train).to(device=device, dtype=dtype)
        y = _to_1d(y_train).to(device=device, dtype=dtype)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_train/y_train length mismatch.")
        self._num_data = int(X.shape[0])

        if self.cfg.standardize_y:
            self.y_mean = float(y.mean().item())
            std = float(y.std().item())
            self.y_std = float(std if std > 1e-8 else 1.0)
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
        y_z = (y - self.y_mean) / self.y_std

        noise_per_point_z = None
        if noise_var_per_point is not None:
            nv = _to_1d(noise_var_per_point).to(device=device, dtype=dtype)
            if nv.shape[0] != y.shape[0]:
                raise ValueError("noise_var_per_point length mismatch with y_train.")
            noise_per_point_z = (nv / (self.y_std ** 2)).clamp_min(
                float(self.cfg.min_noise_var)
            )
            self._heteroscedastic = True
        else:
            self._heteroscedastic = False

        backend = self.cfg.backend
        t0 = time.perf_counter()
        if backend == "sparse_svgp":
            history = self._train_sparse_svgp(X, y_z, noise_per_point_z)
        elif backend == "dkl":
            history = self._train_dkl(X, y_z, noise_per_point_z)
        elif backend == "deepgp":
            history = self._train_deepgp(X, y_z, noise_per_point_z)
        else:
            raise ValueError(f"Unknown backend {backend!r}.")
        train_sec = float(time.perf_counter() - t0)
        self._trained = True
        return GlobalGPTrainResult(
            final_loss=float(history[-1]["loss"]) if history else float("nan"),
            train_sec=train_sec,
            num_data=int(self._num_data),
            backend=str(backend),
            history=history,
        )

    def _train_sparse_svgp(
        self,
        X: torch.Tensor,
        y_z: torch.Tensor,
        noise_per_point_z: torch.Tensor | None,
    ) -> list[dict[str, float]]:
        M = int(self.cfg.num_inducing)
        inducing = _pick_inducing(X, M, int(self.cfg.seed))
        svgp = _RawSVGP(inducing).to(device=self.device, dtype=self.dtype)
        self._svgp = svgp
        params: list[nn.Parameter] = list(svgp.parameters())
        if noise_per_point_z is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                noise=noise_per_point_z.clamp_min(float(self.cfg.min_noise_var)),
                learn_additional_noise=True,
            ).to(device=self.device, dtype=self.dtype)
            self._fixed_likelihood = likelihood
        else:
            likelihood = GaussianLikelihood().to(device=self.device, dtype=self.dtype)
            likelihood.noise = torch.tensor(
                float(self.cfg.init_noise_var), device=self.device, dtype=self.dtype
            )
            self._likelihood = likelihood
        params = params + list(likelihood.parameters())
        optimizer = torch.optim.Adam(
            params, lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay)
        )
        mll = VariationalELBO(likelihood, svgp, num_data=int(X.shape[0]))
        return self._run_minibatch_loop(svgp, likelihood, optimizer, mll, X, y_z, noise_per_point_z)

    def _train_dkl(
        self,
        X: torch.Tensor,
        y_z: torch.Tensor,
        noise_per_point_z: torch.Tensor | None,
    ) -> list[dict[str, float]]:
        features = _FeatureMap(
            d_in=int(X.shape[1]),
            d_hidden=int(self.cfg.dkl_hidden),
            d_latent=int(self.cfg.dkl_latent),
        ).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            n_sub = min(512, X.shape[0])
            z_init = features(X[:n_sub])
        idx = torch.randperm(z_init.size(0), device=self.device)[: int(self.cfg.num_inducing)]
        inducing = z_init[idx].clone()
        svgp = _LatentSVGP(inducing).to(device=self.device, dtype=self.dtype)
        self._features = features
        self._svgp = svgp
        params: list[nn.Parameter] = list(features.parameters()) + list(svgp.parameters())
        if noise_per_point_z is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                noise=noise_per_point_z.clamp_min(float(self.cfg.min_noise_var)),
                learn_additional_noise=True,
            ).to(device=self.device, dtype=self.dtype)
            self._fixed_likelihood = likelihood
        else:
            likelihood = GaussianLikelihood().to(device=self.device, dtype=self.dtype)
            likelihood.noise = torch.tensor(
                float(self.cfg.init_noise_var), device=self.device, dtype=self.dtype
            )
            self._likelihood = likelihood
        params = params + list(likelihood.parameters())
        optimizer = torch.optim.Adam(
            params, lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay)
        )
        mll = VariationalELBO(likelihood, svgp, num_data=int(X.shape[0]))
        return self._run_minibatch_loop(svgp, likelihood, optimizer, mll, X, y_z, noise_per_point_z)

    def _train_deepgp(
        self,
        X: torch.Tensor,
        y_z: torch.Tensor,
        noise_per_point_z: torch.Tensor | None,
    ) -> list[dict[str, float]]:
        if not _DEEPGP_AVAILABLE:
            raise RuntimeError("Deep GP backend requires gpytorch.models.deep_gps.")
        if noise_per_point_z is not None:
            raise NotImplementedError(
                "Heteroscedastic noise is not implemented for the deepgp backend; "
                "Variant D should use sparse_svgp or dkl as the global backend."
            )
        model = _TwoLayerDeepGP(
            d_in=int(X.shape[1]),
            d_hidden=int(self.cfg.deepgp_hidden_dim),
            num_inducing_inner=int(self.cfg.deepgp_num_inducing_inner),
        ).to(device=self.device, dtype=self.dtype)
        self._deepgp = model
        self._likelihood = model.likelihood
        optimizer = torch.optim.Adam(
            list(model.parameters()),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=int(X.shape[0])))
        history: list[dict[str, float]] = []
        bs = min(int(self.cfg.batch_size), int(X.shape[0]))
        for ep in range(int(self.cfg.epochs)):
            perm = torch.randperm(X.shape[0], device=X.device)
            running = 0.0
            n_batches = 0
            for i in range(0, X.shape[0], bs):
                idx = perm[i : i + bs]
                xb = X[idx]
                yb = y_z[idx]
                optimizer.zero_grad(set_to_none=True)
                with gpytorch.settings.num_likelihood_samples(int(self.cfg.deepgp_num_samples)):
                    out = model(xb)
                    loss = -mll(out, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.detach().item())
                n_batches += 1
            mean_loss = running / max(1, n_batches)
            if self.cfg.verbose and (ep % 20 == 0 or ep == int(self.cfg.epochs) - 1):
                print(f"  [deepgp] epoch={ep:03d} loss={mean_loss:.4f}", flush=True)
            history.append({"epoch": float(ep), "loss": float(mean_loss)})
        return history

    def _run_minibatch_loop(
        self,
        svgp: ApproximateGP,
        likelihood: GaussianLikelihood | FixedNoiseGaussianLikelihood,
        optimizer: torch.optim.Optimizer,
        mll: VariationalELBO,
        X: torch.Tensor,
        y_z: torch.Tensor,
        noise_per_point_z: torch.Tensor | None,
    ) -> list[dict[str, float]]:
        bs = min(int(self.cfg.batch_size), int(X.shape[0]))
        history: list[dict[str, float]] = []
        backend = self.cfg.backend
        for ep in range(int(self.cfg.epochs)):
            perm = torch.randperm(X.shape[0], device=X.device)
            running = 0.0
            n_batches = 0
            for i in range(0, X.shape[0], bs):
                idx = perm[i : i + bs]
                xb = X[idx]
                yb = y_z[idx]
                optimizer.zero_grad(set_to_none=True)
                if backend == "dkl":
                    hb = self._features(xb)
                    with gpytorch.settings.cholesky_jitter(float(self.cfg.jitter)):
                        out = svgp(hb)
                        if noise_per_point_z is None:
                            loss = -mll(out, yb)
                        else:
                            loss = -mll(out, yb, noise=noise_per_point_z[idx].clamp_min(float(self.cfg.min_noise_var)))
                else:
                    with gpytorch.settings.cholesky_jitter(float(self.cfg.jitter)):
                        out = svgp(xb)
                        if noise_per_point_z is None:
                            loss = -mll(out, yb)
                        else:
                            loss = -mll(out, yb, noise=noise_per_point_z[idx].clamp_min(float(self.cfg.min_noise_var)))
                loss.backward()
                optimizer.step()
                running += float(loss.detach().item())
                n_batches += 1
            mean_loss = running / max(1, n_batches)
            if self.cfg.verbose and (ep % 20 == 0 or ep == int(self.cfg.epochs) - 1):
                print(f"  [{backend}] epoch={ep:03d} loss={mean_loss:.4f}", flush=True)
            history.append({"epoch": float(ep), "loss": float(mean_loss)})
        return history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _eval_mode(self) -> None:
        if self._features is not None:
            self._features.eval()
        if self._svgp is not None:
            self._svgp.eval()
        if self._deepgp is not None:
            self._deepgp.eval()
        if self._likelihood is not None:
            self._likelihood.eval()
        if self._fixed_likelihood is not None:
            self._fixed_likelihood.eval()

    def _latent_features(self, X: torch.Tensor) -> torch.Tensor:
        if self.cfg.backend == "dkl":
            assert self._features is not None
            return self._features(X)
        return X

    def predict(
        self,
        X: torch.Tensor | np.ndarray,
        *,
        include_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, var)`` on the original y scale for a 2-D ``X``."""
        if not self._trained:
            raise RuntimeError("GlobalGPModel.predict called before train().")
        self._eval_mode()
        device = self.device
        dtype = self.dtype
        X_t = _to_2d(X).to(device=device, dtype=dtype)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if self.cfg.backend == "deepgp":
                assert self._deepgp is not None
                with gpytorch.settings.num_likelihood_samples(int(self.cfg.deepgp_num_samples)):
                    posterior = self._deepgp(X_t)
                # DeepGP outputs a sample-batched distribution; average mean/var.
                mu_z = posterior.mean.mean(dim=0)
                var_z = posterior.variance.mean(dim=0)
            else:
                assert self._svgp is not None
                h_t = self._latent_features(X_t)
                with gpytorch.settings.cholesky_jitter(float(self.cfg.jitter)):
                    latent_dist = self._svgp(h_t)
                mu_z = latent_dist.mean
                var_z = latent_dist.variance
            if include_noise:
                if self._fixed_likelihood is not None:
                    extra = self._fixed_likelihood.second_noise.detach().reshape(())
                    var_z = var_z + extra
                elif self._likelihood is not None:
                    var_z = var_z + self._likelihood.noise.detach().reshape(())
        mu = mu_z * self.y_std + self.y_mean
        var = (var_z * (self.y_std ** 2)).clamp_min(1e-12)
        return mu, var

    def predict_cov(
        self,
        X: torch.Tensor | np.ndarray,
        *,
        include_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, Sigma)`` posterior block on the original y scale for a 2-D ``X``."""
        if not self._trained:
            raise RuntimeError("GlobalGPModel.predict_cov called before train().")
        self._eval_mode()
        device = self.device
        dtype = self.dtype
        X_t = _to_2d(X).to(device=device, dtype=dtype)
        n = int(X_t.shape[0])
        with torch.no_grad():
            if self.cfg.backend == "deepgp":
                assert self._deepgp is not None
                with gpytorch.settings.num_likelihood_samples(int(self.cfg.deepgp_num_samples)):
                    posterior = self._deepgp(X_t)
                mu_z = posterior.mean.mean(dim=0)
                # Approximate covariance by averaging samples across the deep dimension.
                base_cov = posterior.lazy_covariance_matrix.to_dense().mean(dim=0)
                cov_z = 0.5 * (base_cov + base_cov.transpose(-1, -2))
            else:
                assert self._svgp is not None
                h_t = self._latent_features(X_t)
                with gpytorch.settings.cholesky_jitter(float(self.cfg.jitter)):
                    latent_dist = self._svgp(h_t)
                mu_z = latent_dist.mean
                cov_z = latent_dist.lazy_covariance_matrix.to_dense()
                cov_z = 0.5 * (cov_z + cov_z.transpose(-1, -2))
            if include_noise:
                noise = None
                if self._fixed_likelihood is not None:
                    noise = self._fixed_likelihood.second_noise.detach().reshape(())
                elif self._likelihood is not None:
                    noise = self._likelihood.noise.detach().reshape(())
                if noise is not None:
                    cov_z = cov_z + noise * torch.eye(n, device=cov_z.device, dtype=cov_z.dtype)
        mu = mu_z * self.y_std + self.y_mean
        cov = cov_z * (self.y_std ** 2)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        return mu, cov.clamp_min(0.0) if False else cov  # cov can have small negatives; caller handles jitter

    # ------------------------------------------------------------------
    # Batched neighborhood-block helpers
    # ------------------------------------------------------------------

    def predict_neighborhood(
        self,
        X_neighbors: torch.Tensor,
        *,
        include_block: bool = False,
        include_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        """For a ``[T, n, D]`` neighborhood stack, return per-anchor blocks.

        Returns a dict with keys:
        - ``mu``: ``[T, n]`` posterior means.
        - ``var``: ``[T, n]`` pointwise marginal variances.
        - ``block``: ``[T, n, n]`` posterior covariance blocks (only if
          ``include_block=True``; otherwise None).

        Implementation iterates over anchors so it works for any backend
        size; for ``T <= ~200`` this is cheap.
        """
        if not self._trained:
            raise RuntimeError("GlobalGPModel.predict_neighborhood called before train().")
        T, n, _D = X_neighbors.shape
        device = X_neighbors.device
        dtype = X_neighbors.dtype
        mu_out = torch.empty(T, n, device=device, dtype=dtype)
        var_out = torch.empty(T, n, device=device, dtype=dtype)
        block_out: torch.Tensor | None = None
        if include_block:
            block_out = torch.zeros(T, n, n, device=device, dtype=dtype)
        for t in range(T):
            X_t = X_neighbors[t]
            if include_block:
                mu_t, cov_t = self.predict_cov(X_t, include_noise=include_noise)
                mu_out[t] = mu_t.to(device=device, dtype=dtype)
                var_out[t] = torch.diagonal(cov_t).clamp_min(1e-12).to(device=device, dtype=dtype)
                assert block_out is not None
                block_out[t] = cov_t.to(device=device, dtype=dtype)
            else:
                mu_t, var_t = self.predict(X_t, include_noise=include_noise)
                mu_out[t] = mu_t.to(device=device, dtype=dtype)
                var_out[t] = var_t.to(device=device, dtype=dtype)
        return {"mu": mu_out, "var": var_out, "block": block_out}


# ---------------------------------------------------------------------------
# OOB K-fold prediction (Variant C)
# ---------------------------------------------------------------------------


def predict_oob_kfold(
    cfg: GlobalGPConfig,
    X_train: torch.Tensor | np.ndarray,
    y_train: torch.Tensor | np.ndarray,
    *,
    k_folds: int = 5,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, list[GlobalGPModel]]:
    """Train ``k_folds`` global GPs on disjoint folds and predict OOB.

    Returns ``(mu_oob, var_oob, models)``: ``mu_oob`` and ``var_oob`` are
    length-``N`` arrays where index ``i`` is the posterior mean/variance from
    the GP that did **not** see point ``i`` during training.  ``models`` is
    the list of trained fold-models in order so callers can keep them for
    later test prediction (e.g. by averaging predictions).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = _to_2d(X_train).to(device=device, dtype=dtype)
    y = _to_1d(y_train).to(device=device, dtype=dtype)
    N = int(X.shape[0])
    if N == 0:
        raise ValueError("Empty training set for OOB k-fold.")
    k = max(2, min(int(k_folds), N))
    rng = np.random.default_rng(int(cfg.seed) + 9176)
    perm = rng.permutation(N)
    folds: list[np.ndarray] = np.array_split(perm, k)  # type: ignore[assignment]
    mu_oob = torch.empty(N, device=device, dtype=dtype)
    var_oob = torch.empty(N, device=device, dtype=dtype)
    models: list[GlobalGPModel] = []
    for f, val_idx in enumerate(folds):
        val_idx_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)
        train_mask = torch.ones(N, device=device, dtype=torch.bool)
        train_mask[val_idx_t] = False
        X_tr = X[train_mask]
        y_tr = y[train_mask]
        fold_cfg = GlobalGPConfig(**{**cfg.__dict__, "seed": int(cfg.seed) + 1000 * (f + 1)})
        m = GlobalGPModel(fold_cfg, device=device, dtype=dtype)
        m.train(X_tr, y_tr)
        mu_v, var_v = m.predict(X[val_idx_t], include_noise=False)
        mu_oob[val_idx_t] = mu_v.to(device=device, dtype=dtype)
        var_oob[val_idx_t] = var_v.to(device=device, dtype=dtype)
        models.append(m)
    return mu_oob, var_oob, models


def predict_oob_ensemble(
    models: Sequence[GlobalGPModel],
    X_query: torch.Tensor | np.ndarray,
    *,
    include_noise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average predictions over a set of fold-models for held-out queries.

    Uses the standard mixture-of-Gaussians moment matching:
    ``mu = mean_f mu_f``, ``var = mean_f (var_f + mu_f^2) - mu^2``.
    """
    if not models:
        raise ValueError("predict_oob_ensemble requires at least one model.")
    mus: list[torch.Tensor] = []
    vars_: list[torch.Tensor] = []
    for m in models:
        mu_m, var_m = m.predict(X_query, include_noise=include_noise)
        mus.append(mu_m)
        vars_.append(var_m)
    mu_stack = torch.stack(mus, dim=0)
    var_stack = torch.stack(vars_, dim=0)
    mu_avg = mu_stack.mean(dim=0)
    var_total = (var_stack + mu_stack.pow(2)).mean(dim=0) - mu_avg.pow(2)
    return mu_avg, var_total.clamp_min(1e-12)


__all__ = [
    "BACKENDS",
    "GlobalGPConfig",
    "GlobalGPModel",
    "GlobalGPTrainResult",
    "predict_oob_ensemble",
    "predict_oob_kfold",
]
