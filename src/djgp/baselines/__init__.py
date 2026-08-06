"""Non-DJGP baselines: NGBoost and global sparse GPs."""

from djgp.baselines.global_sparse_gp import (
    BACKENDS as GLOBAL_GP_BACKENDS,
    GlobalGPConfig,
    GlobalGPModel,
    GlobalGPTrainResult,
    predict_oob_ensemble,
    predict_oob_kfold,
)
from djgp.baselines.ngboost_regression import run_ngboost_regression

__all__ = [
    "GLOBAL_GP_BACKENDS",
    "GlobalGPConfig",
    "GlobalGPModel",
    "GlobalGPTrainResult",
    "predict_oob_ensemble",
    "predict_oob_kfold",
    "run_ngboost_regression",
]
