"""
Quantized Prediction Models (QPMs)
===================================
Output-Space Regularization for Stable and Interpretable Regression.

Paper: Ortiz-Gómez, C.R. (2025). Quantized Prediction Models:
       Output-Space Regularization for Stable and Interpretable Regression.
       DOI: 10.5281/zenodo.17785949

The quantizer is model-agnostic: pass any 1-D array of latent scores F(x)
(from XGBoost, LightGBM, sklearn GBM, neural nets, etc.) together with binary
labels y to fit a supervised output quantizer.
"""

from .quantizer import QPMQuantizer
from .calibration import AnchoredCalibration, residual_stack
from .metrics import auc_score, ks_score, mse_score, bootstrap_ci, delong_test, churn_metrics
from .scorecard import WoEScorecard

__all__ = [
    "QPMQuantizer",
    "AnchoredCalibration",
    "residual_stack",
    "auc_score",
    "ks_score",
    "mse_score",
    "bootstrap_ci",
    "delong_test",
    "churn_metrics",
    "WoEScorecard",
]

__version__ = "0.1.0"
