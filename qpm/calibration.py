"""
AnchoredCalibration — Drift-Aware Updates with Zero Rank-Order Churn
=====================================================================
Implements Section 3.3 of "Quantized Prediction Models" (Ortiz-Gómez, 2025).

Two mechanisms for stable model updates:
1. Residual Stacking  — fine-tune the latent model F on new data.
2. Anchored Calibration — update only bin representatives {q_k} while
   keeping bin edges fixed, guaranteeing zero rank-order churn.

Both mechanisms are model-agnostic: ``residual_stack`` accepts any base
model that exposes a ``.predict(X)`` method.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class AnchoredCalibration:
    """
    Drift-aware update with guaranteed zero rank-order churn (Appendix A).

    When bin edges are held fixed, every sample retains its bin assignment,
    so rank ordering is preserved exactly.  Only the bin-level PD centers
    {q_k} are updated to reflect new empirical default rates.

    Proposition 1 (Zero Rank-Order Churn):
        If F and bin edges are fixed, then for any x_i, x_j:
            F(x_i) < F(x_j)  =>  Q_new(F(x_i)) <= Q_new(F(x_j)).

    Parameters
    ----------
    alpha : float in [0, 1], default 0.5
        Anchoring weight.
        alpha=1 recovers the original model (no update).
        alpha=0 fully overwrites bin centers with new empirical means.
    monotone_smooth : bool, default True
        If True, project updated representatives onto a monotone sequence
        via isotonic regression before returning.
    pd_cap : float, default 0.80
        Maximum PD value for any bin representative after update.
    prior_weight : float, default 30
        Pseudo-count for Laplace smoothing of empirical bin means on new data.
        Helps stabilise bins with few new observations.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        monotone_smooth: bool = True,
        pd_cap: float = 0.80,
        prior_weight: float = 30.0,
    ):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = alpha
        self.monotone_smooth = monotone_smooth
        self.pd_cap = pd_cap
        self.prior_weight = prior_weight

    def update(
        self,
        qpm,
        F_new: np.ndarray,
        y_new: np.ndarray,
        alpha: float | None = None,
    ) -> np.ndarray:
        """
        Apply Anchored Calibration: update bin representatives in-place.

        The update rule (Eq. in Section 3.3.2):
            mu_k_new = alpha * mu_k_old + (1-alpha) * mu_k_obs

        where mu_k_obs = E[y | F(x) in bin k] on new data.

        Parameters
        ----------
        qpm : QPMQuantizer
            A fitted quantizer.  Its ``bin_edges_`` are NOT modified.
            Its ``bin_reps_`` are updated in-place.
        F_new : array-like of shape (n,)
            Latent scores on new / drift data (from the same base model).
        y_new : array-like of shape (n,)
            Binary labels on new data.
        alpha : float or None
            Override instance-level alpha for this call.

        Returns
        -------
        new_reps : ndarray of shape (K,)
            Updated bin representatives (also stored in qpm.bin_reps_).
        """
        alpha_use = self.alpha if alpha is None else float(alpha)
        F_new = np.asarray(F_new, dtype=float)
        y_new = np.asarray(y_new, dtype=float)

        mu_old = qpm.bin_reps_.copy()
        prior = float(y_new.mean())   # use new-data prevalence as smoothing prior

        K = qpm.n_bins_
        mu_obs = np.zeros(K)
        bin_idx = qpm.get_bin_index(F_new)

        for k in range(K):
            mask = bin_idx == k
            n_k = int(mask.sum())
            if n_k > 0:
                emp_mean = float(y_new[mask].mean())
                # Laplace smoothing
                mu_obs[k] = (n_k * emp_mean + self.prior_weight * prior) / (
                    n_k + self.prior_weight
                )
            else:
                mu_obs[k] = mu_old[k]   # no new data in this bin: keep old value

        # Anchored update
        mu_new = alpha_use * mu_old + (1.0 - alpha_use) * mu_obs
        mu_new = np.minimum(mu_new, self.pd_cap)

        if self.monotone_smooth:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            mu_new = iso.fit_transform(np.arange(K, dtype=float), mu_new)
            mu_new = np.minimum(mu_new, self.pd_cap)

        # Update in-place (bin edges untouched → zero churn)
        qpm.bin_reps_ = mu_new
        return mu_new


def residual_stack(
    base_model,
    X_new: np.ndarray,
    y_new: np.ndarray,
    eta: float = 0.1,
    n_estimators: int = 50,
    max_depth: int = 3,
    random_state: int = 0,
) -> tuple[np.ndarray, object]:
    """
    Fine-tune a latent model via residual stacking (Section 3.3.1).

    Instead of retraining F from scratch, fits a shallow XGBoost booster h
    on the residuals r = y - F(X) and returns the updated predictions:
        F_new(x) = F(x) + eta * h(x)

    The base model is NOT modified.

    Parameters
    ----------
    base_model : object with .predict(X) method
        Any fitted regressor / scorer: XGBoost, LightGBM, sklearn GBM,
        a neural-net wrapper, etc.
    X_new : array-like of shape (n, d)
        Feature matrix for new / drift data.
    y_new : array-like of shape (n,)
        Binary labels for new data.
    eta : float, default 0.1
        Learning rate for the residual booster.
    n_estimators : int, default 50
        Number of boosting rounds for the residual learner.
    max_depth : int, default 3
        Tree depth for the residual learner.
    random_state : int, default 0

    Returns
    -------
    F_updated : ndarray of shape (n,)
        Updated latent scores for X_new.
    residual_booster : fitted XGBRegressor
        The shallow residual booster h(x).  Can be used to update scores
        on other datasets via ``eta * residual_booster.predict(X)``.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for residual_stack. "
            "Install it with: pip install xgboost"
        ) from exc

    X_new = np.asarray(X_new, dtype=float)
    y_new = np.asarray(y_new, dtype=float)

    F_base = np.asarray(base_model.predict(X_new), dtype=float)
    residuals = y_new - F_base

    h = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=eta,
        max_depth=max_depth,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
    )
    h.fit(X_new, residuals)

    F_updated = F_base + eta * h.predict(X_new)
    return F_updated, h
