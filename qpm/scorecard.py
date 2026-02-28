"""
WoE Logistic Scorecard Baseline
================================
Implements the traditional credit-scoring baseline described in Section 4.1
of "Quantized Prediction Models" (Ortiz-Gómez, 2025).

Pipeline:
1. Coarse-class each numeric feature into quantile-based bins.
2. Compute Weight-of-Evidence (WoE) per bin.
3. Fit a logistic regression on WoE-encoded features.
4. Map log-odds to 10 equal-frequency bands to form the score ladder.

This scorecard represents the "white box" side of credit modelling and
serves as the lower-accuracy / higher-interpretability baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class WoEScorecard:
    """
    Weight-of-Evidence logistic regression scorecard.

    Parameters
    ----------
    n_quantile_bins : int, default 10
        Number of quantile bins per numeric feature for WoE encoding.
    n_score_bands : int, default 10
        Number of equal-frequency score bands for the output ladder.
    regularization : float, default 1.0
        Inverse regularisation strength (C) for LogisticRegression.
    eps : float, default 0.5
        Laplace pseudo-count added to event / non-event counts per bin
        to avoid log(0).
    random_state : int, default 0
    """

    def __init__(
        self,
        n_quantile_bins: int = 10,
        n_score_bands: int = 10,
        regularization: float = 1.0,
        eps: float = 0.5,
        random_state: int = 0,
    ):
        self.n_quantile_bins = n_quantile_bins
        self.n_score_bands = n_score_bands
        self.regularization = regularization
        self.eps = eps
        self.random_state = random_state

        # Fitted attributes
        self._bin_edges: dict = {}   # feature -> bin edges array
        self._woe_maps: dict = {}    # feature -> array of WoE values per bin
        self._feature_names: list[str] = []
        self._lr: LogisticRegression | None = None
        self._score_band_edges: np.ndarray | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> "WoEScorecard":
        """
        Fit the WoE scorecard.

        Parameters
        ----------
        X : array-like of shape (n, d) or DataFrame
        y : array-like of shape (n,), binary {0, 1}

        Returns
        -------
        self
        """
        X_df = _to_df(X)
        y = np.asarray(y, dtype=float)
        self._feature_names = list(X_df.columns)

        total_events = y.sum() + self.eps * len(self._feature_names)
        total_non_events = (1 - y).sum() + self.eps * len(self._feature_names)

        # WoE-encode each feature
        X_woe = np.zeros_like(X_df.values, dtype=float)
        for j, col in enumerate(self._feature_names):
            vals = X_df[col].values.astype(float)
            edges, woe_arr = self._fit_woe_feature(vals, y, total_events, total_non_events)
            self._bin_edges[col] = edges
            self._woe_maps[col] = woe_arr
            X_woe[:, j] = self._apply_woe(vals, edges, woe_arr)

        # Logistic regression
        self._lr = LogisticRegression(
            C=self.regularization,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        self._lr.fit(X_woe, y.astype(int))

        # Score band edges (equal-frequency on training log-odds)
        log_odds_train = self._log_odds(X_woe)
        quantiles = np.linspace(0.0, 100.0, self.n_score_bands + 1)[1:-1]
        self._score_band_edges = np.percentile(log_odds_train, quantiles)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default (PD) for each sample.

        Returns
        -------
        pd_hat : ndarray of shape (n,)
        """
        self._check_fitted()
        X_woe = self._transform_woe(X)
        return self._lr.predict_proba(X_woe)[:, 1]

    def score_ladder(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
    ) -> pd.DataFrame:
        """
        Return the 10-band score ladder on a given dataset.

        Columns: Band, LogOdds_min, LogOdds_max, PD, #Acct, %Portf, #Def, DefR.
        """
        self._check_fitted()
        X_woe = self._transform_woe(X)
        log_odds = self._log_odds(X_woe)
        y = np.asarray(y, dtype=float)

        band_idx = np.searchsorted(self._score_band_edges, log_odds, side="right")
        n_bands = self.n_score_bands
        n_total = len(y)
        n_def_total = int(y.sum())

        rows = []
        for b in range(n_bands):
            mask = band_idx == b
            n_acct = int(mask.sum())
            n_def = int(y[mask].sum())
            def_r = n_def / n_acct if n_acct > 0 else 0.0
            lo_min = (
                self._score_band_edges[b - 1] if b > 0 else -np.inf
            )
            lo_max = (
                self._score_band_edges[b] if b < n_bands - 1 else np.inf
            )
            pd_hat = float(np.mean(self.predict_proba(X)[mask])) if n_acct > 0 else 0.0
            rows.append(
                {
                    "Band": b + 1,
                    "LogOdds_min": round(lo_min, 4) if not np.isinf(lo_min) else "-inf",
                    "LogOdds_max": round(lo_max, 4) if not np.isinf(lo_max) else "+inf",
                    "PD": round(pd_hat, 4),
                    "#Acct": n_acct,
                    "%Portf": round(n_acct / n_total, 4) if n_total else None,
                    "#Def": n_def,
                    "DefR": round(def_r, 4),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_woe_feature(
        self,
        vals: np.ndarray,
        y: np.ndarray,
        total_events: float,
        total_non_events: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute quantile bin edges and WoE values for one feature."""
        quantiles = np.linspace(0.0, 100.0, self.n_quantile_bins + 1)[1:-1]
        edges = np.unique(np.percentile(vals, quantiles))

        n_bins = len(edges) + 1
        woe_arr = np.zeros(n_bins)
        bin_idx = np.searchsorted(edges, vals, side="right")

        for k in range(n_bins):
            mask = bin_idx == k
            events_k = float(y[mask].sum()) + self.eps
            non_events_k = float((1 - y[mask]).sum()) + self.eps
            dist_events = events_k / total_events
            dist_non_events = non_events_k / total_non_events
            woe_arr[k] = np.log(dist_events / dist_non_events)

        return edges, woe_arr

    def _apply_woe(
        self,
        vals: np.ndarray,
        edges: np.ndarray,
        woe_arr: np.ndarray,
    ) -> np.ndarray:
        bin_idx = np.searchsorted(edges, vals, side="right")
        return woe_arr[bin_idx]

    def _transform_woe(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X_df = _to_df(X)
        X_woe = np.zeros((len(X_df), len(self._feature_names)), dtype=float)
        for j, col in enumerate(self._feature_names):
            vals = X_df[col].values.astype(float)
            X_woe[:, j] = self._apply_woe(
                vals, self._bin_edges[col], self._woe_maps[col]
            )
        return X_woe

    def _log_odds(self, X_woe: np.ndarray) -> np.ndarray:
        """Return log-odds (decision function) of the logistic regressor."""
        return self._lr.decision_function(X_woe)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("WoEScorecard is not fitted. Call fit() first.")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _to_df(X) -> pd.DataFrame:
    """Convert array-like or DataFrame to DataFrame with consistent column names."""
    if isinstance(X, pd.DataFrame):
        return X
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    return pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(X_arr.shape[1])])
