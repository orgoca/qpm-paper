"""
QPMQuantizer — Supervised Output-Space Quantizer
=================================================
Implements Section 3 of "Quantized Prediction Models" (Ortiz-Gómez, 2025).

The quantizer is fully model-agnostic.  It takes latent scores F (any 1-D
numeric array from XGBoost, LightGBM, sklearn GBMs, neural nets, etc.) and
binary labels y, and learns a supervised, non-uniform output quantizer.

Usage
-----
>>> F_train = any_model.predict(X_train)   # 1-D array of raw scores
>>> F_test  = any_model.predict(X_test)
>>> qpm = QPMQuantizer(focus_pd=(0.05, 0.20), n_low_max=6, n_mid_max=6,
...                    n_high_max=6, min_band_size=300, pd_cap=0.80)
>>> qpm.fit(F_train, y_train)
>>> y_hat = qpm.predict(F_test)            # discrete PD ladder
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class QPMQuantizer:
    """
    Quantized Prediction Model quantizer.

    Constrains model output to a finite, ordered set of prediction levels
    (a "ladder") by learning a supervised, non-uniform quantizer on top of
    any continuous latent scoring function F(x).

    The algorithm (Section 3.2):
    1. Partition the score range into three risk regions based on ``focus_pd``.
    2. Grid-search over ``(n_low, n_mid, n_high)`` bin counts per region.
    3. Within each region, place quantile-based bin edges.
    4. Merge interior bins smaller than ``min_band_size`` into neighbors.
    5. Compute bin representatives via Laplace-smoothed empirical means.
    6. Optionally project to a monotone sequence via isotonic regression.

    Parameters
    ----------
    focus_pd : tuple of float, default (0.05, 0.20)
        (lo, hi) PD thresholds that define the three risk regions:
        low  := F < lo
        mid  := lo <= F <= hi  (focus / high-resolution region)
        high := F > hi
    n_low_max, n_mid_max, n_high_max : int
        Maximum number of bins to attempt in each region (grid search bound).
    min_band_size : int, default 300
        Minimum number of training samples required per interior bin.
        The first and last bins (boundary catch-alls) are exempt.
    prior_pi0 : float or None
        Prior PD for Laplace smoothing.  Defaults to training prevalence.
    prior_weight : float, default 30
        Pseudo-count weight for Laplace smoothing.
    pd_cap : float, default 0.80
        Maximum allowed PD value for any bin representative.
    monotone_smooth : bool, default True
        If True, project bin representatives onto a monotone (non-decreasing)
        sequence via isotonic regression after smoothing.
    """

    def __init__(
        self,
        focus_pd: tuple[float, float] = (0.05, 0.20),
        n_low_max: int = 6,
        n_mid_max: int = 6,
        n_high_max: int = 6,
        min_band_size: int = 300,
        prior_pi0: float | None = None,
        prior_weight: float = 30.0,
        pd_cap: float = 0.80,
        monotone_smooth: bool = True,
    ):
        self.focus_pd = focus_pd
        self.n_low_max = n_low_max
        self.n_mid_max = n_mid_max
        self.n_high_max = n_high_max
        self.min_band_size = min_band_size
        self.prior_pi0 = prior_pi0
        self.prior_weight = prior_weight
        self.pd_cap = pd_cap
        self.monotone_smooth = monotone_smooth

        # Attributes set after fit()
        self.bin_edges_: np.ndarray | None = None   # length K+1; includes -inf and +inf
        self.bin_reps_: np.ndarray | None = None    # length K; smoothed PD per bin
        self.n_bins_: int | None = None
        self.chosen_n_: tuple[int, int, int] | None = None  # (n_low, n_mid, n_high) chosen
        self.prior_pi0_: float | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, F_train: np.ndarray, y_train: np.ndarray) -> "QPMQuantizer":
        """
        Fit the quantizer.

        Parameters
        ----------
        F_train : array-like of shape (n,)
            Latent scores from any backbone model (XGBoost, LightGBM, etc.).
        y_train : array-like of shape (n,)
            Binary labels in {0, 1}.

        Returns
        -------
        self
        """
        F_train = np.asarray(F_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if F_train.ndim != 1:
            raise ValueError("F_train must be 1-D.")

        self.prior_pi0_ = float(self.prior_pi0) if self.prior_pi0 is not None else float(y_train.mean())

        self.bin_edges_, self.chosen_n_ = self._search_bins(F_train, y_train)
        self.n_bins_ = len(self.bin_edges_) - 1
        self.bin_reps_ = self._compute_representatives(F_train, y_train, self.bin_edges_)
        self._is_fitted = True
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        """
        Apply the quantizer: map continuous latent scores to discrete PD levels.

        Parameters
        ----------
        F : array-like of shape (n,)
            Latent scores (from the same backbone used during fit).

        Returns
        -------
        y_hat : ndarray of shape (n,)
            Discrete PD predictions drawn from the learned bin representatives.
        """
        self._check_fitted()
        F = np.asarray(F, dtype=float)
        idx = self.get_bin_index(F)
        return self.bin_reps_[idx]

    def get_bin_index(self, F: np.ndarray) -> np.ndarray:
        """
        Return the 0-based bin index for each latent score.

        Parameters
        ----------
        F : array-like of shape (n,)

        Returns
        -------
        idx : ndarray of int, shape (n,)
        """
        self._check_fitted()
        F = np.asarray(F, dtype=float)
        # bin_edges_ = [-inf, b1, b2, ..., b_K, +inf]
        # searchsorted on the K-1 interior edges maps each score to [0, K-1]
        return np.searchsorted(self.bin_edges_[1:-1], F, side="right")

    def score_ladder(
        self,
        F_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame summarising the score ladder (Table 2 in the paper).

        Columns: Grade, Band, F_min, F_max, PD, #Acct, %Portf, #Def, DefR,
                 Odds, Lift, Cum%Acct, Cum%Def.

        Parameters
        ----------
        F_test, y_test : optional
            If provided, add test-set count / default-rate statistics.
            If omitted, only the structural columns (Grade, Band, F_min,
            F_max, PD) are returned.
        """
        self._check_fitted()
        K = self.n_bins_
        rows = []
        overall_dr = float(y_test.mean()) if y_test is not None else None

        cum_acct = 0
        cum_def = 0
        total_acct = len(F_test) if F_test is not None else None
        total_def = int(y_test.sum()) if y_test is not None else None

        for k in range(K):
            f_min_raw = self.bin_edges_[k]
            f_max_raw = self.bin_edges_[k + 1]
            f_min = 0.0 if np.isinf(f_min_raw) else float(f_min_raw)
            f_max = 1.0 if np.isinf(f_max_raw) else float(f_max_raw)

            row: dict = {
                "Grade": f"G{k + 1:02d}",
                "Band": k,
                "F_min": round(f_min, 4),
                "F_max": round(f_max, 4),
                "PD": round(self.bin_reps_[k], 4),
            }

            if F_test is not None and y_test is not None:
                F_test = np.asarray(F_test, dtype=float)
                y_test = np.asarray(y_test, dtype=float)
                mask = self.get_bin_index(F_test) == k
                n_acct = int(mask.sum())
                n_def = int(y_test[mask].sum())
                cum_acct += n_acct
                cum_def += n_def
                def_r = n_def / n_acct if n_acct > 0 else 0.0
                odds = def_r / (1 - def_r) if def_r < 1 else float("inf")
                lift = (def_r / overall_dr) if overall_dr and overall_dr > 0 else None

                row.update(
                    {
                        "#Acct": n_acct,
                        "%Portf": round(n_acct / total_acct, 4) if total_acct else None,
                        "#Def": n_def,
                        "DefR": round(def_r, 4),
                        "Odds": round(odds, 2),
                        "Lift": round(lift, 4) if lift is not None else None,
                        "Cum%Acct": round(cum_acct / total_acct, 4) if total_acct else None,
                        "Cum%Def": round(cum_def / total_def, 4) if total_def else None,
                    }
                )
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_bins(
        self,
        F_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """
        Grid-search over (n_low, n_mid, n_high) to maximise surviving bins K
        subject to min_band_size for interior bins (Section 3.2).
        """
        focus_lo, focus_hi = self.focus_pd
        low_mask = F_train < focus_lo
        mid_mask = (F_train >= focus_lo) & (F_train <= focus_hi)
        high_mask = F_train > focus_hi

        # Candidates sorted by total K desc, then n_mid desc (prefer focus resolution)
        candidates = sorted(
            [
                (n_l, n_m, n_h)
                for n_l in range(self.n_low_max, 0, -1)
                for n_m in range(self.n_mid_max, 0, -1)
                for n_h in range(self.n_high_max, 0, -1)
            ],
            key=lambda x: (x[0] + x[1] + x[2], x[1]),
            reverse=True,
        )

        best_K = -1
        best_edges: np.ndarray = np.array([-np.inf, np.inf])
        best_n: tuple[int, int, int] = (1, 1, 1)

        for n_l, n_m, n_h in candidates:
            # Early exit: no candidate with more total bins can be found
            if (n_l + n_m + n_h) < best_K:
                break

            raw_edges = self._place_edges(F_train, low_mask, mid_mask, high_mask, n_l, n_m, n_h)
            edges = self._merge_small_bins(F_train, raw_edges)
            K = len(edges) - 1

            if K > best_K:
                best_K = K
                best_edges = edges
                best_n = (n_l, n_m, n_h)

        return best_edges, best_n

    def _place_edges(
        self,
        F: np.ndarray,
        low_mask: np.ndarray,
        mid_mask: np.ndarray,
        high_mask: np.ndarray,
        n_low: int,
        n_mid: int,
        n_high: int,
    ) -> np.ndarray:
        """
        Place quantile-based bin edges within each risk region.

        Each region with n_k bins gets n_k-1 internal quantile cut-points.
        The edges across all regions are concatenated, deduplicated, and
        wrapped with ±inf sentinels.
        """
        interior: list[float] = []

        for mask, n_bins in ((low_mask, n_low), (mid_mask, n_mid), (high_mask, n_high)):
            F_region = F[mask]
            if len(F_region) == 0 or n_bins <= 1:
                continue
            quantiles = np.linspace(0.0, 100.0, n_bins + 1)[1:-1]
            cuts = np.percentile(F_region, quantiles)
            interior.extend(cuts.tolist())

        # Remove duplicates, sort, and add sentinels
        interior_unique = np.unique(interior)
        return np.concatenate([[-np.inf], interior_unique, [np.inf]])

    def _merge_small_bins(self, F: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Merge interior bins with fewer than min_band_size samples into their
        smaller neighbor.  The first and last bins (boundary catch-alls) are
        never merged.
        """
        edges = list(edges)

        while True:
            n_bins = len(edges) - 1
            if n_bins <= 1:
                break

            # Count samples per interior bin (indices 1 .. n_bins-2 in bin-space)
            merged = False
            for k in range(1, n_bins - 1):  # skip first (k=0) and last (k=n_bins-1)
                lo, hi = edges[k], edges[k + 1]
                count = int(((F >= lo) & (F < hi)).sum())
                if count < self.min_band_size:
                    # Merge with the smaller of the left or right neighbor
                    left_count = int(((F >= edges[k - 1]) & (F < edges[k])).sum())
                    right_count_idx = k + 1
                    if right_count_idx < n_bins - 1:
                        right_count = int(((F >= edges[k + 1]) & (F < edges[k + 2])).sum())
                    else:
                        right_count = int(np.iinfo(np.int64).max)  # last bin: never prefer

                    if left_count <= right_count:
                        edges.pop(k)   # absorb into left bin
                    else:
                        edges.pop(k + 1)   # absorb into right bin
                    merged = True
                    break  # restart scan after each merge

            if not merged:
                break

        return np.array(edges)

    def _compute_representatives(
        self,
        F: np.ndarray,
        y: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        """
        Compute bin-level PD representatives (Section 3.2):
        1. Empirical conditional mean per bin.
        2. Laplace/Bayesian smoothing toward prior_pi0_.
        3. PD cap at pd_cap.
        4. Optional monotone projection via isotonic regression.
        """
        K = len(edges) - 1
        reps = np.zeros(K)

        # Inline bin assignment — avoids the _is_fitted guard (called mid-fit)
        bin_idx = np.searchsorted(edges[1:-1], F, side="right")

        for k in range(K):
            mask = bin_idx == k
            n_k = int(mask.sum())
            emp_mean = float(y[mask].mean()) if n_k > 0 else self.prior_pi0_

            # Laplace / Bayesian smoothing
            smoothed = (n_k * emp_mean + self.prior_weight * self.prior_pi0_) / (
                n_k + self.prior_weight
            )
            reps[k] = min(smoothed, self.pd_cap)

        if self.monotone_smooth:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            reps = iso.fit_transform(np.arange(K, dtype=float), reps)
            reps = np.minimum(reps, self.pd_cap)

        return reps

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("QPMQuantizer is not fitted. Call fit() first.")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = f", K={self.n_bins_}" if self._is_fitted else ""
        return (
            f"QPMQuantizer(focus_pd={self.focus_pd}, "
            f"n_low_max={self.n_low_max}, n_mid_max={self.n_mid_max}, "
            f"n_high_max={self.n_high_max}, min_band_size={self.min_band_size}"
            f"{fitted})"
        )
