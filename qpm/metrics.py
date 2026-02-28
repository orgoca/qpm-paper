"""
Evaluation Metrics
==================
All metrics used in "Quantized Prediction Models" (Ortiz-Gómez, 2025).

Includes:
- AUC, KS, MSE
- Bootstrap confidence intervals (bias-corrected and accelerated, BCa)
- DeLong test for paired AUC comparison
- Churn metrics (percentile shift, band churn, native QPM band churn)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(np.asarray(y_true), np.asarray(scores)))


def ks_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: maximum separation between the empirical
    CDFs of defaulters and non-defaulters (two-sample KS).
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    return float(stats.ks_2samp(s[y == 1], s[y == 0]).statistic)


def mse_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mean squared error."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    return float(np.mean((y - s) ** 2))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (BCa)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    ci: float = 0.95,
    method: str = "bca",
    seed: int = 0,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for a scalar metric.

    Implements the bias-corrected and accelerated (BCa) interval by default,
    matching the methodology in Appendix E of the paper.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    scores : array-like of shape (n,)
    metric_fn : callable(y_true, scores) -> float
        E.g. ``auc_score``, ``ks_score``.
    n_boot : int, default 2000
        Number of bootstrap resamples (paper uses 2000).
    ci : float, default 0.95
        Confidence level.
    method : {'bca', 'percentile'}
        'bca'       — bias-corrected and accelerated (default, recommended).
        'percentile'— plain percentile interval.
    seed : int, default 0

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper confidence bounds.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    n = len(y)

    theta_hat = metric_fn(y, s)

    # Bootstrap distribution
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = metric_fn(y[idx], s[idx])

    alpha = 1.0 - ci

    if method == "percentile":
        lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        return lo, hi

    # BCa correction
    # Bias correction z0
    z0 = float(stats.norm.ppf(np.mean(boot_stats < theta_hat)))

    # Acceleration a via jackknife
    jack_stats = np.empty(n)
    for i in range(n):
        idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        jack_stats[i] = metric_fn(y[idx], s[idx])
    jack_mean = jack_stats.mean()
    diffs = jack_mean - jack_stats
    num = np.sum(diffs**3)
    den = 6.0 * (np.sum(diffs**2) ** 1.5)
    a = float(num / den) if den != 0 else 0.0

    # Adjusted percentiles
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    def _adj(z_alpha: float) -> float:
        denom = 1.0 - a * (z0 + z_alpha)
        if denom == 0:
            return alpha / 2
        arg = z0 + (z0 + z_alpha) / denom
        return float(stats.norm.cdf(arg))

    p_lo = _adj(z_lo)
    p_hi = _adj(z_hi)

    lo = float(np.percentile(boot_stats, 100 * p_lo))
    hi = float(np.percentile(boot_stats, 100 * p_hi))
    return lo, hi


# ---------------------------------------------------------------------------
# DeLong test for correlated AUC comparison (Appendix E)
# ---------------------------------------------------------------------------


def delong_test(
    y_true: np.ndarray,
    scores1: np.ndarray,
    scores2: np.ndarray,
) -> tuple[float, float]:
    """
    DeLong test for comparing two correlated AUC estimates.

    Uses the exact, distribution-free DeLong (1988) covariance estimator.
    Suitable for paired predictions (both models evaluated on the same test
    set), as in Experiment 3 of the paper.

    The test statistic is:
        Z = (AUC1 - AUC2) / sqrt(Var(AUC1) + Var(AUC2) - 2*Cov(AUC1, AUC2))

    with a two-sided p-value under the standard normal approximation.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels.
    scores1, scores2 : array-like of shape (n,)
        Predicted scores from model 1 and model 2.

    Returns
    -------
    z_stat : float
    p_value : float (two-sided)
    """
    y = np.asarray(y_true, dtype=float)
    s1 = np.asarray(scores1, dtype=float)
    s2 = np.asarray(scores2, dtype=float)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both classes.")

    # Placement values (V10, V01) for each model
    def _placement(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        V10[i] = fraction of negatives with score < scores[pos_i]  (for each positive)
        V01[j] = fraction of positives with score > scores[neg_j]  (for each negative)
        """
        s_pos = scores[pos_idx]
        s_neg = scores[neg_idx]
        v10 = np.array([np.mean(s_neg < sp) + 0.5 * np.mean(s_neg == sp) for sp in s_pos])
        v01 = np.array([np.mean(s_pos > sn) + 0.5 * np.mean(s_pos == sn) for sn in s_neg])
        return v10, v01

    v10_1, v01_1 = _placement(s1)
    v10_2, v01_2 = _placement(s2)

    auc1 = float(v10_1.mean())
    auc2 = float(v10_2.mean())

    # DeLong covariance matrix S = [[s11, s12],[s12, s22]]
    def _cov(v10_a, v01_a, v10_b, v01_b, auc_a, auc_b):
        term1 = np.cov(v10_a, v10_b)[0, 1] / n_pos
        term2 = np.cov(v01_a, v01_b)[0, 1] / n_neg
        return term1 + term2

    s11 = np.var(v10_1, ddof=1) / n_pos + np.var(v01_1, ddof=1) / n_neg
    s22 = np.var(v10_2, ddof=1) / n_pos + np.var(v01_2, ddof=1) / n_neg
    s12 = _cov(v10_1, v01_1, v10_2, v01_2, auc1, auc2)

    var_diff = s11 + s22 - 2.0 * s12
    if var_diff <= 0:
        return 0.0, 1.0

    z_stat = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
    return float(z_stat), p_value


# ---------------------------------------------------------------------------
# Churn metrics (Section 4.1.3)
# ---------------------------------------------------------------------------


def churn_metrics(
    scores_v1: np.ndarray,
    scores_v2: np.ndarray,
    bins_v1: np.ndarray | None = None,
    bins_v2: np.ndarray | None = None,
    n_global_bands: int = 5,
) -> dict:
    """
    Compute prediction churn between model version 1 and version 2.

    Two churn measures are reported in the paper:

    1. **Percentile churn** (continuous models, or QPM via quantized PD):
       Mean absolute change in percentile rank on the common population.

    2. **Band churn** (n_global_bands equal-size global percentile bands):
       Fraction of accounts that move to a different band.  Accounts
       moving two or more bands are reported separately.

    3. **Native band churn** (QPM-specific, requires ``bins_v1`` / ``bins_v2``):
       Fraction of accounts that change QPM bin assignment between versions.

    Parameters
    ----------
    scores_v1, scores_v2 : array-like of shape (n,)
        Model scores (continuous or quantized PD) for the same population
        evaluated under version 1 and version 2.
    bins_v1, bins_v2 : array-like of int, shape (n,), optional
        QPM bin indices (from ``QPMQuantizer.get_bin_index``) for v1 and v2.
        Required to compute native band churn.
    n_global_bands : int, default 5
        Number of global percentile bands for the band churn metric.

    Returns
    -------
    result : dict with keys:
        'mean_abs_percentile_shift'  — mean |rank_v1 - rank_v2|
        'band_churn_rate'            — fraction changing global band
        'two_plus_band_churn_rate'   — fraction moving ≥2 global bands
        'native_band_churn_rate'     — fraction changing QPM bin (if provided)
        'native_two_plus_churn_rate' — fraction moving ≥2 QPM bins (if provided)
    """
    s1 = np.asarray(scores_v1, dtype=float)
    s2 = np.asarray(scores_v2, dtype=float)
    n = len(s1)

    # Percentile ranks (0 to 1)
    rank_v1 = _percentile_rank(s1)
    rank_v2 = _percentile_rank(s2)
    mean_abs_shift = float(np.mean(np.abs(rank_v1 - rank_v2)))

    # Global percentile bands (equal-size)
    band_edges = np.linspace(0.0, 1.0, n_global_bands + 1)
    global_band_v1 = np.searchsorted(band_edges[1:-1], rank_v1, side="right")
    global_band_v2 = np.searchsorted(band_edges[1:-1], rank_v2, side="right")

    band_diff = np.abs(global_band_v1.astype(int) - global_band_v2.astype(int))
    band_churn_rate = float(np.mean(band_diff >= 1))
    two_plus_band_churn_rate = float(np.mean(band_diff >= 2))

    result = {
        "mean_abs_percentile_shift": mean_abs_shift,
        "band_churn_rate": band_churn_rate,
        "two_plus_band_churn_rate": two_plus_band_churn_rate,
    }

    # Native QPM band churn
    if bins_v1 is not None and bins_v2 is not None:
        b1 = np.asarray(bins_v1, dtype=int)
        b2 = np.asarray(bins_v2, dtype=int)
        native_diff = np.abs(b1 - b2)
        result["native_band_churn_rate"] = float(np.mean(native_diff >= 1))
        result["native_two_plus_churn_rate"] = float(np.mean(native_diff >= 2))

    return result


def _percentile_rank(scores: np.ndarray) -> np.ndarray:
    """
    Return fractional percentile rank in [0, 1] for each score.
    Ties are broken by averaging (mid-rank convention).
    """
    n = len(scores)
    order = np.argsort(scores)
    ranks = np.empty(n)
    ranks[order] = np.arange(n)

    # Handle ties: assign mean rank to tied values
    sorted_scores = scores[order]
    unique_vals, first_occ, counts = np.unique(
        sorted_scores, return_index=True, return_counts=True
    )
    for val, idx, cnt in zip(unique_vals, first_occ, counts):
        if cnt > 1:
            mid = idx + (cnt - 1) / 2.0
            ranks[order[idx : idx + cnt]] = mid

    return ranks / (n - 1) if n > 1 else np.zeros(n)
