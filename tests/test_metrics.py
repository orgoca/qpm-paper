import numpy as np
import pytest

from qpm import auc_score, ks_score, mse_score, bootstrap_ci, delong_test, churn_metrics


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------


def test_auc_perfect():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auc_score(y, s) == pytest.approx(1.0)


def test_auc_random():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.uniform(0, 1, 500)
    assert 0.4 < auc_score(y, s) < 0.6


def test_ks_perfect():
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([np.linspace(0, 0.4, 100), np.linspace(0.6, 1.0, 100)])
    assert ks_score(y, s) > 0.9


def test_ks_random():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = rng.uniform(0, 1, 500)
    assert ks_score(y, s) < 0.2


def test_mse_perfect():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert mse_score(y, y) == pytest.approx(0.0)


def test_mse_known():
    y = np.array([1.0, 0.0])
    s = np.array([0.0, 1.0])
    assert mse_score(y, s) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_contains_true(synthetic_data):
    F, y = synthetic_data
    true_auc = auc_score(y, F)
    lo, hi = bootstrap_ci(y, F, auc_score, n_boot=500, seed=7)
    assert lo < true_auc < hi


def test_bootstrap_ci_percentile(synthetic_data):
    F, y = synthetic_data
    lo, hi = bootstrap_ci(y, F, auc_score, n_boot=200, method="percentile", seed=7)
    assert lo < hi


def test_bootstrap_ci_ordering(synthetic_data):
    F, y = synthetic_data
    lo, hi = bootstrap_ci(y, F, auc_score, n_boot=200, seed=0)
    assert lo < hi


# ---------------------------------------------------------------------------
# DeLong test
# ---------------------------------------------------------------------------


def test_delong_same_scores(synthetic_data):
    F, y = synthetic_data
    z, p = delong_test(y, F, F)
    assert z == pytest.approx(0.0, abs=1e-10)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_delong_returns_floats(synthetic_data):
    F, y = synthetic_data
    rng = np.random.default_rng(3)
    s2 = rng.uniform(0, 1, len(F))
    z, p = delong_test(y, F, s2)
    assert isinstance(z, float)
    assert isinstance(p, float)


def test_delong_p_in_01(synthetic_data):
    F, y = synthetic_data
    rng = np.random.default_rng(5)
    s2 = rng.uniform(0, 1, len(F))
    _, p = delong_test(y, F, s2)
    assert 0.0 <= p <= 1.0


def test_delong_no_single_class():
    y_bad = np.zeros(10)
    s = np.random.uniform(0, 1, 10)
    with pytest.raises(ValueError, match="both classes"):
        delong_test(y_bad, s, s)


# ---------------------------------------------------------------------------
# Churn metrics
# ---------------------------------------------------------------------------


def test_churn_no_churn(synthetic_data):
    F, _ = synthetic_data
    result = churn_metrics(F, F)
    assert result["mean_abs_percentile_shift"] == pytest.approx(0.0, abs=1e-10)
    assert result["band_churn_rate"] == pytest.approx(0.0)
    assert result["two_plus_band_churn_rate"] == pytest.approx(0.0)


def test_churn_keys_present(synthetic_data):
    F, _ = synthetic_data
    result = churn_metrics(F, F)
    assert "mean_abs_percentile_shift" in result
    assert "band_churn_rate" in result
    assert "two_plus_band_churn_rate" in result
    assert "native_band_churn_rate" not in result


def test_churn_native_keys(synthetic_data, fitted_qpm):
    F, _ = synthetic_data
    qpm, _, _ = fitted_qpm
    b = qpm.get_bin_index(F)
    result = churn_metrics(F, F, bins_v1=b, bins_v2=b)
    assert result["native_band_churn_rate"] == pytest.approx(0.0)
    assert result["native_two_plus_churn_rate"] == pytest.approx(0.0)


def test_churn_rates_in_01(synthetic_data):
    F, _ = synthetic_data
    rng = np.random.default_rng(9)
    F2 = rng.uniform(0, 1, len(F))
    result = churn_metrics(F, F2)
    assert 0.0 <= result["band_churn_rate"] <= 1.0
    assert 0.0 <= result["two_plus_band_churn_rate"] <= 1.0
