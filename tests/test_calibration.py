import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from qpm import AnchoredCalibration
from qpm.calibration import residual_stack


def test_alpha_1_no_change(fitted_qpm, synthetic_data):
    qpm, F, y = fitted_qpm
    original_reps = qpm.bin_reps_.copy()
    cal = AnchoredCalibration(alpha=1.0)
    cal.update(qpm, F, y)
    np.testing.assert_array_almost_equal(qpm.bin_reps_, original_reps)


def test_alpha_0_changes_reps(fitted_qpm, synthetic_data):
    qpm, F, y = fitted_qpm
    original_reps = qpm.bin_reps_.copy()
    rng = np.random.default_rng(20)
    F_new = rng.uniform(0, 1, 500)
    y_new = rng.binomial(1, F_new)
    cal = AnchoredCalibration(alpha=0.0)
    cal.update(qpm, F_new, y_new)
    # At least some bins should differ
    assert not np.allclose(qpm.bin_reps_, original_reps)


def test_bin_edges_unchanged(fitted_qpm, synthetic_data):
    qpm, F, y = fitted_qpm
    original_edges = qpm.bin_edges_.copy()
    cal = AnchoredCalibration(alpha=0.5)
    cal.update(qpm, F, y)
    np.testing.assert_array_equal(qpm.bin_edges_, original_edges)


def test_n_bins_unchanged(fitted_qpm, synthetic_data):
    qpm, F, y = fitted_qpm
    original_n_bins = qpm.n_bins_
    cal = AnchoredCalibration(alpha=0.5)
    cal.update(qpm, F, y)
    assert qpm.n_bins_ == original_n_bins


def test_result_monotone(fitted_qpm, synthetic_data):
    qpm, F, y = fitted_qpm
    cal = AnchoredCalibration(alpha=0.3, monotone_smooth=True)
    new_reps = cal.update(qpm, F, y)
    assert np.all(new_reps[1:] >= new_reps[:-1])


def test_result_capped(fitted_qpm, synthetic_data):
    qpm, F, y = fitted_qpm
    cal = AnchoredCalibration(alpha=0.0, pd_cap=0.5)
    new_reps = cal.update(qpm, F, y)
    assert np.all(new_reps <= 0.5)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        AnchoredCalibration(alpha=1.5)


def test_update_returns_array(fitted_qpm, synthetic_data):
    qpm, F, y = fitted_qpm
    cal = AnchoredCalibration(alpha=0.5)
    result = cal.update(qpm, F, y)
    assert isinstance(result, np.ndarray)
    assert result.shape == (qpm.n_bins_,)


def test_alpha_override(fitted_qpm, synthetic_data):
    qpm, F, y = fitted_qpm
    original_reps = qpm.bin_reps_.copy()
    cal = AnchoredCalibration(alpha=0.0)
    # Override to alpha=1 → no change
    cal.update(qpm, F, y, alpha=1.0)
    np.testing.assert_array_almost_equal(qpm.bin_reps_, original_reps)


# ---------------------------------------------------------------------------
# residual_stack
# ---------------------------------------------------------------------------


@pytest.fixture
def residual_stack_data():
    rng = np.random.default_rng(77)
    n, d = 300, 5
    X = rng.normal(0, 1, (n, d))
    y = (rng.normal(0, 1, n) > 0).astype(float)
    base = GradientBoostingRegressor(n_estimators=20, random_state=0)
    base.fit(X, y)
    return base, X, y


def test_residual_stack_output_shape(residual_stack_data):
    base, X, y = residual_stack_data
    F_updated, booster = residual_stack(base, X, y, n_estimators=10)
    assert F_updated.shape == (len(X),)


def test_residual_stack_returns_booster(residual_stack_data):
    base, X, y = residual_stack_data
    _, booster = residual_stack(base, X, y, n_estimators=10)
    assert hasattr(booster, "predict")


def test_residual_stack_differs_from_base(residual_stack_data):
    base, X, y = residual_stack_data
    F_base = base.predict(X)
    F_updated, _ = residual_stack(base, X, y, n_estimators=10)
    assert not np.allclose(F_updated, F_base)


def test_residual_stack_booster_predict_shape(residual_stack_data):
    base, X, y = residual_stack_data
    _, booster = residual_stack(base, X, y, n_estimators=10)
    preds = booster.predict(X)
    assert preds.shape == (len(X),)
