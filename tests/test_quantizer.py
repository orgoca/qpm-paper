import numpy as np
import pytest

from qpm import QPMQuantizer


def test_fit_returns_self(synthetic_data):
    F, y = synthetic_data
    qpm = QPMQuantizer(min_band_size=50)
    result = qpm.fit(F, y)
    assert result is qpm


def test_predict_shape(fitted_qpm, synthetic_data):
    qpm, _, _ = fitted_qpm
    F, _ = synthetic_data
    preds = qpm.predict(F)
    assert preds.shape == F.shape


def test_predict_values_from_bin_reps(fitted_qpm, synthetic_data):
    qpm, _, _ = fitted_qpm
    F, _ = synthetic_data
    preds = qpm.predict(F)
    for p in preds:
        assert p in qpm.bin_reps_


def test_bin_reps_monotone(fitted_qpm):
    qpm, _, _ = fitted_qpm
    reps = qpm.bin_reps_
    assert np.all(reps[1:] >= reps[:-1]), "bin_reps_ should be non-decreasing"


def test_bin_reps_monotone_disabled(synthetic_data):
    F, y = synthetic_data
    qpm = QPMQuantizer(min_band_size=50, monotone_smooth=False)
    qpm.fit(F, y)
    assert qpm.bin_reps_ is not None


def test_bin_reps_capped(fitted_qpm):
    qpm, _, _ = fitted_qpm
    assert np.all(qpm.bin_reps_ <= qpm.pd_cap)


def test_n_bins_positive(fitted_qpm):
    qpm, _, _ = fitted_qpm
    assert qpm.n_bins_ >= 1


def test_predict_before_fit_raises():
    qpm = QPMQuantizer()
    with pytest.raises(RuntimeError, match="not fitted"):
        qpm.predict(np.array([0.1, 0.2]))


def test_get_bin_index_before_fit_raises():
    qpm = QPMQuantizer()
    with pytest.raises(RuntimeError, match="not fitted"):
        qpm.get_bin_index(np.array([0.1]))


def test_get_bin_index_range(fitted_qpm, synthetic_data):
    qpm, _, _ = fitted_qpm
    F, _ = synthetic_data
    idx = qpm.get_bin_index(F)
    assert idx.min() >= 0
    assert idx.max() <= qpm.n_bins_ - 1


def test_fit_2d_raises(synthetic_data):
    F, y = synthetic_data
    qpm = QPMQuantizer(min_band_size=50)
    with pytest.raises(ValueError, match="1-D"):
        qpm.fit(F.reshape(-1, 1), y)


def test_score_ladder_no_test_data(fitted_qpm):
    qpm, _, _ = fitted_qpm
    df = qpm.score_ladder()
    assert list(df.columns[:5]) == ["Grade", "Band", "F_min", "F_max", "PD"]
    assert len(df) == qpm.n_bins_


def test_score_ladder_with_test_data(fitted_qpm, synthetic_data):
    qpm, F, y = fitted_qpm
    df = qpm.score_ladder(F, y)
    expected_cols = {"Grade", "Band", "F_min", "F_max", "PD", "#Acct", "%Portf", "#Def", "DefR"}
    assert expected_cols.issubset(set(df.columns))
    assert df["#Acct"].sum() == len(F)


def test_repr_fitted(fitted_qpm):
    qpm, _, _ = fitted_qpm
    r = repr(qpm)
    assert f"K={qpm.n_bins_}" in r


def test_repr_unfitted():
    qpm = QPMQuantizer()
    assert "QPMQuantizer" in repr(qpm)
    assert "K=" not in repr(qpm)


def test_chosen_n_set(fitted_qpm):
    qpm, _, _ = fitted_qpm
    assert qpm.chosen_n_ is not None
    assert len(qpm.chosen_n_) == 3
