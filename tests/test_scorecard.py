import numpy as np
import pandas as pd
import pytest

from qpm import WoEScorecard


@pytest.fixture
def scorecard_data():
    rng = np.random.default_rng(99)
    n = 800
    X = rng.normal(0, 1, (n, 4))
    logit = X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return X, y


@pytest.fixture
def fitted_scorecard(scorecard_data):
    X, y = scorecard_data
    sc = WoEScorecard(n_quantile_bins=5, n_score_bands=5)
    sc.fit(X, y)
    return sc, X, y


def test_fit_returns_self(scorecard_data):
    X, y = scorecard_data
    sc = WoEScorecard(n_quantile_bins=5, n_score_bands=5)
    assert sc.fit(X, y) is sc


def test_predict_proba_shape(fitted_scorecard, scorecard_data):
    sc, X, y = fitted_scorecard
    preds = sc.predict_proba(X)
    assert preds.shape == (len(X),)


def test_predict_proba_in_01(fitted_scorecard, scorecard_data):
    sc, X, y = fitted_scorecard
    preds = sc.predict_proba(X)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_score_ladder_row_count(fitted_scorecard, scorecard_data):
    sc, X, y = fitted_scorecard
    df = sc.score_ladder(X, y)
    assert len(df) == sc.n_score_bands


def test_score_ladder_account_sum(fitted_scorecard, scorecard_data):
    sc, X, y = fitted_scorecard
    df = sc.score_ladder(X, y)
    assert df["#Acct"].sum() == len(y)


def test_score_ladder_columns(fitted_scorecard, scorecard_data):
    sc, X, y = fitted_scorecard
    df = sc.score_ladder(X, y)
    for col in ("Band", "PD", "#Acct", "#Def", "DefR"):
        assert col in df.columns


def test_unfitted_predict_proba_raises():
    sc = WoEScorecard()
    with pytest.raises(RuntimeError, match="not fitted"):
        sc.predict_proba(np.ones((5, 3)))


def test_unfitted_score_ladder_raises():
    sc = WoEScorecard()
    with pytest.raises(RuntimeError, match="not fitted"):
        sc.score_ladder(np.ones((5, 3)), np.zeros(5))


def test_dataframe_input(scorecard_data):
    X, y = scorecard_data
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    sc = WoEScorecard(n_quantile_bins=5, n_score_bands=5)
    sc.fit(df, y)
    preds = sc.predict_proba(df)
    assert preds.shape == (len(y),)


def test_auc_better_than_random(fitted_scorecard, scorecard_data):
    from qpm import auc_score

    sc, X, y = fitted_scorecard
    preds = sc.predict_proba(X)
    assert auc_score(y, preds) > 0.55
