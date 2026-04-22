import numpy as np
import pytest


@pytest.fixture
def synthetic_data():
    """2000-sample binary classification dataset with a predictive continuous score."""
    rng = np.random.default_rng(42)
    n = 2000
    F = rng.uniform(0.0, 1.0, n)
    y = rng.binomial(1, F)
    return F, y


@pytest.fixture
def fitted_qpm(synthetic_data):
    from qpm import QPMQuantizer

    F, y = synthetic_data
    qpm = QPMQuantizer(
        focus_pd=(0.3, 0.7),
        n_low_max=3,
        n_mid_max=3,
        n_high_max=3,
        min_band_size=50,
    )
    qpm.fit(F, y)
    return qpm, F, y
