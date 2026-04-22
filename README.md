# Quantized Prediction Models (QPMs)

Output-Space Regularization for Stable and Interpretable Regression

Carlos R. Ortiz-Gomez, Independent Researcher, December 2025  
DOI: [10.5281/zenodo.17785949](https://doi.org/10.5281/zenodo.17785949)

## Overview

This repository contains the reference implementation and reproducibility notebooks for the QPM paper.

QPM is a post-hoc output quantizer: you first train any continuous scorer `F(x)`, then fit `QPMQuantizer` on top of those scores to produce an ordered, discrete PD ladder. The method is model-agnostic and works with XGBoost, LightGBM, scikit-learn models, neural networks, or any source of 1-D scores.

```python
from qpm import QPMQuantizer

# 1) Train any backbone model and obtain latent scores
F_train = model.predict(X_train)
F_test = model.predict(X_test)

# 2) Fit QPM post-hoc
qpm = QPMQuantizer(
    focus_pd=(0.05, 0.20),
    n_low_max=6, n_mid_max=6, n_high_max=6,
    min_band_size=300, pd_cap=0.80,
)
qpm.fit(F_train, y_train)

# 3) Quantized predictions (discrete PD ladder)
y_hat = qpm.predict(F_test)
```

## Installation

```bash
git clone https://github.com/orgoca/qpm-paper.git
cd qpm-paper
pip install -e .
```

Python 3.9+ required.

## Reproducing Experiments

Run notebooks in order:

| Notebook | Experiment | Outputs |
|---|---|---|
| `notebooks/01_uci_default.ipynb` | Exp 1: UCI Default | Table 1, Table 2, Figure 1, churn analysis |
| `notebooks/02_home_credit.ipynb` | Exp 2: Home Credit | Table 3, churn analysis |
| `notebooks/03_framingham.ipynb` | Exp 3: Framingham CHD | AUC CIs, DeLong test, discrete risk ladder |

Each notebook writes canonical outputs to:

- `artifacts/exp01/report.json`
- `artifacts/exp02/report.json`
- `artifacts/exp03/report.json`

An executed reference notebook for Exp 3 is also included at `notebooks/03_framingham_executed.ipynb`.

## Data Setup

See [data/README.md](data/README.md) for details.

- Exp 1 (UCI): downloaded automatically via `ucimlrepo`
- Exp 2 (Home Credit): Kaggle API token required
- Exp 3 (Framingham): Kaggle API token required

## Package Structure

```text
qpm/
|- quantizer.py     # QPMQuantizer: supervised output quantization
|- calibration.py   # AnchoredCalibration + residual_stack
|- metrics.py       # AUC, KS, MSE, BCa bootstrap, DeLong, churn
`- scorecard.py     # WoE logistic scorecard baseline
```

## Canonical Results (Current Repo Runs)

The values below were produced by running the notebooks locally. Artifact files
(`artifacts/*/report.json`) are generated on-disk when you run the notebooks
but are not committed to the repository.

### Exp 1: UCI Default

| Model | MSE | AUC | KS |
|---|---:|---:|---:|
| Scorecard (WoE LR, 10 bands) | 0.1383 | 0.7627 | 0.4077 |
| GBM (continuous latent) | 0.1359 | 0.7721 | 0.4190 |
| QPM (quantized PD) | 0.1366 | 0.7685 | 0.4126 |

- QPM bins: 14
- Churn reduction (scenario B vs A): 4.40x
- Anchored calibration native churn (scenario C): 0.00

### Exp 2: Home Credit

| Model | MSE | AUC | KS |
|---|---:|---:|---:|
| GBM (continuous latent) | 0.0686 | 0.7498 | 0.3781 |
| QPM (quantized PD) | 0.0686 | 0.7492 | 0.3745 |

- QPM bins: 22
- Native churn reduction vs GBM retrain: 4.84x

### Exp 3: Framingham CHD

- GBM AUC: 0.6897 (95% BCa CI: 0.6401-0.7348)
- QPM quantized AUC: 0.6866 (95% BCa CI: 0.6369-0.7312)
- DeLong test p-value: 0.4675 (no significant AUC difference)
- QPM bins: 9
- PD range across bins: 3.7% to 53.0%

These current runs preserve the core paper conclusions: QPM maintains most discriminative power while producing a stable, interpretable discrete risk ladder and materially improving update stability.

## Citation

```bibtex
@article{ortizgomez2025qpm,
  title   = {Quantized Prediction Models: Output-Space Regularization
             for Stable and Interpretable Regression},
  author  = {Ortiz-G{\'o}mez, Carlos R.},
  year    = {2025},
  month   = {December},
  doi     = {10.5281/zenodo.17785949},
  note    = {Independent Researcher}
}
```

## License

MIT
