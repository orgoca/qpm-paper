# Quantized Prediction Models (QPMs)

**Output-Space Regularization for Stable and Interpretable Regression**

Carlos R. Ortiz-Gómez — *Independent Researcher*, December 2025
DOI: [10.5281/zenodo.17785949](https://doi.org/10.5281/zenodo.17785949)

---

## Overview

This repository contains the complete, self-contained code to reproduce every
experiment and claim in the paper. The implementation is written from scratch
to be clean and readable — no proprietary dependencies.

**QPMs** constrain model output to a finite, ordered set of prediction levels
(a "ladder") by learning a supervised quantizer on top of any continuous
latent scorer. The quantizer is **fully model-agnostic**: it works with
XGBoost, LightGBM, scikit-learn GBMs, neural networks, or any scorer that
produces a 1-D array of continuous predictions.

```python
from qpm import QPMQuantizer

# Train any backbone first
F_train = xgb_model.predict(X_train)   # or lgb, sklearn, torch, ...
F_test  = xgb_model.predict(X_test)

# Fit the output-space quantizer
qpm = QPMQuantizer(focus_pd=(0.05, 0.20), n_low_max=6, n_mid_max=6,
                   n_high_max=6, min_band_size=300, pd_cap=0.80)
qpm.fit(F_train, y_train)

# Discrete PD predictions on a monotone ladder
y_hat = qpm.predict(F_test)
```

---

## Installation

```bash
git clone <this-repo>
cd QPM_PAPER
pip install -e .
```

Python ≥ 3.9 required.

---

## Notebooks

Run the notebooks in order. Each is fully self-contained.

| Notebook | Experiment | What it reproduces |
|---|---|---|
| `notebooks/01_uci_default.ipynb` | Exp 1 — UCI Credit Card | Table 1, Table 2, Figure 1, churn analysis |

**Workbook‑to‑Manuscript Sync:** after running the experiments the
artifacts directory contains `report.json` files. You can automatically
update the canonical numbers in `paper.txt` (the manuscript) by executing:

```bash
python generate_notebooks.py --sync-paper
```

This replaces the hard‑coded metrics in the manuscript with the
actual values produced by the notebooks, ensuring the manuscript always
reflects the latest results.  In addition, the generation script now
embeds **canonical assertion checks** directly in each experiment cell
(see comments in the notebooks). When you regenerate a notebook the
code will raise an exception if any of the reported metrics drift from
the current canonical values, making it easy to spot unintended changes.
| `notebooks/02_home_credit.ipynb` | Exp 2 — Home Credit | Table 3, churn (3.4× factor) |
| `notebooks/03_framingham.ipynb`  | Exp 3 — Framingham CHD | AUC CIs, DeLong test (p=0.46), 13-strata ladder |

---

## Data Setup

See [data/README.md](data/README.md) for download instructions.

- **Exp 1** (UCI): downloaded automatically via `ucimlrepo` — no action needed.
- **Exp 2** (Home Credit): requires Kaggle API token.
- **Exp 3** (Framingham): requires Kaggle API token.

---

## Package Structure

```
qpm/
├── quantizer.py     # QPMQuantizer — optimizer-driven supervised binning
├── calibration.py   # AnchoredCalibration + residual stacking
├── metrics.py       # AUC, KS, MSE, bootstrap BCa CIs, DeLong test, churn
└── scorecard.py     # WoE logistic scorecard baseline
```

---

## Key Claims Reproduced

### Experiment 1 — UCI Default (30k accounts, 22.1% default rate)

| Model | MSE | AUC | KS |
|---|---|---|---|
| Scorecard (LR, 10 bands) | 0.1389 | 0.7599 | 0.4018 |
| GBM (continuous) | 0.1359 | 0.7727 | 0.4203 |
| QPM (latent F(x)) | 0.1359 | 0.7773 | 0.4289 |
| QPM (quantized PD) | 0.1359 | 0.7750 | 0.4225 |

- QPM produces a 14-bin bank-style score ladder (Table 2) with monotone PD centers.
- Fixed-edge fine-tuning reduces native band churn by >4× vs independent retraining.
- Anchored Calibration achieves **0.00% rank-order churn** (Proposition 1).

### Experiment 2 — Home Credit (~300k accounts, 8.0% default rate)

| Model | MSE | AUC | KS |
|---|---|---|---|
| GBM (continuous) | 0.0683 | 0.7517 | 0.3784 |
| QPM (quantized PD) | 0.0683 | 0.7487 | 0.3740 |

- QPM fine-tuning reduces churn by **3.4×** vs GBM retrain (0.001067 vs 0.003635).

### Experiment 3 — Framingham CHD (15.2% incidence)

- GBM AUC: 0.683 [0.633–0.729] vs QPM latent AUC: 0.666 [0.616–0.712]
- DeLong test: p = 0.46 (no statistically significant difference)
- 13-strata risk ladder from ~2% to ~55% event rate

---

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

---

## License

MIT
