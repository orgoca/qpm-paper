# Data Download Instructions

All three datasets used in the paper are publicly available.

---

## Experiment 1 — UCI Default of Credit Card Clients

**No manual download required.**

The notebook uses the `ucimlrepo` package to download the dataset automatically:

```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=350)
```

Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Citation:
> Yeh, I-C., and Lien, C-H. (2009). The comparisons of data mining techniques
> for the predictive accuracy of probability of default of credit card clients.
> *Expert Systems with Applications*, 36(2), 2473–2480.

---

## Experiment 2 — Home Credit Default Risk

**Requires a Kaggle account and API token.**

1. Create a Kaggle account at https://www.kaggle.com
2. Go to **Account → API → Create New Token** — this downloads `kaggle.json`
3. Place `kaggle.json` in `~/.kaggle/kaggle.json` (Linux/Mac) or
   `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
4. Accept the competition terms at:
   https://www.kaggle.com/competitions/home-credit-default-risk

The notebook will then run:

```bash
kaggle competitions download -c home-credit-default-risk -p data/home_credit/
```

You only need `application_train.csv` from the downloaded archive.

Source: https://www.kaggle.com/competitions/home-credit-default-risk/data

---

## Experiment 3 — Framingham Heart Study CHD

**Requires a Kaggle account and API token** (same setup as above).

The notebook will run:

```bash
kaggle datasets download -d shreyjain601/framingham-heart-study -p data/framingham/
```

Source: https://www.kaggle.com/datasets/shreyjain601/framingham-heart-study

Citation:
> Lung National Heart and Blood Institute. Framingham heart study dataset.
> https://www.kaggle.com/datasets/shreyjain601/framingham-heart-study, 2020.
