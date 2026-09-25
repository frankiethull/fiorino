# fiorino

![Fiorino](media/fiorino.png)

**226,000 tables in the record run (≈400,000 across the lineage) vs
100,000,000+ at the big labs — a nano-lab build.**

Nano quattrocento tabular fondamento model with synthetic and real-data priors —
same idea as TabICL / TabPFN wrappers: `fit` stores the in-context prompt
(no gradients), `predict` answers queries in one forward pass.

This repo is the metarepo: model checkpoints on Hugging Face plus sibling
client packages for Python and R.

## Checkpoints (Hugging Face)

Repo: `Nanite-Labs/nanites-fiorino-tabular`

| File | Head |
| ---- | ---- |
| `fiorino-classification-bifronte.pt` / `.safetensors` | classifier (default) |
| `fiorino-regression-bifronte.pt` / `.safetensors` | regressor (bucket readout) |
| `fiorino-classification-alfa.pt` / `.safetensors` | legacy fallback |

Both bifronte heads share one trunk. The regressor trails RandomForest on
heavy tails — tracked openly in the paper; the bar is beating RF on
TabArena-13. Checkpoints download once on first fit (per-head file
resolution, `.safetensors` preferred).

## Packages

| Dir | Package | Interface |
| --- | ------- | --------- |
| `python/` | `fiorino` (PyPI) | scikit-learn: `FiorinoClassifier`, `FiorinoRegressor` — see `python/README.md` |
| `r/` | `fiorino` (R) | `fiorino_classifier()`, `fiorino_regressor()` via reticulate — see `r/fiorino/README.md` |

```python
from fiorino import FiorinoClassifier  # python/

clf = FiorinoClassifier()  # repo_id="Nanite-Labs/nanites-fiorino-tabular"
clf.fit(X_train, y_train)
clf.predict_proba(X_test)
```

```r
library(fiorino)  # r/fiorino

clf <- fiorino_classifier()
clf <- fio_fit(clf, X_train, y_train)
fio_predict_proba(clf, X_test)
```

## Layout

```
fiorino/
  README.md  media/        <- canonical (mirrored into python/ and r/fiorino/)
  python/                  <- Python package
    src/fiorino/           <- _arch.py snapshot of experts/fiorino/model.py
  r/fiorino/               <- R package (reticulate bridge, no port drift)
```

## Notes & limits (v0.1)

- Arch snapshot `python/src/fiorino/_arch.py` re-syncs from the monorepo on release.
- Context caps: 2048 rows (subsampled), ~44 features.
- Preprocessing mirrors training (train-only encodings/stats).
