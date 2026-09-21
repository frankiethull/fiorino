# fiorino

### nano quattrocento tabular fondamento model with synthetic and real-data priors

![Fiorino](media/fiorino.png)

Scikit-learn interface to [Fiorino](https://github.com/frankiethull/fiorino) —
nano quattrocento tabular fondamento model with synthetic and real-data priors.
Same idea as TabICL / TabPFN wrappers: `fit` stores the in-context prompt
(no gradients), `predict` answers queries in one forward pass.

Weights: `Nanite-Labs/nanites-fiorino-tabular`
(`fiorino-classification-alfa` release).

```python
from fiorino_tab import FiorinoClassifier

clf = FiorinoClassifier()  # repo_id="Nanite-Labs/nanites-fiorino-tabular"
clf.fit(X_train, y_train)  # downloads fiorino-classification-alfa once
clf.predict_proba(X_test)
```

## Alfa scope (classification only)

This release ships the classifier. 

## Notes & limits (v0.1)

- `arch/` snapshot: `src/fiorino_tab/_arch.py` is a snapshot of the
  monorepo `experts/fiorino/model.py` (re-synced on release).
- Context caps: 2048 rows (subsampled), ~44 features.
- Preprocessing mirrors training (train-only encodings/stats).
- Tests: `pytest tests/ --checkpoint <local file|HF repo id> --cls-only`
  (add `--device cuda` for GPU).
