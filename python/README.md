# fiorino

![Fiorino](media/fiorino.png)

**226,000 tables in the record run (≈400,000 across the lineage) vs
100,000,000+ at the big labs — a nano-lab build.**

Scikit-learn interface to [Fiorino](https://github.com/frankiethull/fiorino) —
nano quattrocento tabular fondamento model with synthetic and real-data priors.
Same idea as TabICL / TabPFN wrappers: `fit` stores the in-context prompt
(no gradients), `predict` answers queries in one forward pass.

Weights: `Nanite-Labs/nanites-fiorino-tabular`
(`fiorino-classification-alfa` release).

```python
from fiorino import FiorinoClassifier, FiorinoRegressor

clf = FiorinoClassifier()  # repo_id="Nanite-Labs/nanites-fiorino-tabular"
clf.fit(X_train, y_train)  # downloads fiorino-classification-bifronte once
clf.predict_proba(X_test)

reg = FiorinoRegressor()   # downloads fiorino-regression-bifronte once
reg.fit(X_train, y_train)
reg.predict(X_test)
```

## Bifronte scope (multihead)

Bifronte ships both heads from one shared trunk: `fiorino-classification-bifronte`
(classifier) and `fiorino-regression-bifronte` (bucket readout). The
regressor trails RandomForest on heavy tails — tracked openly in the
paper; the bar is beating RF on TabArena-13.

## Notes & limits (v0.1)

- `arch/` snapshot: `src/fiorino/_arch.py` is a snapshot of the
  monorepo `experts/fiorino/model.py` (re-synced on release).
- Context caps: 2048 rows (subsampled), ~44 features.
- Preprocessing mirrors training (train-only encodings/stats).
- Tests: `pytest tests/ --checkpoint <local file|HF repo id> --cls-only`
  (add `--device cuda` for GPU).
