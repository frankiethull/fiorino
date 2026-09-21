"""Smoke: fit/predict on tiny synthetic data (CPU-safe sizes)."""
import numpy as np
import pandas as pd


def _data(n=60, c=6, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.normal(size=(n, c)),
                     columns=[f"f{i}" for i in range(c)])
    return X, rng


def test_classifier(checkpoint, device):
    from fiorino_tab import FiorinoClassifier
    X, rng = _data()
    y = (X["f0"] + 0.5 * X["f1"] > 0).astype(int)
    clf = FiorinoClassifier(checkpoint=checkpoint, device=device)
    clf.fit(X.iloc[:40], y.iloc[:40])
    p = clf.predict_proba(X.iloc[40:])
    assert p.shape == (20, 2) and np.isfinite(p).all()
    acc = (clf.predict(X.iloc[40:]) == y.iloc[40:].to_numpy()).mean()
    print("cls acc:", round(float(acc), 3))
    return acc


def test_regressor(checkpoint, device, cls_only_artifact):
    if cls_only_artifact:
        import pytest
        pytest.skip("classification-only artifact (regression ships in beta)")
    from fiorino_tab import FiorinoRegressor
    X, rng = _data()
    y = (2 * X["f0"] - X["f2"] + 0.1 * rng.normal(size=len(X))).to_numpy()
    reg = FiorinoRegressor(checkpoint=checkpoint, device=device)
    reg.fit(X.iloc[:40], y[:40])
    pred = reg.predict(X.iloc[40:])
    assert pred.shape == (20,) and np.isfinite(pred).all()
    ss = 1 - np.sum((y[40:] - pred) ** 2) / np.sum((y[40:] - y[40:].mean()) ** 2)
    print("reg r2:", round(float(ss), 3))
    return ss
