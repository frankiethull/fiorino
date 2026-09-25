"""Preprocessing mirroring FiorinoNano training (benchmark.py contract).

fit_preprocessor learns column roles/encodings/stats on TRAIN data only;
transform applies them (unknown categories -> NaN -> missing channel).
Target coding (ordinal ids / log-quantile buckets) lives here too so the
estimators stay thin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BUCKETS = 50
MAX_ROWS = 2048
MAX_COLS = 128


def _col_type(name, series) -> int:
    lname = str(name).lower()
    if any(k in lname for k in ("date", "year", "month", "day", "time")):
        return 2
    if not pd.api.types.is_numeric_dtype(series):
        return 1
    return 0


def fit_preprocessor(X: pd.DataFrame) -> dict:
    X = pd.DataFrame(X)
    if X.shape[1] > MAX_COLS:
        raise ValueError(f"fiorino supports <={MAX_COLS} features, got {X.shape[1]}")
    const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if const:
        X = X.drop(columns=const)
    feat_cols = list(X.columns)
    kinds, mappings = [], {}
    for c in feat_cols:
        if pd.api.types.is_numeric_dtype(X[c]):
            kinds.append(_col_type(c, X[c]))
        else:
            kinds.append(1)
            uniq = sorted(set(v for v in X[c].astype(object) if pd.notna(v)))
            mappings[c] = {v: i for i, v in enumerate(uniq)}
    return {"feat_cols": feat_cols, "kinds": kinds, "mappings": mappings,
            "mean": None, "std": None}  # filled by finalize_preprocessor


def _encode(X: pd.DataFrame, spec: dict) -> np.ndarray:
    cols = []
    for c in spec["feat_cols"]:
        s = X[c] if c in X.columns else pd.Series(np.nan, index=X.index)
        if c in spec["mappings"]:
            mp = spec["mappings"][c]
            cols.append(s.astype(object).map(
                lambda v: mp.get(v, np.nan)).astype(np.float32).to_numpy())
        else:
            cols.append(pd.to_numeric(s, errors="coerce").astype(np.float32).to_numpy())
    return np.column_stack(cols) if cols else np.zeros((len(X), 0), np.float32)


def finalize_preprocessor(Xtr: pd.DataFrame, spec: dict) -> dict:
    """Fit imputation + z-score stats on encoded TRAIN rows (in place)."""
    Xe = _encode(Xtr, spec)
    mm = np.isnan(Xe)
    clean = np.where(mm, 0.0, Xe)
    col_mean = np.where(mm.all(axis=0), 0.0, clean.sum(axis=0) / np.maximum(mm.shape[0] - mm.sum(axis=0), 1))
    spec["mean"] = col_mean.astype(np.float32)
    sd = np.sqrt(np.maximum(((np.where(mm, 0.0, Xe - col_mean) ** 2).sum(axis=0)
                             / max(len(Xe), 1)), 0.0)) + 1e-8
    spec["std"] = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    return spec


def transform(X: pd.DataFrame, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (x z-scored clipped, missing flag)."""
    Xe = _encode(X, spec)
    mm = np.isnan(Xe)
    xi = np.where(mm, np.broadcast_to(spec["mean"], Xe.shape), Xe)
    x = np.clip((xi - spec["mean"]) / spec["std"], -100, 100)
    return x.astype(np.float32), mm.astype(np.float32)


def encode_cls_target(y: pd.Series):
    uniq = sorted(set(v for v in y.astype(object) if pd.notna(v)))
    mp = {v: i for i, v in enumerate(uniq)}
    ids = np.array([mp.get(v, -1) for v in y.astype(object)], dtype=np.float32)
    return ids, uniq, mp


def reg_buckets(train_vals: np.ndarray, raw_values: np.ndarray):
    """Log-quantile buckets (train-relative) + raw-space mean centers."""
    tv = np.asarray(train_vals, dtype=float)
    med = float(np.nanmedian(tv[np.isfinite(tv)])) if np.isfinite(tv).any() else 0.0
    lt = np.log1p(np.where(np.isfinite(tv) & (tv >= 0.0), tv, med))
    rv = np.log1p(np.where(np.isfinite(np.asarray(raw_values, dtype=float))
                           & (np.asarray(raw_values, dtype=float) >= 0.0),
                           np.asarray(raw_values, dtype=float), med))
    edges = np.percentile(lt, np.linspace(0, 100, N_BUCKETS + 1))
    if not np.all(np.diff(edges) > 0):
        return (np.full_like(rv, N_BUCKETS // 2, dtype=np.float32),
                np.full(N_BUCKETS, float(np.median(np.expm1(lt)))))
    edges[0], edges[-1] = -np.inf, np.inf
    y = np.clip(np.digitize(rv, edges[1:-1]), 0, N_BUCKETS - 1).astype(np.float32)
    btr = np.digitize(lt, edges[1:-1]).clip(0, N_BUCKETS - 1)
    centers = np.full(N_BUCKETS, float(np.median(np.expm1(lt))))
    for b in range(N_BUCKETS):
        sel = btr == b
        if sel.sum():
            centers[b] = float(np.mean(np.expm1(lt[sel])))
    return y, centers


def reg_stats(train_vals: np.ndarray):
    tv = np.asarray(train_vals, dtype=float)
    tv = tv[np.isfinite(tv) & (tv > 0)]
    lv = np.log1p(tv if len(tv) else np.array([1.0]))
    return [float(lv.mean()), float(lv.std() + 1e-8)]
