"""Scikit-learn interface to FiorinoNano (TabICL/TabPFN-style usage).

1. weights download from HuggingFace on first fit,
2. fit(X, y) stores the in-context prompt (no gradients),
3. predict / predict_proba run one forward pass per call.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from ._arch import FiorinoNanoModel, ckpt_arch, load_ckpt_compat
from . import _data as D


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _safetensors_to_pt(path: str) -> str:
    """Wrap a .safetensors state dict as a torch checkpoint file (cached
    next to the download; our loader reads torch checkpoints)."""
    from safetensors.torch import load_file
    out = str(path) + ".as_pt_cache.pt"
    if not Path(out).exists():
        sd = load_file(path, device="cpu")
        torch.save({"model_state_dict": sd}, out)
    return out


class _BaseFiorino(BaseEstimator):
    # Fiorino releases: HF repo Nanite-Labs/nanites-fiorino-tabular;
    # per-head weights (bifronte+), alfa file kept as fallback. Sister
    # GitHub repo "fiorino".
    HF_REPO = "Nanite-Labs/nanites-fiorino-tabular"
    HF_WEIGHTS = ("fiorino-classification-bifronte.pt",
                  "fiorino-classification-bifronte.safetensors",
                  "fiorino-classification-alfa.pt",
                  "fiorino-classification-alfa.safetensors")

    def __init__(self, checkpoint=None,
                 repo_id="Nanite-Labs/nanites-fiorino-tabular",
                 device="auto", seed=0, max_rows=2048, infer_k=1024):
        self.checkpoint = checkpoint
        self.repo_id = repo_id
        self.device = device
        self.seed = seed
        self.max_rows = max_rows
        self.infer_k = infer_k

    def _load_model(self):
        if getattr(self, "_model", None) is not None:
            return self._model
        ckpt = self.checkpoint
        if ckpt is None:
            ckpt = self._download_weights(self.repo_id)
        elif not Path(str(ckpt)).exists():
            # Not a local path -> treat as an HF repo id.
            ckpt = self._download_weights(str(ckpt))
        if str(ckpt).endswith(".safetensors"):
            ckpt = _safetensors_to_pt(str(ckpt))
        arch = ckpt_arch(torch.load(ckpt, map_location="cpu"))
        model = FiorinoNanoModel(
            col_emb=arch["col_emb"], pool_mode=arch["pool_mode"],
            reg_head_type=arch["reg_head_type"],
            mask_thinking=arch.get("mask_thinking", False))
        load_ckpt_compat(model, ckpt)
        self._model = model.to(_device(self.device)).eval()
        return self._model

    @classmethod
    def _download_weights(cls, repo_id: str) -> str:
        """Snapshot-download a weights repo and return the best weights file."""
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo_id=repo_id)
        for name in list(cls.HF_WEIGHTS) + ["model.safetensors", "model.pt"]:
            p = Path(d) / name
            if p.exists():
                return str(p)
        pts = sorted(Path(d).glob("*.pt")) + sorted(Path(d).glob("*.safetensors"))
        if not pts:
            raise FileNotFoundError(f"no weights in {repo_id}")
        return str(pts[0])

    def _prep_fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        if len(X) > self.max_rows:
            idx = rng.choice(len(X), size=self.max_rows, replace=False)
            idx.sort()
            X, y = X.iloc[idx], y.iloc[idx]
        spec = D.fit_preprocessor(X)
        D.finalize_preprocessor(X, spec)
        return X, y, spec

    def _forward(self, Xq: pd.DataFrame):
        model = self._load_model()
        dev = next(model.parameters()).device
        Xc, _ = D.transform(self.X_, self.spec_)
        Xq_, m_q = D.transform(Xq, self.spec_)
        x = np.concatenate([Xc, Xq_], axis=0).astype(np.float32)
        mm = np.concatenate([np.zeros_like(Xc), m_q], axis=0).astype(np.float32)
        n_ctx = len(Xc)
        ct = torch.from_numpy(
            np.asarray(self.spec_["kinds"], dtype=np.int64)).unsqueeze(0).to(dev)
        ys = torch.tensor(self.y_stats_, dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if str(dev) == "cuda" else "cpu",
                                    enabled=str(dev) == "cuda"):
                out = model._forward(
                    torch.from_numpy(x).unsqueeze(0).to(dev),
                    torch.from_numpy(self._y_input()).unsqueeze(0).to(dev),
                    sep=n_ctx, missing=torch.from_numpy(mm).unsqueeze(0).to(dev),
                    col_types=ct, y_stats=ys, is_reg=self._is_reg())
        return out[0, :len(Xq_)].detach().cpu()


class FiorinoClassifier(_BaseFiorino, ClassifierMixin):
    """In-context tabular classifier: fit stores the prompt, predict reads it."""

    def _is_reg(self):
        return False

    def _y_input(self):
        return self.y_ids_

    def fit(self, X, y):
        X, y, spec = self._prep_fit(X, y)
        ids, uniq, mp = D.encode_cls_target(y)
        self.spec_, self.X_ = spec, X
        self.classes_ = np.asarray(uniq, dtype=object)
        self.class_map_ = mp
        self.y_ids_ = ids
        tr = ids[ids >= 0]
        self.y_stats_ = [float(tr.mean()), float(tr.std() + 1e-8)] if len(tr) else [0.0, 1.0]
        self._load_model()
        return self

    def predict_proba(self, X):
        Xq = pd.DataFrame(X).reset_index(drop=True)
        logits = self._forward(Xq).numpy()
        n_out = min(logits.shape[1], len(self.classes_))
        p = torch.softmax(torch.from_numpy(logits[:, :n_out]), dim=-1).numpy()
        return (p / p.sum(axis=1, keepdims=True)).astype(float)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class FiorinoRegressor(_BaseFiorino, RegressorMixin):
    """In-context tabular regressor (bucket head + expected-value decode)."""
    HF_WEIGHTS = ("fiorino-regression-bifronte.pt",
                  "fiorino-regression-bifronte.safetensors")

    def _is_reg(self):
        return True

    def _y_input(self):
        return self.y_bucket_

    def fit(self, X, y):
        X, y = pd.DataFrame(X).reset_index(drop=True), pd.Series(
            np.asarray(y, dtype=float)).reset_index(drop=True)
        if len(X) > self.max_rows:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(len(X), size=self.max_rows, replace=False)
            idx.sort()
            X, y = X.iloc[idx], y.iloc[idx]
        spec = D.fit_preprocessor(X)
        D.finalize_preprocessor(X, spec)
        self.spec_, self.X_ = spec, X
        tv = y.to_numpy(dtype=float)
        tvf = tv[np.isfinite(tv)]
        if len(tvf) < 2:
            raise ValueError("need >=2 finite regression targets to fit")
        ids, centers = D.reg_buckets(tvf, tv)
        self.centers_ = centers
        self.y_bucket_ = ids.astype(np.float32)
        self.y_stats_ = D.reg_stats(tvf)
        model = self._load_model()
        if getattr(model, "reg_head_type", "bucket") != "bucket":
            import warnings
            warnings.warn("FiorinoRegressor v1 decodes bucket logits; this "
                          "checkpoint uses a different reg head — predictions "
                          "may be off. Use a bucket checkpoint.")
        return self

    def predict(self, X):
        Xq = pd.DataFrame(X).reset_index(drop=True)
        logits = self._forward(Xq).numpy()
        K = min(logits.shape[1], len(self.centers_))
        p = torch.softmax(torch.from_numpy(logits[:, :K]), dim=-1).numpy()
        p = p / p.sum(axis=1, keepdims=True)
        return (p @ np.asarray(self.centers_[:K], dtype=float)).astype(float).ravel()
