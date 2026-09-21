"""
SNAPSHOT NOTICE: this file is a snapshot of experts/fiorino/model.py from
the nanites monorepo (for the standalone fiorino-tab package). The monorepo
copy is authoritative; re-sync on release. Model code is MIT (lab).
Checkpoints carry their own arch record (see ckpt_arch); loaders below
adapt automatically.

FiorinoNano — A small tabular foundation model for in-context learning.

Architecture inspired by (NOT a copy of):
  - modded-nanoTabPFN (borawhocodess, Apache-2.0) — dual-axis attention,
    thinking rows, feature grouping, residual decay, Muon optimizer
  - EXAONE-Tabular (LG AI Research, 2026) — native missingness channel,
    feature-summary tokens, quantile-bucket regression head
  - TabDPT (Layer 6 AI, 2025) — masked-column SSL training objective
  - TabPFN-3 (Prior Labs, 2026) — staged feature processing ideas

Key differences from modded-nanoTabPFN:
  1. Missingness channel in FeatureEncoder (EXAONE-inspired)
  2. Learnable feature-summary tokens along the feature axis
  3. Quantile-bucket head for regression targets
  4. Multi-target SSL training (column masking, TabDPT-inspired)

Reusable components attributed to modded-nanoTabPFN (Apache-2.0):
  Muon optimizer, LowerPrecisionRMSNorm, zeropower backends.
"""
from __future__ import annotations
import collections

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Reused from modded-nanoTabPFN (borawhocodess), Apache-2.0.
# Adapted, not verbatim copied.
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


@torch.compile
def zeropower_via_newtonschulz5_compiled(G, steps=10, eps=1e-7):
    return zeropower_via_newtonschulz5(G, steps, eps)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix parameters (2D tensors).
    Adapted from modded-nanoTabPFN (Apache-2.0).
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 weight_decay=0.0, backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5_compiled(g, steps=group["backend_steps"])
                scale = max(g.size(0), g.size(1)) ** 0.5
                p.data.add_(g, alpha=-lr * scale)
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - lr * group["weight_decay"])


class LowerPrecisionRMSNorm(nn.RMSNorm):
    """RMSNorm that runs in float32 even under bf16 autocast.
    Adapted from modded-nanoTabPFN / TabPFN (Apache-2.0).
    """
    def forward(self, x):
        if x.dtype in (torch.float16, torch.bfloat16):
            with torch.amp.autocast("cuda", enabled=False):
                return super().forward(x)
        return super().forward(x)


# ---------------------------------------------------------------------------
# FiorinoNano model — our unique architecture
# ---------------------------------------------------------------------------

class ThinkingRows(nn.Module):
    """Learnable task-level scratchpad rows, prepended to the sequence.
    These serve as a shared computation space for dual-axis attention,
    analogous to EXAONE's item-summary tokens but at nano scale.
    """
    def __init__(self, num_thinking_rows: int, e: int):
        super().__init__()
        self.num_thinking_rows = num_thinking_rows
        self.row_tokens = nn.Parameter(torch.empty(num_thinking_rows, e))
        nn.init.normal_(self.row_tokens)

    def forward(self, x, sep):
        b, r, c, e = x.shape
        thinking = self.row_tokens.unsqueeze(0).unsqueeze(2).expand(b, -1, c, -1)
        x = torch.cat([thinking, x], dim=1)
        return x, sep + self.num_thinking_rows


class FeatureSummaryTokens(nn.Module):
    """Learnable per-feature summary tokens (EXAONE-inspired).

    Appends a small number of learnable tokens along the feature axis.
    These participate in cross-column attention, capturing global column
    statistics. They are excluded from the decoder (which only reads real
    feature embeddings).
    """
    def __init__(self, num_summaries: int, e: int):
        super().__init__()
        self.num_summaries = num_summaries
        self.tokens = nn.Parameter(torch.empty(num_summaries, e))
        nn.init.normal_(self.tokens)

    def forward(self, x):
        b, r, c, e = x.shape
        summary = self.tokens.unsqueeze(0).unsqueeze(0).expand(b, r, -1, -1)
        return torch.cat([x, summary], dim=2)


class FeatureEncoder(nn.Module):
    """Encodes cell values with cyclic feature grouping + missingness.

    Differences from modded-nanoTabPFN:
      - Extra missingness channel (EXAONE-inspired): an explicit per-cell
        missing flag is passed in and concatenated to the grouped values
        before the linear projection. Missing cells arrive imputed (zero),
        so the model learns both the (imputed) value and that it is missing.
    """
    def __init__(self, e: int, feature_group_size: int = 5, col_emb: bool = False,
                 max_cols: int = 128):
        super().__init__()
        self.feature_group_size = feature_group_size
        self.col_emb = col_emb
        # +1 for missingness channel (EXAONE-inspired)
        self.linear_layer = nn.Linear(feature_group_size + 1, e)
        # learned column-identity embeddings (TabICL-v2/TabPFN-v3/EXAONE-style)
        # give the attention a stable per-column anchor to key on.
        if col_emb:
            self.col_pos = nn.Parameter(torch.zeros(max_cols, e))
            nn.init.normal_(self.col_pos, std=0.02)
        # column-TYPE embeddings (2026-09-18): 0=numeric, 1=categorical,
        # 2=temporal. The encoder was type-blind (date-ranks, gaussianized
        # ids and values all look identical post-encoding); this restores it.
        self.type_emb = nn.Embedding(3, e)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        # PLE-style periodic embeddings (2026-09-18): multi-frequency
        # sin/cos of the standardized cell value, projected and added.
        # Gives the transformer high-frequency sensitivity over smooth
        # continuous surfaces (precise value reading for regression).
        # New params only — old checkpoints load with fresh init.
        self.ple_freqs = [1.0, 2.0, 4.0, 8.0]
        self.ple = nn.Linear(2 * len(self.ple_freqs), e)
        nn.init.normal_(self.ple.weight, std=0.02)
        nn.init.zeros_(self.ple.bias)

    def forward(self, x: torch.Tensor, missing: torch.Tensor, sep: int,
                col_types: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, R, C) z-scored values, cells imputed where missing
        # missing: (B, R, C) 1.0 where originally missing
        x_in = x  # keep the raw cells for the periodic path below
        n_cols = x.shape[-1]
        idxs = torch.arange(n_cols, dtype=torch.long, device=x.device)
        # cyclic feature grouping
        x_grouped = torch.stack(
            [x[:, :, (idxs + (2**i - 1)) % n_cols] for i in range(self.feature_group_size)],
            dim=-1,
        )  # (B, R, C, fgs)
        # normalize using train-split statistics
        mean = x_grouped[:, :sep].mean(dim=1, keepdim=True)
        std = x_grouped[:, :sep].std(dim=1, keepdim=True) + 1e-8
        x_normed = (x_grouped - mean) / std
        x_normed = torch.clip(x_normed, min=-100, max=100)
        # missingness indicator (EXAONE-inspired native handling)
        missing_grouped = torch.stack(
            [missing[:, :, (idxs + (2**i - 1)) % n_cols] for i in range(self.feature_group_size)],
            dim=-1,
        )
        missing_flag = missing_grouped.max(dim=-1, keepdim=True).values  # (B, R, C, 1)
        x_input = torch.cat([x_normed, missing_flag], dim=-1)  # (B, R, C, fgs+1)
        x = self.linear_layer(x_input)  # (B, R, C, E)
        if self.col_emb:
            x = x + self.col_pos[:n_cols].view(1, 1, n_cols, x.shape[-1])
        if col_types is not None:
            # (B, C) long ids -> (B, 1, C, E), broadcast over rows
            x = x + self.type_emb(col_types).unsqueeze(1)
        # Periodic path: standardize per column on train rows, then
        # multi-frequency sin/cos. Bounded by construction; clamped for safety.
        with torch.no_grad():
            _mu = x_in.mean(dim=1, keepdim=True)
            _sd = x_in.std(dim=1, keepdim=True, correction=0) + 1e-8
        zn = ((x_in - _mu) / _sd).clamp(-8, 8).unsqueeze(-1)  # (B, R, C, 1)
        fr = torch.as_tensor(self.ple_freqs, dtype=zn.dtype,
                             device=zn.device).view(1, 1, 1, -1)
        ang = zn * fr
        ple_in = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        x = x + self.ple(ple_in.to(x.dtype))
        return x


class TargetEncoder(nn.Module):
    """Fixed per-class/bucket embedding (NanoTabICL ClassEmbedding-style).

    Labels are treated as ids into a FIXED table (max_outputs classes/buckets,
    plus one reserved mask token), so a given class has identical input
    geometry across every task regardless of class balance or proportion.

    This replaced the previous CHPT-style z-scored value encoding, which had
    two crippling defects for in-context learning:
      (a) for a balanced binary task it mapped class 0 to EXACTLY the masked
          sentinel (-1.0), making "label=0" indistinguishable from "no label";
      (b) across tasks, the same class got different normalized values (mean/
          std differ with class proportions), so the model could never learn a
          consistent "in-context class -> readout channel" rule — killing
          cross-task ICL transfer while leaving single-task overfit intact.
    """
    def __init__(self, e: int, max_outputs: int):
        super().__init__()
        self.max_outputs = max_outputs
        self.embed = nn.Embedding(max_outputs + 1, e)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, y: torch.Tensor, num_rows: int, sep: int) -> torch.Tensor:
        if y.dim() > 2:
            y = y.squeeze(-1)  # some callers pass (B, R, 1)
        vis = y >= 0                                # visible labels, not mask/pad
        mask_id = torch.full_like(y, self.max_outputs, dtype=torch.long)
        ids = torch.where(vis, y.round().long().clamp(0, self.max_outputs - 1), mask_id)
        emb = self.embed(ids)  # (B, R, E)
        pad = torch.zeros(y.size(0), num_rows - y.shape[1], self.embed.embedding_dim,
                          device=y.device)
        emb = torch.cat([emb, pad], dim=1) if pad.shape[1] > 0 else emb
        return emb.unsqueeze(2)  # (B, R', 1, E)


class TransformerEncoderLayer(nn.Module):
    """Dual-axis attention layer: feature-attn then datapoint-attn + MLP.
    Pre-norm (RMSNorm) with residual connections.
    Datapoint attention uses train-attend-to-train / test-attend-to-train
    cross-attention (PFN-style, no query-query attention).

    Differences from modded-nanoTabPFN: structural redesign for clarity,
    same dual-axis principle. Uses explicit QKV decomposition with SDPA.
    """
    def __init__(self, a: int, e: int, h: int, eps: float = 1e-5):
        super().__init__()
        self.num_heads = a
        self.head_dim = e // a
        assert e % a == 0

        self.qkv_features = nn.Linear(e, 3 * e)
        self.qkv_datapoints = nn.Linear(e, 3 * e)

        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, e)

        self.norm1 = LowerPrecisionRMSNorm(e, eps=eps)
        self.norm2 = LowerPrecisionRMSNorm(e, eps=eps)
        self.norm3 = LowerPrecisionRMSNorm(e, eps=eps)

    def forward(self, src: torch.Tensor, sep: int, causal: bool = False,
                n_think: int = 0) -> torch.Tensor:
        b, r, c, e = src.shape

        # --- Feature-axis attention (within each row, across features) ---
        x = src.reshape(b * r, c, e)
        res = x
        x = self.norm1(x)
        qkv = self.qkv_features(x).reshape(b * r, c, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b * r, c, e)
        src = (res + x).reshape(b, r, c, e)

        # --- Datapoint-axis attention (within each feature, across rows) ---
        # Train rows attend to train; test rows attend to train (cross-attn).
        x = src.transpose(1, 2).reshape(b * c, r, e)
        res = x
        x = self.norm2(x)
        qkv = self.qkv_datapoints(x).reshape(b * c, r, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_left, q_right = q.split([sep, r - sep], dim=2)
        if causal:
            # Classic PFN inductive setup: causal attention over the row axis;
            # row i can attend every row before it (visible or masked).  The
            # decoder then reads ALL rows' feature columns; the caller selects
            # the masked (query) positions.
            x = F.scaled_dot_product_attention(
                torch.cat([q_left, q_right], dim=2), k, v, is_causal=True
            )
        else:
            # P1-3/CAST (2026-09-18): thinking rows are task-constant tokens.
            # Train queries must not attend them (contamination); test
            # queries keep them (task summary). n_think=0 -> old behavior.
            # STATUS 2026-09-19: warm-start run (run_p13) DROPPED cls
            # 0.818->0.791 -- harmful under warm-start (confounded with
            # corpus-heavy mix; needs a from-scratch test to judge fairly).
            # Default OFF until then; enable via mask_thinking=True.
            if getattr(self, "mask_thinking", False):
                k_all, v_all = k[:, :, :sep, :], v[:, :, :sep, :]
                k_tr, v_tr = k[:, :, n_think:sep, :], v[:, :, n_think:sep, :]
                x_left = F.scaled_dot_product_attention(q_left, k_tr, v_tr)
                x_right = F.scaled_dot_product_attention(q_right, k_all, v_all)
            else:
                k_train, v_train = k[:, :, :sep, :], v[:, :, :sep, :]
                x_left = F.scaled_dot_product_attention(q_left, k_train, v_train)
                x_right = F.scaled_dot_product_attention(q_right, k_train, v_train)
            x = torch.cat([x_left, x_right], dim=2)
        x = x.transpose(1, 2).reshape(b * c, r, e)
        src = (res + x).reshape(b, c, r, e).transpose(2, 1)

        # --- MLP ---
        x = self.norm3(src)
        x = self.linear2(F.gelu(self.linear1(x)))
        src = src + x
        return src


class TransformerEncoderStack(nn.Module):
    """Stack of dual-axis attention layers with exponential residual decay."""
    def __init__(self, l: int, a: int, e: int, h: int, residual_decay: float = 0.95):
        super().__init__()
        self.residual_decay = residual_decay
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(a, e, h) for _ in range(l)]
        )

    def forward(self, x: torch.Tensor, sep: int, causal: bool = False,
                n_think: int = 0) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = x * (self.residual_decay ** i)
            x = block(x, sep=sep, causal=causal, n_think=n_think)
        return x


class Decoder(nn.Module):
    """MLP decoder: embedding → hidden → output logits.

    In the default (mean) mode the C feature embeddings are mean-pooled.
    In 'attn' mode a learned query token attends over the C features
    (softmax selection), so a single informative column is not diluted by
    noise columns before classification.
    """
    def __init__(self, e: int, h: int, o: int, mode: str = "mean"):
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.query = nn.Parameter(torch.empty(1, 1, e))
            nn.init.normal_(self.query, std=0.02)
            self.scale = (e ** -0.5)
        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, o)
        # CHPT-style target conditioning (2026-09-18): train-split (mean,
        # std) of the target (log1p-raw for reg, ids for cls) projected and
        # added pre-MLP, putting every task on one scale. ZERO-INIT so a
        # warm start begins as a no-op and learns the conditioning in.
        self.cond = nn.Linear(2, e)
        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)

    def forward(self, x: torch.Tensor, y_stats: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, E) mean-mode; (B, R, C, E) attn-mode
        if self.mode == "attn" and x.ndim == 4:
            # Explicit per-row broadcast (a (B,1,E) query relies on implicit
            # batch broadcasting that torch.compile's fake tensors reject).
            q = self.query.expand(x.size(0), x.size(1), -1).unsqueeze(-2)  # (B, R, 1, E)
            scores = (q @ x.transpose(-2, -1) * self.scale)  # (B, R, 1, C)
            w = F.softmax(scores, dim=-1)
            x = (w @ x).squeeze(-2)  # (B, R, E)
        elif x.ndim == 4:
            x = x.mean(dim=2)
        if y_stats is not None:
            # (B, 2) or (2,) -> broadcast over rows; supports (B,R,E)/(B,E)
            c = self.cond(y_stats.reshape(-1, 2).to(x.dtype))
            x = x + (c.unsqueeze(1) if x.ndim == 3 else c)
        return self.linear2(F.gelu(self.linear1(x)))


class FiorinoNanoModel(nn.Module):
    """Tabular foundation model for in-context learning on ledger data.

    Unique features (vs modded-nanoTabPFN):
      - Native missingness channel in feature encoding
      - Feature-summary tokens (EXAONE-inspired)
      - Quantile-bucket regression head (TabPFN-3/EXAONE-inspired)

    Args:
        l: number of transformer layers
        a: number of attention heads
        e: embedding size
        h: MLP hidden size
        max_outputs: max number of output classes (cls) or buckets (reg)
        residual_decay: exponential decay of residual stream per layer
        thinking_rows: number of learnable scratchpad rows
        feature_group_size: cyclic feature grouping size
        feature_summaries: number of learnable feature-summary tokens
    """

    def __init__(
        self,
        l: int = 5,
        a: int = 4,
        e: int = 256,
        h: int = 768,
        max_outputs: int = 64,
        residual_decay: float = 0.95,
        thinking_rows: int = 24,
        feature_group_size: int = 5,
        feature_summaries: int = 4,
        causal_layout: bool = False,
        pool_mode: str = "mean",
        col_emb: bool = False,
        reg_head_type: str = "bucket",
        mask_thinking: bool = False,
    ):
        super().__init__()
        self.l = l
        self.a = a
        self.e = e
        self.h = h
        self.max_outputs = max_outputs
        self.feature_group_size = feature_group_size
        self.causal_layout = causal_layout
        self.n_thinking_rows = thinking_rows
        self.mask_thinking = mask_thinking

        self.feature_encoder = FeatureEncoder(e, feature_group_size=feature_group_size,
                                          col_emb=col_emb)
        self.target_encoder = TargetEncoder(e, max_outputs)
        self.feature_summary = FeatureSummaryTokens(feature_summaries, e)
        self.n_feature_summary = feature_summaries
        self.thinking_rows = ThinkingRows(thinking_rows, e)
        self.transformer_stack = TransformerEncoderStack(l, a, e, h, residual_decay)
        self.decoder = Decoder(e, h, max_outputs, mode=pool_mode)
        # Separate regression head (TabDPT precedent, 2026-09-18): the
        # shared readout forces cls channels and reg buckets to fight over
        # the same logits. Old checkpoints load without it (fresh init).
        self.reg_decoder = Decoder(e, h, max_outputs, mode=pool_mode)
        # MDN head (2026-09-18): mixture-density alternative to buckets —
        # K Gaussians in LOG-target space (multimodal + uncertainty with
        # continuous support, the tail case buckets punt). Fresh params;
        # used only when reg_head_type == "mdn".
        self.mdn_k = 8
        self.mdn_head = nn.Linear(e, 3 * self.mdn_k)
        nn.init.normal_(self.mdn_head.weight, std=0.02)
        nn.init.zeros_(self.mdn_head.bias)
        # MDN conditioning (same CHPT role as Decoder.cond; zero-init).
        self.mdn_cond = nn.Linear(2, e)
        nn.init.zeros_(self.mdn_cond.weight)
        nn.init.zeros_(self.mdn_cond.bias)
        # Standardized continuous head (2026-09-18): predicts
        # z = (log1p(y)-mu)/sigma with MSE + pinball in z-space, decode
        # inverts via train stats. The train-side normalized-space bet
        # (decode-only variants all washed). Fresh params.
        self.zhead = nn.Linear(e, 1)
        # Direct quantile head (2026-09-18, EXAONE-style): K quantiles in
        # LOG-target space, pinball loss, median point estimate. Buckets
        # quantize-then-decode; this predicts the distribution directly.
        # 2026-09-21 (quant stack, Nori-style): grid covers the tails
        # (0.05/0.95); loss weights levels linearly outward
        # (w=1+tail_w*2|t-0.5|); decode is the analytical quantile-function
        # mean with exponential tail extrapolation. K=9 unchanged so old
        # checkpoints keep their qhead shapes.
        self.quant_levels = (0.05, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75,
                             0.9, 0.95)
        self.qhead = nn.Linear(e, len(self.quant_levels))
        self.reg_head_type = reg_head_type

    def _forward(self, x_src: torch.Tensor, y_src: torch.Tensor, sep: int,
                 missing: torch.Tensor | None = None,
                 col_types: torch.Tensor | None = None,
                 y_stats: torch.Tensor | None = None,
                 is_reg: bool = False):
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)

        C = x_src.shape[2]  # number of feature columns (before summaries)

        if missing is None:
            missing = torch.zeros_like(x_src)

        # encode features with missingness (EXAONE-inspired)
        x_emb = self.feature_encoder(x_src, missing, sep, col_types)  # (B, R, C, E)
        # append feature-summary tokens along feature axis
        x_emb = self.feature_summary(x_emb)  # (B, R, C+fs, E)

        num_rows = x_emb.shape[1]
        y_emb = self.target_encoder(y_src, num_rows, sep)  # (B, R, 1, E)
        src = torch.cat([x_emb, y_emb], dim=2)  # (B, R, C+fs+1, E)

        # prepend thinking rows (task-level scratchpad)
        src, sep = self.thinking_rows(src, sep)

        # dual-axis transformer
        output = self.transformer_stack(src, sep, causal=self.causal_layout,
                                        n_think=0 if self.causal_layout
                                        else self.n_thinking_rows)

        # decode: read feature columns (exclude summaries + target).  Target is
        # always last column, feature summaries are C..C+fs.
        if self.causal_layout:
            tr = self.n_thinking_rows
            features = output[:, tr:tr + x_src.shape[1], :C, :]  # (B, R, C, E): all rows candidates
            # the caller selects the masked (query) row positions
        else:
            features = output[:, sep:, :C, :]  # (B, n_test, C, E)
        if is_reg and self.reg_head_type not in (
                "bucket", "mdn", "zhead", "quant"):
            raise RuntimeError(
                "classification-only release (reg_head_type="
                f"{self.reg_head_type!r}): regression ships when it beats "
                "RandomForest on our TabArena-13 benchmark")
        if is_reg and self.reg_head_type == "mdn":
            pooled = features.mean(dim=2) if features.ndim == 4 else features
            if y_stats is not None:
                c = self.mdn_cond(y_stats.reshape(-1, 2).to(pooled.dtype))
                pooled = pooled + (c.unsqueeze(1) if pooled.ndim == 3 else c)
            return self.mdn_head(pooled)
        if is_reg and self.reg_head_type == "zhead":
            pooled = features.mean(dim=2) if features.ndim == 4 else features
            return self.zhead(pooled)
        if is_reg and self.reg_head_type == "quant":
            pooled = features.mean(dim=2) if features.ndim == 4 else features
            return self._ordered_quantiles(self.qhead(pooled))
        logits = (self.reg_decoder if is_reg else self.decoder)(
            features, y_stats)  # pooling handled inside Decoder
        return logits

    def _ordered_quantiles(self, raw: torch.Tensor) -> torch.Tensor:
        """Reparameterize K free outputs as ORDERED quantiles: median +
        softplus spreads (guaranteed monotone: the failure mode of the
        first quantile attempt was crossing quantiles -> garbage median).
        Median at argmin|tau-0.5|; works for any grid with K>=3."""
        taus = list(self.quant_levels)
        m = int(min(range(len(taus)), key=lambda i: abs(taus[i] - 0.5)))
        med = raw[..., m:m + 1]
        low_gaps = F.softplus(raw[..., :m])
        high_gaps = F.softplus(raw[..., m + 1:])
        q_low = med - torch.cumsum(low_gaps.flip(-1), dim=-1).flip(-1)
        q_high = med + torch.cumsum(high_gaps, dim=-1)
        return torch.cat([q_low, med, q_high], dim=-1)

    def forward(self, x_src: torch.Tensor, y_src: torch.Tensor, sep: int):
        return self._forward(x_src, y_src, sep)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def mdn_split(params: torch.Tensor, k: int):
    """Split (..., 3K) mixture params into (pi_logits, mu, logsigma)."""
    pi, mu, ls = params[..., :k], params[..., k:2 * k], params[..., 2 * k:]
    return pi, mu, ls


def mdn_nll(params: torch.Tensor, y_log: torch.Tensor, k: int) -> torch.Tensor:
    """Mean NLL of log-targets under the mixture (stable logsumexp)."""
    pi, mu, ls = mdn_split(params, k)
    logp = F.log_softmax(pi, dim=-1)
    z = (y_log.unsqueeze(-1) - mu) / (ls.exp() + 1e-8)
    comp = -0.5 * z ** 2 - ls - 0.5 * np.log(2 * np.pi)
    return (-torch.logsumexp(logp + comp, dim=-1)).mean()


def mdn_mean(params: torch.Tensor, k: int) -> torch.Tensor:
    """Expected value (in the modeled, i.e. LOG, space)."""
    pi, mu, _ = mdn_split(params, k)
    return (F.softmax(pi, dim=-1) * mu).sum(-1)


def mdn_reg_step(params: torch.Tensor, y_raw, k: int,
                 device=None) -> tuple:
    """One MDN regression step: NLL + decoded raw values + diagnostics.

    params: (..., 3K) mixture params in LOG-target space; y_raw: raw
    targets (any shape, NaN-tolerant). Returns
    (nll, pred_raw_np, ent_pi, std_mu). Shared by train/val/benchmark
    so the MDN path stays identical everywhere.
    """
    dev = device if device is not None else params.device
    flat = params.reshape(-1, 3 * k)
    yt = torch.as_tensor(np.asarray(y_raw, dtype=np.float32),
                         device=dev).reshape(-1)
    m = torch.isfinite(yt)
    if bool(m.any()):
        nll = mdn_nll(flat[m], yt[m].clamp(min=0.0).log1p(), k)
    else:
        nll = torch.zeros((), device=dev)
    with torch.no_grad():
        mu_log = mdn_mean(flat, k)
        # expm1 has no bf16 CUDA kernel: decode in float32.
        pred = torch.expm1(mu_log.float().clamp(max=20.0)).detach().cpu().numpy()
        pi, mu, _ = mdn_split(flat.detach(), k)
        pw = F.softmax(pi, dim=-1)
        ent = float((-(pw * pw.clamp(min=1e-12).log()).sum(-1)).mean())
        std = float(mu.std())
    return nll, np.asarray(pred, dtype=float), ent, std


def zhead_loss_and_decode(pred_z: torch.Tensor, y_raw, mean: float,
                          std: float) -> tuple:
    """Standardized continuous head: 0.5*MSE + 0.5*pinball(avg tau) in
    z-space, decode inverts to raw. pred_z: (...,) standardized preds;
    y_raw: raw targets (NaN-tolerant); mean/std: log-space train stats
    (std floored at 0.5: near-constant train splits would otherwise
    divide by ~1e-8 and detonate the loss). Returns (loss, pred_raw_np).
    Shared by train/val/benchmark/eval."""
    dev = pred_z.device
    std = max(float(std), 0.5)
    yt = torch.as_tensor(np.asarray(y_raw, dtype=np.float32),
                         device=dev).reshape(-1)
    m = torch.isfinite(yt)
    pz = pred_z.reshape(-1)
    if bool(m.any()):
        yl = (torch.log1p(torch.clamp(yt[m], min=0.0)) - mean) / (std + 1e-8)
        d = yl - pz[m]
        mse = (d ** 2).mean()
        pin = sum(torch.maximum(t * d, (t - 1.0) * d)
                  for t in (0.1, 0.5, 0.9)) / 3.0
        pin = pin.mean()
        loss = 0.5 * mse + 0.5 * pin
    else:
        loss = torch.zeros((), device=dev)
    with torch.no_grad():
        raw = torch.expm1((pz.float().clamp(-8, 8) * std + mean))
        pred_np = raw.detach().cpu().numpy()
    return loss, np.asarray(pred_np, dtype=float)


def quantile_dist_mean(q: torch.Tensor, taus) -> torch.Tensor:
    """Analytical mean of the piecewise-linear quantile function with
    exponential tail extrapolation (Nori-style, clean-room).

    q: (..., K) monotone quantile preds; taus: K levels in (0,1).
    Body = trapezoidal integral; tails fit q(t)=q0+/-l*ln from the outer
    2 quantiles each side (l clamped to [0,3] — z-space slopes; the
    caller winsorizes to train support after inversion, so this can only
    move the mean sensibly toward the tail, never detonate it).
    Float32 compute (bf16 would quantize the narrow tau intervals)."""
    q = q.float()
    tau = torch.as_tensor(np.asarray(taus, dtype=np.float64),
                          device=q.device, dtype=torch.float32)
    K = q.shape[-1]
    if K < 3:
        return q.squeeze(-1) if K == 1 else 0.5 * (q[..., 0] + q[..., -1])
    dt = tau[1:] - tau[:-1]
    mid = (0.5 * (q[..., :-1] + q[..., 1:]) * dt).sum(-1)
    # Left tail: q(t) = q0 + lL*ln(t/t0) over [0, t0] -> area t0*(q0-lL).
    lt = torch.log(tau[:2].clamp(min=1e-12))
    lL = ((q[..., 1] - q[..., 0]) /
          (lt[1] - lt[0]).clamp(min=1e-12)).clamp(0.0, 3.0)
    left = tau[0] * (q[..., 0] - lL)
    # Right tail: q(t) = qK + lR*ln((1-tK)/(1-t)) -> area rem*(qK+lR).
    rt = torch.log((1.0 - tau[-2:]).clamp(min=1e-12))
    lR = ((q[..., -1] - q[..., -2]) /
          (rt[0] - rt[1]).clamp(min=1e-12)).clamp(0.0, 3.0)
    right = (1.0 - tau[-1]) * (q[..., -1] + lR)
    out = left + mid + right
    simp = (tau[0] * q[..., 0] + mid +
            (1.0 - tau[-1]) * q[..., -1])
    return torch.where(torch.isfinite(out), out, simp)


def quant_loss_and_decode(pred_q: torch.Tensor, y_raw, taus,
                          mean: float, std: float,
                          tail_w: float = 0.0) -> tuple:
    """Direct quantile head (EXAONE-style) as a full stack (Nori-style):
    pinball over taus in LOG-standardized space with levels weighted
    linearly outward (w=1+tail_w*2|t-0.5| — extreme quantiles set the
    tails, so they pay more), monotone by construction (ordered
    reparameterization), point decode = analytical quantile-function mean
    with exponential tail extrapolation, inverted to raw.
    pred_q: (..., K); taus: sequence; y_raw: raw targets (NaN-tolerant);
    mean/std: log-space train stats (std floored at 0.5, see zhead).
    Returns (loss, pred_raw_np)."""
    dev = pred_q.device
    std = max(float(std), 0.5)
    K = len(taus)
    yt = torch.as_tensor(np.asarray(y_raw, dtype=np.float32),
                         device=dev).reshape(-1)
    m = torch.isfinite(yt)
    pq = pred_q.reshape(-1, K)
    t = torch.as_tensor(np.asarray(taus, dtype=np.float32),
                        device=dev).view(1, K)
    if bool(m.any()):
        yl = (torch.log1p(torch.clamp(yt[m], min=0.0)) - mean) / (std + 1e-8)
        d = yl.unsqueeze(-1) - pq[m]
        pin = torch.maximum(t * d, (t - 1.0) * d)
        if tail_w > 0:
            w = 1.0 + float(tail_w) * (2.0 * (t - 0.5).abs())
            pin = pin * w
        loss = pin.mean()
    else:
        loss = torch.zeros((), device=dev)
    with torch.no_grad():
        mz = quantile_dist_mean(pq.float().detach(), taus)
        raw = torch.expm1((mz * std + mean).clamp(-8, 20.0))
        pred_np = raw.detach().cpu().numpy()
    return loss, np.asarray(pred_np, dtype=float)


def winsorize_to_support(pred_np, train_raw):
    """Clip predictions to the observed train support. Buckets get this
    guarantee structurally (edge centers <= train max); continuous MDN
    means can overshoot by 100x via expm1, so enforce the same bound
    explicitly. Deployment-legit: never predict outside observed support
    by orders of magnitude."""
    pred_np = np.asarray(pred_np, dtype=float)
    tr = np.asarray(train_raw, dtype=float)
    tr = tr[np.isfinite(tr)]
    if len(tr) == 0:
        return pred_np
    return np.clip(pred_np, tr.min(), tr.max())


def ckpt_arch(ckpt_or_sd) -> dict:
    """Architecture flags for a checkpoint. Prefers the stored arch record
    (written by train_loop); falls back to key-presence heuristics for old
    checkpoints. NOTE: key-presence alone misfires now that every arch
    carries every head (fresh-init keys present) — hence the record.
    (Also: detection must run on prefix-STRIPPED keys; raw _orig_mod.
    keys silently disabled col_emb in all early evals.)"""
    if isinstance(ckpt_or_sd, dict) and "model_state_dict" in ckpt_or_sd:
        rec = {k: ckpt_or_sd.get(k) for k in
               ("reg_head_type", "pool_mode", "col_emb", "mask_thinking")}
        if rec["reg_head_type"] in ("bucket", "mdn", "zhead", "quant",
                                    "none"):
            return {"col_emb": bool(ckpt_or_sd.get(
                        "col_emb",
                        _keys_arch(ckpt_or_sd["model_state_dict"])["col_emb"])),
                    "pool_mode": ckpt_or_sd.get(
                        "pool_mode",
                        _keys_arch(ckpt_or_sd["model_state_dict"])["pool_mode"]),
                    "reg_head_type": rec["reg_head_type"],
                    "mask_thinking": bool(ckpt_or_sd.get("mask_thinking", False))}
        sd = ckpt_or_sd["model_state_dict"]
    else:
        sd = ckpt_or_sd
    return _keys_arch(sd)


def _keys_arch(sd) -> dict:
    keys = [k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.")
            else k for k in sd]
    return {
        "col_emb": any(k.startswith("feature_encoder.col_pos") for k in keys),
        "pool_mode": "attn" if any("decoder.query" in k for k in keys) else "mean",
        # Heuristic only: post-head-unification every ckpt carries all
        # head keys; prefer the stored record (see ckpt_arch).
        "reg_head_type": ("mdn" if any("mdn_head" in k for k in keys)
                          else "bucket"),
    }


def load_ckpt_compat(model: torch.nn.Module, ckpt_path, map_location="cpu"):
    """Load a checkpoint tolerantly across architecture additions.

    Strips torch.compile (_orig_mod.) prefixes and loads with strict=False
    so older checkpoints (pre column-type embeddings / CHPT conditioning)
    still load — new params keep their init (cond is zero-init: no-op).
    Prints what was missing/unexpected instead of failing.
    """
    ck = torch.load(ckpt_path, map_location=map_location)
    sd = ck["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    except RuntimeError:
        # Shape mismatch (e.g. col_pos grew 64->128): copy key-by-key what
        # fits, keep fresh init for the rest.
        missing, unexpected = [], []
        own = model.state_dict()
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
            elif k in own:
                # Prefix-compatible shapes (e.g. col_pos 64->128 rows):
                # copy the overlapping slice, keep fresh init for the rest.
                try:
                    sl = tuple(slice(0, min(a, b))
                               for a, b in zip(own[k].shape, v.shape))
                    own[k][sl].copy_(v[sl])
                    missing.append(f"{k} (shape {tuple(v.shape)}->"
                                   f"{tuple(own[k].shape)}, prefix kept)")
                except Exception:
                    missing.append(f"{k} (shape {tuple(v.shape)}->"
                                   f"{tuple(own[k].shape)}, fresh init)")
            else:
                unexpected.append(k)
        for k in own:
            if k not in sd:
                missing.append(k)
    if missing:
        print(f"  [ckpt-compat] randomly-init (not in ckpt): {missing}")
    if unexpected:
        print(f"  [ckpt-compat] ignored (not in model): {unexpected}")
    return ck
