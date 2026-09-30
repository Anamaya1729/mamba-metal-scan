"""A small Mamba-style selective state space model.

Everything here is written from scratch: the gated causal depthwise
convolution, the diagonal selective SSM, and the associative prefix scan that
makes the SSM parallel. PyTorch is used for tensors and autograd only
-- no mamba-ssm, no selective_scan_kernel, no pretrained parts.

The recurrence is

    h_t = a_t * h_{t-1} + b_t         y_t = C_t . h_t

with a_t = exp(-dt_t * softplus(A)) in (0, 1). It is computed two ways, and
the tests prove they agree to 1e-5:

  - scan_recurrence: the left-to-right loop. O(T) depth, not parallel, but
    the obvious reference implementation.
  - parallel_scan:   an associative scan (Hillis-Steele up-sweep, O(log T)
                     depth), optionally sliced over time with a state carry
                     to bound the memory of the intermediates.

d_inner is *solved for* so that a Mamba block costs the same as a transformer
block: the two models are compared at an equal parameter budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_metal.scan import selective_scan


@dataclass
class MambaConfig:
    # d_inner=None auto-sizes a block to the transformer's parameter budget.
    vocab_size: int
    d_model: int = 384
    n_layers: int = 10
    d_ff: int = 1536          # only used to match the transformer's budget
    max_seq_len: int = 512
    d_state: int = 16
    d_conv: int = 4
    d_inner: Optional[int] = None
    scan_chunk: int = 0       # 0 = scan the whole sequence in one pass
    # 'auto'/'metal': the fused Metal selective scan on MPS (mamba_metal/scan.py),
    # which keeps the SSM state in GPU registers instead of materialising the
    # (B, T, d_inner, d_state) tensors. 'torch' forces the reference path.
    scan_impl: str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    norm_eps: float = 1e-6


def transformer_block_params(d_model: int, d_ff: int) -> int:
    """What one pre-norm transformer block costs: qkv + proj + mlp + two RMSNorms."""
    return 4 * d_model * d_model + 2 * d_model * d_ff + 2 * d_model


def mamba_block_params(d_model: int, d_inner: int, d_state: int, d_conv: int) -> int:
    """What one MambaBlock costs (mirrors MambaBlock.__init__)."""
    return (2 * d_model * d_inner            # in_proj -> (x, z)
            + d_inner * d_conv + d_inner     # depthwise causal conv + bias
            + d_inner * d_inner + d_inner    # dt_proj + bias
            + 3 * d_inner * d_state          # A, B_proj, C_proj
            + d_inner                        # D skip
            + d_inner * d_model              # out_proj
            + d_model)                       # RMSNorm (ONE per block, not two)


def d_inner_for_budget(target: int, d_model: int, d_state: int, d_conv: int) -> int:
    """The largest multiple of 8 whose Mamba block fits the parameter budget.

    Cost is quadratic in d_inner (the dt projection is d_inner^2), so this
    solves d_inner^2 + c*d_inner + d_model = target and rounds down.
    """
    c = 3 * d_model + d_conv + 3 + 3 * d_state
    root = (math.sqrt(c * c + 4 * (target - d_model)) - c) / 2.0
    return max(8, int(root // 8) * 8)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


# ------------------------------------------------------------------ the scans

def scan_recurrence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h_t = a_t * h_{t-1} + b_t, left to right. Not parallel; the reference."""
    h = torch.zeros_like(b[:, 0])
    out = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def _scan_core(a: torch.Tensor, b: torch.Tensor):
    """Associative scan over dim 1, padded up to a power of two.

    Returns (prefix_a, h): prefix_a[t] = prod_{j<=t} a_j and
    h[t] = sum_{k<=t} b[k] * prod_{j=k+1..t} a[j].
    """
    T = a.shape[1]
    n = 1
    while n < T:
        n *= 2
    if n != T:
        # (a=1, b=0) is the identity element, so padding the tail cannot
        # change any prefix that lies inside the real sequence.
        pad = (0, 0) * (a.dim() - 2) + (0, n - T)
        a = F.pad(a, pad, value=1.0)
        b = F.pad(b, pad, value=0.0)
    level = 0
    while (1 << level) < n:
        s = 1 << level
        a_prev, b_prev = a[:, :-s], b[:, :-s]
        a_cur, b_cur = a[:, s:], b[:, s:]
        # Composing the earlier pair (prev) with the later pair (cur):
        # (a_p, b_p) then (a_c, b_c)  =>  (a_p * a_c, a_c * b_p + b_c)
        a = torch.cat([a[:, :s], a_prev * a_cur], dim=1)
        b = torch.cat([b[:, :s], a_cur * b_prev + b_cur], dim=1)
        level += 1
    return a[:, :T], b[:, :T]


def parallel_scan(a: torch.Tensor, b: torch.Tensor, chunk: int = 0) -> torch.Tensor:
    """h_t = a_t * h_{t-1} + b_t for every t at once (time is dim 1).

    chunk > 0 scans time in slices, carrying the state between them, which
    bounds the memory of the up-sweep's intermediates without changing the
    result (the carry is tested against the recurrence in tests/test_model.py).
    """
    if not chunk or a.shape[1] <= chunk:
        return _scan_core(a, b)[1]
    hs, carry = [], None
    for i in range(0, a.shape[1], chunk):
        prefix_a, h = _scan_core(a[:, i:i + chunk], b[:, i:i + chunk])
        if carry is not None:
            # The slice was scanned assuming a zero state; fold the real one in.
            h = h + prefix_a * carry.unsqueeze(1)
        carry = h[:, -1]
        hs.append(h)
    return torch.cat(hs, dim=1)


# ------------------------------------------------------------------- the block

class MambaBlock(nn.Module):
    """Gated causal conv + diagonal selective SSM, one residual wrapper."""

    def __init__(self, cfg: MambaConfig, d_inner: int):
        super().__init__()
        D, S, K = cfg.d_model, cfg.d_state, cfg.d_conv
        self.d_inner = d_inner
        self.scan_chunk = cfg.scan_chunk
        self.scan_impl = cfg.scan_impl
        self.dt_min, self.dt_max = cfg.dt_min, cfg.dt_max
        self.norm = RMSNorm(D, cfg.norm_eps)
        self.in_proj = nn.Linear(D, 2 * d_inner, bias=False)
        # Causal by construction: pad K-1, convolve, keep the first T outputs.
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=K, padding=K - 1,
                              groups=d_inner, bias=True)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.A = nn.Parameter(torch.arange(1, S + 1, dtype=torch.float32)
                              .repeat(d_inner, 1))
        self.B_proj = nn.Linear(d_inner, S, bias=False)
        self.C_proj = nn.Linear(d_inner, S, bias=False)
        self.D_skip = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, D, bias=False)

    def _ssm(self, x):
        """(a, b, C) for the recurrence, from the post-conv post-SiLU x."""
        dt, B_t, C_t, Aneg = self._proj(x)
        a = torch.exp(-dt.unsqueeze(-1) * Aneg)
        b = dt.unsqueeze(-1) * B_t.unsqueeze(2) * x.unsqueeze(-1)
        return a, b, C_t

    def _proj(self, x):
        """(dt, B, C, Aneg) from the post-conv post-SiLU x.

        Aneg = softplus(A) is the (d_inner, d_state) matrix the kernels use;
        the fused scan applies it directly instead of building `a`.
        """
        dt = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(self.dt_proj(x))
        B_t = self.B_proj(x)
        C_t = self.C_proj(x)
        Aneg = F.softplus(self.A)
        return dt, B_t, C_t, Aneg

    def _scan(self, x, return_state=False):
        """The selective scan: y_t = sum_n C_tn h_tn for the recurrence in _ssm.

        On MPS with scan_impl 'auto'/'metal' this is mamba_metal/scan.py's
        fused kernel (state in registers, no (B, T, d_inner, d_state) tensor);
        'torch' is the pure-PyTorch reference; anything else falls back to the
        associative scan, which is the CPU path the tests have always used.
        return_state=True also yields h[:, -1] for the recurrent cache.
        """
        dt, B_t, C_t, Aneg = self._proj(x)
        r = selective_scan(x, dt, Aneg, B_t, C_t, impl=self.scan_impl,
                           return_state=return_state)
        if r is not None:
            return r
        a = torch.exp(-dt.unsqueeze(-1) * Aneg)
        b = dt.unsqueeze(-1) * B_t.unsqueeze(2) * x.unsqueeze(-1)
        h = parallel_scan(a, b, chunk=self.scan_chunk)
        y = (h * C_t.unsqueeze(2)).sum(-1)
        return (y, h[:, -1]) if return_state else y

    def forward(self, x, cache=None):
        residual = x
        x = self.norm(x)
        x, z = self.in_proj(x).chunk(2, dim=-1)
        K = self.conv.kernel_size[0]
        if cache is not None and cache.get("h") is not None and x.shape[1] == 1:
            # Recurrent step: one token against the cached conv window and state.
            prev = cache["conv"]                       # (B, d_inner, K-1)
            xc = torch.cat([prev, x.transpose(1, 2)], dim=2)
            x = F.silu(self.conv(xc)[:, :, K - 1:K]).transpose(1, 2)
            cache["conv"] = xc[:, :, 1:]
            dt, B_t, C_t, Aneg = self._proj(x)
            a = torch.exp(-dt.unsqueeze(-1) * Aneg)
            b = dt.unsqueeze(-1) * B_t.unsqueeze(2) * x.unsqueeze(-1)
            h = cache["h"] * a[:, 0] + b[:, 0]
            cache["h"] = h
            y = (h * C_t.unsqueeze(2)).sum(-1)
        else:
            xc = x.transpose(1, 2)
            x = F.silu(self.conv(xc)[:, :, :xc.shape[2]]).transpose(1, 2)
            if cache is not None:
                tail = xc[:, :, -(K - 1):]
                if tail.shape[2] < K - 1:
                    tail = F.pad(tail, (K - 1 - tail.shape[2], 0))
                cache["conv"] = tail
            # The fused scan's only written state is the chunk checkpoints, and
            # the last one IS h[:, -1], so the recurrent cache costs nothing.
            y, h_last = self._scan(x, return_state=True)
            if cache is not None:
                cache["h"] = h_last
        y = y + self.D_skip
        return residual + self.out_proj(y * F.silu(z))


class TinyMamba(nn.Module):
    """A Mamba language model sized to a transformer's parameter budget."""

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.d_inner
        if d_inner is None:
            target = transformer_block_params(cfg.d_model, cfg.d_ff)
            d_inner = d_inner_for_budget(target, cfg.d_model, cfg.d_state, cfg.d_conv)
        self.d_inner = d_inner
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(MambaBlock(cfg, d_inner) for _ in range(cfg.n_layers))
        self.ln_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, cache=None):
        if cache is not None and "blocks" not in cache:
            cache["blocks"] = [{} for _ in self.blocks]
        x = self.tok(idx)
        block_caches = cache["blocks"] if cache is not None else [None] * len(self.blocks)
        for block, bc in zip(self.blocks, block_caches):
            x = block(x, bc)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def count_params(self) -> int:
        """Unique parameters (the tied head is counted once)."""
        return sum(p.numel() for p in self.parameters())
