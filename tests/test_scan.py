"""Correctness gates for the Metal selective scan (mamba_metal/scan.py).

Everything that touches MPS is skipped when the host has no MPS device, so
the file is inert on Linux and runs in full on an Apple-silicon Mac:

1. the pure-PyTorch reference matches mamba.py's scan_recurrence in float64
2. the kernels' forward and every gradient (x, dt, Aneg, B, C) match the
   reference at fp32, over shapes that break the easy assumptions
3. end to end, a TinyMamba's loss and every parameter gradient are the same
   whether the block scans with scan_impl 'torch' or 'metal'
"""
from __future__ import annotations

import pytest
import torch

from mamba_metal.model import MambaConfig, TinyMamba, scan_recurrence
from mamba_metal.scan import (selective_scan, selective_scan_metal,
                              selective_scan_ref)

SEED = 1337

VOCAB = 200
ATOL, RTOL = 1e-4, 1e-3

HAS_MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(
    not HAS_MPS, reason="no MPS device on this host")


def _inputs(bt, t_len, d, n, device="cpu", seed=SEED, dtype=torch.float32):
    """Inputs shaped like a Mamba block's: x/dt (B,T,D), B/C (B,T,N), Aneg (D,N)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(bt, t_len, d, generator=g, dtype=dtype).to(device)
    dt = (0.001 + 0.099 * torch.rand(bt, t_len, d, generator=g, dtype=dtype)).to(device)
    aneg = torch.nn.functional.softplus(
        torch.arange(1, n + 1, dtype=dtype).repeat(d, 1) + 0.1).to(device)
    b_in = (0.5 * torch.randn(bt, t_len, n, generator=g, dtype=dtype)).to(device)
    c_in = (0.5 * torch.randn(bt, t_len, n, generator=g, dtype=dtype)).to(device)
    return x, dt, aneg, b_in, c_in


# ------------------------------------------------------------------ gate 1

def test_reference_matches_the_recurrence_in_float64():
    """selective_scan_ref is the same recurrence mamba.py has always used."""
    x, dt, aneg, b_in, c_in = _inputs(2, 40, 7, 5, dtype=torch.float64, seed=SEED + 3)
    a = torch.exp(-dt.unsqueeze(-1) * aneg.unsqueeze(0))
    b = dt.unsqueeze(-1) * b_in.unsqueeze(2) * x.unsqueeze(-1)
    h = scan_recurrence(a, b)
    expected = (h * c_in.unsqueeze(2)).sum(-1)
    y = selective_scan_ref(x, dt, aneg, b_in, c_in)
    assert torch.allclose(y, expected, atol=1e-10), \
        (y - expected).abs().max().item()


def test_reference_matches_the_recurrence_odd_shapes():
    for t_len, d, n in ((1, 1, 4), (33, 5, 8), (64, 1, 16), (97, 3, 4)):
        x, dt, aneg, b_in, c_in = _inputs(1, t_len, d, n, dtype=torch.float64,
                                          seed=SEED + t_len)
        a = torch.exp(-dt.unsqueeze(-1) * aneg.unsqueeze(0))
        b = dt.unsqueeze(-1) * b_in.unsqueeze(2) * x.unsqueeze(-1)
        h = scan_recurrence(a, b)
        y = selective_scan_ref(x, dt, aneg, b_in, c_in)
        assert torch.allclose(y, (h * c_in.unsqueeze(2)).sum(-1), atol=1e-10)


def test_unknown_scan_impl_is_rejected():
    x, dt, aneg, b_in, c_in = _inputs(1, 4, 8, 4)
    with pytest.raises(ValueError, match="scan_impl"):
        selective_scan(x, dt, aneg, b_in, c_in, impl="nonsense")


@needs_mps
def test_metal_requires_mps_tensors():
    """scan_impl 'metal' on a non-MPS tensor raises instead of falling back."""
    x, dt, aneg, b_in, c_in = _inputs(1, 8, 16, 4)
    with pytest.raises(RuntimeError, match="MPS"):
        selective_scan(x, dt, aneg, b_in, c_in, impl="metal")
    # 'auto' on CPU declines and hands None back to the caller.
    assert selective_scan(x, dt, aneg, b_in, c_in, impl="auto") is None


# ------------------------------------------------------------------ gate 2

_SHAPES = [
    # (B, T, D, N) -- T not a multiple of 64, D not a multiple of the
    # threadgroup width, batch 1, a single timestep, and a production-size run.
    (2, 64, 32, 16),
    (1, 100, 17, 16),
    (4, 63, 128, 16),
    (1, 1, 16, 16),
    (2, 130, 40, 8),      # d_state 8 as well: the kernels are compiled per d_state
    (3, 512, 856, 16),    # an 18M model's d_inner
]


@pytest.mark.parametrize("bt,t_len,d,n", _SHAPES)
@needs_mps
def test_metal_forward_matches_the_reference(bt, t_len, d, n):
    x, dt, aneg, b_in, c_in = _inputs(bt, t_len, d, n, device="mps", seed=SEED + d)
    y = selective_scan_metal(x, dt, aneg, b_in, c_in)
    ref = selective_scan_ref(x, dt, aneg, b_in, c_in)
    assert y.shape == (bt, t_len, d)
    assert y.dtype == torch.float32
    assert torch.allclose(y, ref, atol=ATOL, rtol=RTOL), \
        f"max delta {(y - ref).abs().max().item():.3e} " \
        f"vs scale {ref.abs().max().item():.3e}"


@pytest.mark.parametrize("bt,t_len,d,n", _SHAPES)
@needs_mps
def test_metal_gradients_match_the_reference(bt, t_len, d, n):
    """Every one of the five gradients, analytic on both sides."""
    x, dt, aneg, b_in, c_in = _inputs(bt, t_len, d, n, device="mps", seed=SEED + d)
    inputs = [t.clone().requires_grad_(True) for t in (x, dt, aneg, b_in, c_in)]
    dy = torch.randn(bt, t_len, d, device="mps")

    y = selective_scan_metal(*inputs)
    g_metal = torch.autograd.grad(y, inputs, grad_outputs=dy)

    ref = selective_scan_ref(*inputs)
    g_ref = torch.autograd.grad(ref, inputs, grad_outputs=dy)

    names = ("dx", "ddt", "dAneg", "dB", "dC")
    for name, gm, gr in zip(names, g_metal, g_ref):
        assert gm.shape == gr.shape, name
        assert torch.allclose(gm, gr, atol=ATOL, rtol=RTOL), \
            f"{name}: max delta {(gm - gr).abs().max().item():.3e} " \
            f"vs scale {gr.abs().max().item():.3e}"


@needs_mps
def test_metal_gradients_match_finite_differences():
    """The kernels' gradients are the real derivatives, not just consistent."""
    bt, t_len, d, n = 1, 24, 12, 8
    x, dt, aneg, b_in, c_in = _inputs(bt, t_len, d, n, device="mps", seed=SEED + 11)
    dy = torch.randn(bt, t_len, d, device="mps")

    def f(i, j, v):
        inp = [t.clone() for t in (x, dt, aneg, b_in, c_in)]
        view = inp[i].reshape(-1)
        view[j] = v
        return (selective_scan_metal(*inp) * dy).sum()

    inputs = [t.clone().requires_grad_(True) for t in (x, dt, aneg, b_in, c_in)]
    grads = torch.autograd.grad(selective_scan_metal(*inputs), inputs, grad_outputs=dy)

    # Sample a handful of entries per tensor (finite differences are slow).
    g = torch.Generator(device="cpu").manual_seed(SEED)
    worst = 0.0
    for i, tensor in enumerate(inputs):
        flat = tensor.reshape(-1)
        idx = torch.randperm(flat.numel(), generator=g)[:6]
        for j in idx.tolist():
            h = 1e-3 * max(1.0, abs(flat[j].item()))
            num = (f(i, j, flat[j] + h) - f(i, j, flat[j] - h)) / (2 * h)
            ana = grads[i].reshape(-1)[j]
            scale = max(1.0, abs(num), abs(ana))
            worst = max(worst, (num - ana).abs().item() / scale)
    assert worst < 0.05, f"finite-difference mismatch: {worst:.3f}"


# ------------------------------------------------------------------ gate 3

def _tinymamba(scan_impl, device):
    cfg = MambaConfig(vocab_size=VOCAB, d_model=64, n_layers=2, d_ff=128,
                      max_seq_len=64, d_state=8, d_conv=4, scan_chunk=0,
                      scan_impl=scan_impl)
    torch.manual_seed(SEED)
    model = TinyMamba(cfg).to(device)
    model.eval()
    return model


@needs_mps
def test_end_to_end_torch_vs_metal():
    """Same TinyMamba, same weights: identical loss and parameter gradients."""
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, VOCAB, (2, 48), generator=g).to("mps")

    ref = _tinymamba("torch", "mps")
    met = _tinymamba("metal", "mps")
    met.load_state_dict(ref.state_dict())

    for name, p in ref.named_parameters():
        assert torch.equal(p, dict(met.named_parameters())[name])

    loss_r, grads_r = _loss_and_grads(ref, idx.clone())
    loss_m, grads_m = _loss_and_grads(met, idx.clone())

    assert torch.allclose(loss_r, loss_m, atol=ATOL, rtol=RTOL), \
        f"loss {loss_r.item():.6f} vs {loss_m.item():.6f}"
    assert len(grads_r) == len(grads_m)
    worst = 0.0
    for (name, gr), (_, gm) in zip(grads_r, grads_m):
        assert gr.shape == gm.shape, name
        scale = max(1.0, gr.abs().max().item())
        worst = max(worst, (gr - gm).abs().max().item() / scale)
        assert torch.allclose(gr, gm, atol=ATOL, rtol=RTOL), \
            f"{name}: delta {(gr - gm).abs().max().item():.3e} / scale {scale:.3e}"
    assert worst < 1e-3, f"worst relative gradient delta {worst:.3e}"


def _loss_and_grads(model, idx):
    model.zero_grad(set_to_none=True)
    _, loss = model(idx, idx)
    loss.backward()
    return loss.detach(), [(n, p.grad.detach().clone()) for n, p in
                           model.named_parameters()]


@needs_mps
def test_scan_chunk_boundaries_agree_with_the_reference():
    """The checkpoint stride K is an implementation detail, not a result."""
    x, dt, aneg, b_in, c_in = _inputs(2, 512, 96, 16, device="mps", seed=SEED + 5)
    ref = selective_scan_ref(x, dt, aneg, b_in, c_in)
    for chunk in (1, 7, 64, 511, 512):
        y = selective_scan_metal(x, dt, aneg, b_in, c_in, chunk=chunk)
        assert torch.allclose(y, ref, atol=ATOL, rtol=RTOL), \
            f"chunk {chunk}: {(y - ref).abs().max().item():.3e}"


@needs_mps
def test_returned_state_is_the_recurrence_state():
    x, dt, aneg, b_in, c_in = _inputs(1, 100, 16, 16, device="mps", seed=SEED + 7)
    _, state = selective_scan_metal(x, dt, aneg, b_in, c_in, return_state=True)
    _, ref_state = selective_scan_ref(x, dt, aneg, b_in, c_in, return_state=True)
    assert state.shape == ref_state.shape
    assert torch.allclose(state, ref_state, atol=ATOL, rtol=RTOL)


# ------------------------------------------------------------------ gate 3/4

def _peak_during(fn) -> int:
    """Peak driver memory while fn runs, sampled on a thread (no MPS peak API)."""
    import threading, time

    peak = 0
    stop = threading.Event()

    def poll():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, torch.mps.driver_allocated_memory())
            time.sleep(0.01)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    try:
        fn()
        torch.mps.synchronize()
    finally:
        stop.set()
        th.join(timeout=5)
    return peak


@needs_mps
def test_18m_train_step_fits_in_16gb():
    """The configuration that OOM'd at 47.7 GB on the PyTorch scan.

    18M-class TinyMamba (d_model 384, 10 layers, config d_state/d_conv),
    batch 16 x seq 512, one full train step: forward, backward, AdamW.
    """
    cfg = MambaConfig(vocab_size=VOCAB, d_model=384, n_layers=10, d_ff=1536,
                      max_seq_len=512, d_state=8, d_conv=4, scan_chunk=128,
                      scan_impl="metal")
    torch.manual_seed(SEED)
    model = TinyMamba(cfg).to("mps")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, VOCAB, (16, 513), generator=g).to("mps")

    def step():
        _, loss = model(idx[:, :-1], idx[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    step()                                     # warm up, allocate everything
    torch.mps.synchronize()
    torch.mps.empty_cache()
    peak = _peak_during(step)
    gb = peak / 2**30
    print(f"\n[gate] 18M-class step: peak MPS {gb:.2f} GB (target < 16 GB), "
          f"{model.count_params():,} params")
    assert gb < 16.0, f"peak {gb:.2f} GB exceeds the 16 GB target"
    del model, opt
    torch.mps.empty_cache()

