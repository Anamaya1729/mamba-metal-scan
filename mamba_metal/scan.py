"""Fused selective-scan kernels in Metal for the Mamba block.

The plain PyTorch scan (mamba_metal/model.py) materialises the recurrence
as full (batch, time, d_inner, d_state) tensors, and parallel_scan keeps
the up-sweep's intermediates alive for autograd. On an M4 Max that blew up at
47.7 GB at 18M parameters (batch 16 x seq 512) and OOM'd. These kernels are
our Metal equivalent of the hardware-aware scan the official implementation
uses on CUDA: the state for one (batch, channel) pair lives in NVEC float4
registers for the whole kernel, no per-timestep intermediate is ever written
in the forward, and the backward recomputes what it needs from checkpoints.
The same 18M step peaks at 4.14 GB with them (bench/bench.py).

    dA[t,d,n] = exp(-dt[t,d] * Aneg[d,n])
    h[t,d,n]  = dA[t,d,n] * h[t-1,d,n] + dt[t,d] * B[t,n] * x[t,d]    h[-1] = 0
    y[t,d]    = sum_n C[t,n] * h[t,d,n]

FORWARD: one thread per (b, d). NVEC float4 registers hold h[N]; the loop
over t recomputes dA, advances h and writes y. The only state that touches
memory is ckpt (B, ceil(T/K), D, N): h written every K steps, which is what
lets the backward avoid materialising h for the whole sequence.

BACKWARD: one thread per (b, d), chunks from last to first. Each chunk
recomputes its h from ckpt into a (B, K, D, N) scratch (written then read by
the *same* thread, so no cross-thread synchronisation is needed) and walks t
backwards accumulating the adjoint g[N] in registers, carried across chunk
boundaries. Because dA*h_{t-1} = h_t - b_t, every gradient needs only h_t
(never h_{t-1}):

    g_t    = dy_t * C_t + dA_{t+1} * g_{t+1}
    dx_t   = sum_n g_tn * dt_t * B_tn
    dC_tn  = dy_t * h_tn
    dB_tn  = sum_d g_tdn * dt_td * x_td
    ddt_td = sum_n (g_tdn * B_tn * x_td - Aneg_dn * g_tdn * (h_tdn - b_tdn))
    dAneg  = sum_{b,t} -dt_td * g_tdn * (h_tdn - b_tdn)

dB and dC are sums over d, so each threadgroup covers DBLOCK consecutive
channels of one batch element, reduces its DBLOCK partials in threadgroup
memory, and accumulates into the output with atomic_fetch_add_explicit
(Metal 3 float atomics). dx, ddt and dAneg have no reduction at all: one
thread owns each (b, t, d), and dAneg's (b, t) sum is accumulated in
registers and written once per thread.

No mamba-ssm, no CUDA, no prebuilt binaries: the kernels are compiled from
the MSL source below, via torch.mps.compile_shader, once per (d_state, DBLOCK)
and cached for the process.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

try:
    import torch.mps as _mps
except ImportError:  # pragma: no cover -- torch.mps always exists in 2.x
    _mps = None

SSM_CHUNK = 64        # K: the forward checkpoints h every K steps
DBLOCK = 64           # channels per threadgroup (the over-d reduction width)


# --------------------------------------------------------------------- the MSL

_MSL = r"""
using namespace metal;

#define D_STATE {DS}
#define NVEC (D_STATE / 4)
#define DBLOCK {DB}

#define LOAD4(P, O) float4((P)[O], (P)[O + 1], (P)[O + 2], (P)[O + 3])
#define STORE4(P, O, V) do { (P)[O] = (V).x; (P)[O + 1] = (V).y; \
                             (P)[O + 2] = (V).z; (P)[O + 3] = (V).w; } while (0)

// ---------------------------------------------------------------- forward
// x, dt, y: (B, T, D).  Aneg: (D, N).  Bm, Cm: (B, T, N).
// ckpt: (B, nchunk, D, N), the state at the END of each chunk.
kernel void ssm_fwd(device const float* x,
                    device const float* dt,
                    device const float* Aneg,
                    device const float* Bm,
                    device const float* Cm,
                    device float* y,
                    device float* ckpt,
                    constant uint& Bt,
                    constant uint& T,
                    constant uint& D,
                    constant uint& Kc,
                    uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint b = gid.y;
    if (d >= D || b >= Bt) return;
    const uint N = D_STATE;
    const uint DSN = D * N;
    const uint nchunk = (T + Kc - 1) / Kc;

    float4 an[NVEC];
    for (uint j = 0; j < NVEC; ++j)
        an[j] = LOAD4(Aneg, d * N + 4 * j);

    float4 h[NVEC];
    for (uint j = 0; j < NVEC; ++j) h[j] = 0.0f;

    for (uint t = 0; t < T; ++t) {
        const uint xt = (b * T + t) * D + d;      // x, dt, y
        const uint st = (b * T + t) * N;          // Bm, Cm
        const float dtv = dt[xt];
        const float bx = dtv * x[xt];             // dt * x, broadcast over n

        float yv = 0.0f;
        for (uint j = 0; j < NVEC; ++j) {
            float4 bv = LOAD4(Bm, st + 4 * j) * bx;
            float4 da = exp(-dtv * an[j]);
            h[j] = da * h[j] + bv;                // a*h + b, same order as the reference
            yv += dot(LOAD4(Cm, st + 4 * j), h[j]);
        }
        y[xt] = yv;

        const uint tn = t + 1;
        if (tn % Kc == 0 || tn == T) {
            const uint c = (tn % Kc == 0) ? tn / Kc - 1 : nchunk - 1;
            const uint o = (b * nchunk + c) * DSN + d * N;
            for (uint j = 0; j < NVEC; ++j) STORE4(ckpt, o + 4 * j, h[j]);
        }
    }
}

// --------------------------------------------------------------- backward
// dy, dx, ddt: (B, T, D).  dAneg: (D, N).  dBm, dCm: (B, T, N).
// ckpt: forward checkpoints.  scratch: (B, Kc, D, N), h of the chunk in flight.
kernel void ssm_bwd(device const float* x,
                    device const float* dt,
                    device const float* Aneg,
                    device const float* Bm,
                    device const float* Cm,
                    device const float* dy,
                    device float* dx,
                    device float* ddt,
                    device atomic_float* dAneg,
                    device atomic_float* dBm,
                    device atomic_float* dCm,
                    device const float* ckpt,
                    device float* scratch,
                    constant uint& Bt,
                    constant uint& T,
                    constant uint& D,
                    constant uint& Kc,
                    uint2 gid [[thread_position_in_grid]],
                    uint2 lid [[thread_position_in_threadgroup]])
{
    const uint N = D_STATE;
    const uint DSN = D * N;
    const uint nchunk = (T + Kc - 1) / Kc;
    const uint d = gid.x;
    const uint b = gid.y;
    const bool act = (d < D && b < Bt);
    const uint dl = lid.x;

    // dAneg accumulates over the whole backward walk in registers; the two
    // over-d partials are transient (one timestep each).
    threadgroup float shB[DBLOCK * N];
    threadgroup float shC[DBLOCK * N];

    float4 an[NVEC];
    if (act)
        for (uint j = 0; j < NVEC; ++j) an[j] = LOAD4(Aneg, d * N + 4 * j);

    float4 g[NVEC];     // adjoint, carried across chunks in registers
    float4 dAcc[NVEC];  // dAneg partial, summed over t
    for (uint j = 0; j < NVEC; ++j) { g[j] = 0.0f; dAcc[j] = 0.0f; }

    for (int c = int(nchunk) - 1; c >= 0; --c) {
        const uint t0 = uint(c) * Kc;
        const uint t1 = min((uint(c) + 1) * Kc, T);

        // Pass 1: recompute this chunk's h from its checkpoint into scratch.
        // No barrier here: every slot is written and read by the same thread.
        float4 h[NVEC];
        if (act) {
            if (c > 0) {
                const uint o = (b * nchunk + uint(c) - 1) * DSN + d * N;
                for (uint j = 0; j < NVEC; ++j) h[j] = LOAD4(ckpt, o + 4 * j);
            } else {
                for (uint j = 0; j < NVEC; ++j) h[j] = 0.0f;
            }
            for (uint t = t0; t < t1; ++t) {
                const float dtv = dt[(b * T + t) * D + d];
                const float bx = dtv * x[(b * T + t) * D + d];
                const uint st = (b * T + t) * N;
                const uint oh = (b * Kc + (t - t0)) * DSN + d * N;
                for (uint j = 0; j < NVEC; ++j) {
                    float4 bv = LOAD4(Bm, st + 4 * j) * bx;
                    float4 da = exp(-dtv * an[j]);
                    h[j] = da * h[j] + bv;
                    STORE4(scratch, oh + 4 * j, h[j]);
                }
            }
        }

        // Pass 2: walk the chunk backwards. Every thread must hit the barriers,
        // so the work is guarded and the loop count is the same for all.
        for (int ti = int(t1) - 1; ti >= int(t0); --ti) {
            const uint t = uint(ti);
            const uint xt = (b * T + t) * D + d;
            const uint st = (b * T + t) * N;
            float dxv = 0.0f;
            float ddtv = 0.0f;
            if (act) {
                const float dtv = dt[xt];
                const float xv = x[xt];
                const float dyv = dy[xt];
                const float dtn = (t + 1 < T) ? dt[xt + D] : 0.0f;
                const uint oh = (b * Kc + (t - t0)) * DSN + d * N;
                for (uint j = 0; j < NVEC; ++j) {
                    float4 Bv = LOAD4(Bm, st + 4 * j);
                    float4 Cv = LOAD4(Cm, st + 4 * j);
                    float4 bv = Bv * (dtv * xv);        // b_t = dt * B * x
                    float4 hm = LOAD4(scratch, oh + 4 * j) - bv;   // dA*h_{t-1} = h - b
                    float4 da = (t + 1 < T) ? exp(-dtn * an[j]) : float4(0.0f);
                    g[j] = dyv * Cv + da * g[j];        // g_t = dy*C + dA_{t+1} g_{t+1}
                    float4 gb = g[j] * (dtv * xv);      // dB partial (sums over d)
                    dxv += dtv * dot(g[j], Bv);         // dx: no x factor
                    ddtv += xv * dot(g[j], Bv) - dot(an[j], g[j] * hm);
                    dAcc[j] += (-dtv) * g[j] * hm;      // dAneg partial
                    STORE4(shB, dl * N + 4 * j, gb);
                    STORE4(shC, dl * N + 4 * j, dyv * LOAD4(scratch, oh + 4 * j));
                }
                dx[xt] = dxv;
                ddt[xt] = ddtv;
            } else {
                for (uint j = 0; j < NVEC; ++j) {
                    STORE4(shB, dl * N + 4 * j, float4(0.0f));
                    STORE4(shC, dl * N + 4 * j, float4(0.0f));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (dl < N) {      // one thread per state lane, sum the DBLOCK channels
                float sb = 0.0f, sc = 0.0f;
                for (uint i = 0; i < DBLOCK; ++i) {
                    sb += shB[i * N + dl];
                    sc += shC[i * N + dl];
                }
                atomic_fetch_add_explicit(dBm + st + dl, sb, memory_order_relaxed);
                atomic_fetch_add_explicit(dCm + st + dl, sc, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // dAneg (D, N): each thread owns one d, so it accumulates its own row in
    // registers over the whole walk and atomic-adds it once at the end -- no
    // threadgroup reduction needed.
    if (act) {
        const uint o = d * N;
        for (uint j = 0; j < NVEC; ++j)
            for (uint k = 0; k < 4; ++k)
                atomic_fetch_add_explicit(dAneg + o + 4 * j + k, dAcc[j][k],
                                          memory_order_relaxed);
    }
}
"""


_LIBS: dict[tuple[int, int], Any] = {}


def _msl_source(d_state: int, dblock: int) -> str:
    return _MSL.replace("{DS}", str(d_state)).replace("{DB}", str(dblock))


def _get_lib(d_state: int, dblock: int = DBLOCK) -> Any:
    """Compile (and cache) the kernels for one (d_state, dblock) pair."""
    if d_state % 4 != 0 or d_state <= 0:
        raise ValueError(f"d_state must be a positive multiple of 4 for the Metal "
                         f"scan; got {d_state} (scan_impl='torch' takes any size)")
    key = (d_state, dblock)
    lib = _LIBS.get(key)
    if lib is None:
        if _mps is None or not hasattr(torch._C, "_mps_compileShader"):
            raise RuntimeError("MPS is not available in this torch build")
        lib = _mps.compile_shader(_msl_source(d_state, dblock))
        _LIBS[key] = lib
    return lib


def _launch(x: torch.Tensor, dblock: int) -> tuple[int, int]:
    """Grid geometry: DBLOCK channels per threadgroup, one row per batch."""
    d = x.shape[2]
    return ((d + dblock - 1) // dblock * dblock, x.shape[0]), (dblock, 1)


_SCRATCH: dict[tuple, torch.Tensor] = {}


def _get_scratch(bt: int, chunk: int, d: int, n: int) -> torch.Tensor:
    """The (B, K, D, N) chunk buffer, kept alive and reused across layers.

    Every slot is written before it is read by the same thread, and one
    layer's backward finishes before the next one's starts, so a single
    buffer serves the whole backward pass.
    """
    key = (bt, chunk, d, n)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = torch.empty(bt, chunk, d, n, device="mps", dtype=torch.float32)
        _SCRATCH[key] = buf
    return buf


# ------------------------------------------------------- reference (pure torch)

def selective_scan_ref(x: torch.Tensor, dt: torch.Tensor, aneg: torch.Tensor,
                       b_in: torch.Tensor, c_in: torch.Tensor,
                       return_state: bool = False):
    """The same recurrence, left to right, in pure PyTorch.

    Differentiable, any device, any dtype, any d_state -- this is the
    reference the Metal kernels are checked against, and the "torch" path.
    """
    bt, t_len, d = x.shape
    n = aneg.shape[1]
    h = torch.zeros(bt, d, n, dtype=x.dtype, device=x.device)
    out = []
    for i in range(t_len):
        dA = torch.exp(-dt[:, i, :].unsqueeze(-1) * aneg.unsqueeze(0))   # (B,D,N)
        b_t = dt[:, i, :].unsqueeze(-1) * b_in[:, i, :].unsqueeze(1) * x[:, i, :].unsqueeze(-1)
        h = dA * h + b_t
        out.append((c_in[:, i, :].unsqueeze(1) * h).sum(-1))             # (B,D)
    y = torch.stack(out, dim=1)
    if return_state:
        return y, h
    return y


# ------------------------------------------------------------------- the kernels

class SelectiveScanMetal(torch.autograd.Function):
    """selective_scan_metal(x, dt, Aneg, B, C) -> y, forward and backward."""

    @staticmethod
    def forward(ctx, x, dt, aneg, b_in, c_in, chunk=SSM_CHUNK, dblock=DBLOCK,
                return_state=False):
        if not (x.is_mps and dt.is_mps and aneg.is_mps
                and b_in.is_mps and c_in.is_mps):
            raise RuntimeError("scan_impl='metal' needs every input on the MPS "
                               f"device; saw devices "
                               f"{x.device}, {dt.device}, {aneg.device}, "
                               f"{b_in.device}, {c_in.device}")
        if not all(t.dtype == torch.float32
                   for t in (x, dt, aneg, b_in, c_in)):
            raise RuntimeError("scan_impl='metal' is fp32-only (the kernels are "
                               "float32); got "
                               + ", ".join(str(t.dtype) for t in (x, dt, aneg, b_in, c_in)))
        if x.dim() != 3 or dt.shape != x.shape or aneg.dim() != 2:
            raise ValueError(f"bad shapes: x {tuple(x.shape)}, dt {tuple(dt.shape)}, "
                             f"Aneg {tuple(aneg.shape)}")
        bt, t_len, d = x.shape
        n = aneg.shape[1]
        if aneg.shape[0] != d:
            raise ValueError(f"Aneg {tuple(aneg.shape)} does not match x's "
                             f"channel count {d}")
        if b_in.shape != (bt, t_len, n) or c_in.shape != (bt, t_len, n):
            raise ValueError(f"B {tuple(b_in.shape)} / C {tuple(c_in.shape)} must be "
                             f"({bt}, {t_len}, {n})")

        lib = _get_lib(n, dblock)
        x = x.contiguous()
        dt = dt.contiguous()
        aneg = aneg.contiguous()
        b_in = b_in.contiguous()
        c_in = c_in.contiguous()
        y = torch.empty(bt, t_len, d, device="mps", dtype=torch.float32)
        ckpt = torch.zeros(bt, (t_len + chunk - 1) // chunk, d, n,
                           device="mps", dtype=torch.float32)
        threads, groups = _launch(x, dblock)
        lib.ssm_fwd(x, dt, aneg, b_in, c_in, y, ckpt, bt, t_len, d, chunk,
                    threads=threads, group_size=groups)
        ctx.save_for_backward(x, dt, aneg, b_in, c_in, ckpt)
        ctx.chunk, ctx.dblock = chunk, dblock
        if return_state:
            # The last checkpoint IS h[:, -1]: the state at the end of the
            # final chunk. Nothing extra has to be computed or stored. It is
            # returned detached: it is a snapshot for the recurrent cache at
            # inference, not a value the training graph should carry.
            return y, ckpt[:, -1].detach()
        return y

    @staticmethod
    def backward(ctx, *grads):
        # One grad per autograd-tracked output. The returned state is detached,
        # so only dy arrives; *grads keeps this honest if that ever changes.
        dy = grads[0]
        x, dt, aneg, b_in, c_in, ckpt = ctx.saved_tensors
        chunk, dblock = ctx.chunk, ctx.dblock
        bt, t_len, d = x.shape
        n = aneg.shape[1]
        dy = dy.contiguous()

        lib = _get_lib(n, dblock)
        dx = torch.empty_like(x)
        ddt = torch.empty_like(dt)
        dAneg = torch.zeros_like(aneg)     # atomically summed over (b, t)
        dB = torch.zeros_like(b_in)        # atomically summed over d
        dC = torch.zeros_like(c_in)
        scratch = _get_scratch(bt, chunk, d, n)
        threads, groups = _launch(x, dblock)
        lib.ssm_bwd(x, dt, aneg, b_in, c_in, dy, dx, ddt, dAneg, dB, dC,
                    ckpt, scratch, bt, t_len, d, chunk,
                    threads=threads, group_size=groups)
        return dx, ddt, dAneg, dB, dC, None, None, None


def selective_scan_metal(x: torch.Tensor, dt: torch.Tensor, aneg: torch.Tensor,
                         b_in: torch.Tensor, c_in: torch.Tensor,
                         chunk: int = SSM_CHUNK,
                         dblock: int = DBLOCK,
                         return_state: bool = False):
    """y = selective_scan(x, dt, Aneg, B, C) on MPS (see module docstring)."""
    return SelectiveScanMetal.apply(x, dt, aneg, b_in, c_in, chunk, dblock,
                                    return_state)


def selective_scan(x: torch.Tensor, dt: torch.Tensor, aneg: torch.Tensor,
                   b_in: torch.Tensor, c_in: torch.Tensor,
                   impl: str = "auto",
                   chunk: int = SSM_CHUNK,
                   return_state: bool = False):
    """Dispatch the scan on `impl`.

      'metal'  the fused kernels; raises if any input is not on MPS.
      'torch'  selective_scan_ref, the pure-PyTorch reference, anywhere.
      'auto'   the kernels on MPS, otherwise None so the caller can fall
               back to whatever it already had (model.py's
               parallel_scan on CPU). 'auto' never silently changes the math.
    """
    if impl == "torch":
        return selective_scan_ref(x, dt, aneg, b_in, c_in, return_state=return_state)
    if impl not in ("auto", "metal"):
        raise ValueError(f"unknown scan_impl {impl!r}; want 'auto', 'metal' or 'torch'")
    if x.is_mps:
        return selective_scan_metal(x, dt, aneg, b_in, c_in, chunk=chunk,
                                    return_state=return_state)
    if impl == "metal":
        raise RuntimeError(f"scan_impl='metal' needs MPS tensors; got device "
                           f"{x.device.type!r}")
    return None
