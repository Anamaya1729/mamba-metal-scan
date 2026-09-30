# mamba-metal-scan

A fused selective-scan kernel for Mamba on Apple silicon, forward and backward, written in Metal and callable from PyTorch.

The official Mamba kernels are CUDA-only. On a Mac, the usual fallback is a PyTorch scan that materialises the whole `(batch, time, d_inner, d_state)` state tensor and keeps the scan's intermediates alive for autograd. That runs out of memory fast. An 18M-parameter Mamba at batch 16 x sequence 512 peaked at **47.7 GB** and OOM'd on a 36 GB M4 Max.

With this kernel the same training step peaks at **4.18 GB**. It has been training an 80M Mamba on 2B tokens on that laptop since 30 September 2026.

## Numbers

All measured on a MacBook Pro M4 Max (32-core GPU, 36 GB), PyTorch 2.14, fp32.

| Model | Batch x seq | PyTorch scan | This kernel |
|---|---|---|---|
| 18M Mamba, one full train step | 16 x 512 | OOM at 47.7 GB | 4.18 GB peak |
| 60M Mamba, small vocab | 16 x 512 | — | 13,600 tok/s, 7.6 GB peak |
| 80M Mamba, 32,768 vocab, real pretraining run | 16 x 512 | — | ~6,500 tok/s sustained, ~13 GB |

The 80M run's validation loss went from 4.21 at step 5k to 3.68 at step 27k with no instability. The kernel runs in the training loop, not just in the benchmark.

## Install

```bash
git clone https://github.com/Anamaya1729/mamba-metal-scan
cd mamba-metal-scan
pip install -e ".[test]"
```

You need an Apple-silicon Mac and a PyTorch build with `torch.mps.compile_shader` (2.8 or newer). Nothing is precompiled: the Metal source is compiled on first use and cached per `d_state`.

## Use it in your own Mamba

The kernel computes the diagonal selective scan:

```
dA[t,d,n] = exp(-dt[t,d] * Aneg[d,n])
h[t,d,n]  = dA[t,d,n] * h[t-1,d,n] + dt[t,d] * B[t,n] * x[t,d]      h[-1] = 0
y[t,d]    = sum_n C[t,n] * h[t,d,n]
```

If your block already computes `x`, `dt`, `A`, `B` and `C`, replace the scan with one call:

```python
import torch.nn.functional as F
from mamba_metal import selective_scan

# x, dt: (batch, time, d_inner)   B, C: (batch, time, d_state)
# Aneg:  (d_inner, d_state), positive -- e.g. softplus(A_log)
y = selective_scan(x, dt, F.softplus(A), B, C)           # (batch, time, d_inner)
```

- `impl="auto"` (default) runs the Metal kernel when the tensors are on MPS and returns `None` otherwise, so you can fall back to your own CPU path.
- `impl="metal"` forces the kernel and raises off MPS.
- `impl="torch"` runs the pure-PyTorch reference anywhere.
- `return_state=True` also returns the final state `h[:, -1]`, for a recurrent cache at inference. It costs nothing extra because it is the last checkpoint the forward already wrote.

Gradients flow to all five inputs. Add the `D` skip, the gate and the projections around it as usual.

A complete small Mamba language model that uses the kernel is in `mamba_metal/model.py` (`MambaConfig`, `TinyMamba`). Its `d_inner` is solved so that each block costs the same as a transformer block of the same `d_model`/`d_ff`. That makes transformer-vs-Mamba comparisons fair at equal parameters.

```python
from mamba_metal import MambaConfig, TinyMamba

cfg = MambaConfig(vocab_size=32768, d_model=640, n_layers=12, d_ff=2560,
                  d_state=16, scan_impl="auto")
model = TinyMamba(cfg).to("mps")
logits, loss = model(tokens, targets)
```

## How it works

**Forward:** one GPU thread per `(batch, channel)`. The whole `d_state` vector lives in registers as `float4`s for the full length of the sequence. Each step recomputes `dA`, advances `h` and writes `y`; no per-timestep state is ever written to memory. The only thing stored is a checkpoint of `h` every 64 steps.

**Backward:** walks the chunks from last to first. For each chunk it recomputes `h` from the checkpoint, then walks time backwards, carrying the adjoint in registers across chunk boundaries. Because `dA * h[t-1] = h[t] - b[t]`, every gradient needs only `h[t]`, never `h[t-1]`. `dB` and `dC` are sums over channels: each threadgroup reduces 64 channels in threadgroup memory, then adds into the output with Metal 3 float atomics. `dx`, `ddt` and `dA` need no cross-thread reduction.

This is the same idea as the hardware-aware scan in the original Mamba paper (keep the state on-chip, recompute instead of storing), done for Metal instead of CUDA.

## Limits

- **fp32 only.** Run the Mamba blocks outside bf16/fp16 autocast. The kernel raises on anything but float32. Under autocast on MPS it crashes.
- `d_state` must be a multiple of 4 (the `float4` registers). Mamba's usual 16 is fine.
- Tested on an M4 Max only. It should run on any Apple GPU with Metal 3 float atomics, but I haven't measured anything else. Numbers from other chips are welcome.

## Tests

```bash
pytest
```

On a Mac all 29 tests run. On Linux the 18 MPS tests skip and the CPU ones still run. They check:

- The PyTorch reference matches the plain recurrence to 1e-10 in float64.
- The kernel's forward and all five gradients match the reference at fp32, on shapes chosen to break easy assumptions: sequence lengths that aren't multiples of 64, channel counts that aren't multiples of the threadgroup, a single timestep, batch 1, and `d_state` 8 and 16.
- The gradients match finite differences, so they are the real derivatives and not merely consistent with the reference.
- Checkpoint strides of 1, 7, 64, 511 and 512 all give the same answer.
- End to end, a model's loss and every parameter gradient are identical with `scan_impl="torch"` and `"metal"`.
- An 18M training step fits under 16 GB.
- Token-by-token decoding with the recurrent cache matches rerunning the full context, at batch size 4.

## Benchmark

```bash
python bench/bench.py              # 18M and 60M, one full train step each
python bench/bench.py --ref        # also the pure-PyTorch scan at a size it survives
```

Close other GPU work first; the numbers are peak memory and tokens/sec for forward + backward + AdamW.

## License

MIT
