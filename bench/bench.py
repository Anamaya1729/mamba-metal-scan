#!/usr/bin/env python3
"""Memory and throughput for the fused Metal selective scan.

Two questions this answers:

  * does an 18M-parameter TinyMamba survive batch 16 x seq 512 on the
    Mac (the configuration that OOM'd at 47.7 GB on the PyTorch scan)?
  * what does the scan cost at the 60M class?

Peak MPS memory is sampled on a thread while one real train step runs
(forward + backward + AdamW), so the number includes autograd's saved
tensors, not just the steady state. Tokens/sec is forward+backward only.

    python bench/bench.py              # both sizes, metal scan
    python bench/bench.py --ref        # also the 'torch' scan,
                                                  # at a size it can survive
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mamba_metal.model import (MambaConfig, TinyMamba,  # noqa: E402
                               mamba_block_params, transformer_block_params)

SEED = 1337
MAMBA = dict(d_state=16, d_conv=4, scan_chunk=128, dt_min=0.001, dt_max=0.1)

# Sizes are the transformer budget each Mamba is matched to.
SIZES = {
    "18M": dict(d_model=384, n_layers=10, d_ff=1536, batch=16, seq=512),
    "60M": dict(d_model=640, n_layers=12, d_ff=2560, batch=16, seq=512),
}
REF_SIZE = dict(d_model=128, n_layers=2, d_ff=512, batch=4, seq=128)


def make_cfg(size, vocab, scan_impl):
    kw = dict(vocab_size=vocab, max_seq_len=size["seq"],
              d_state=MAMBA["d_state"], d_conv=MAMBA["d_conv"],
              scan_chunk=MAMBA["scan_chunk"], dt_min=MAMBA["dt_min"],
              dt_max=MAMBA["dt_max"], scan_impl=scan_impl)
    kw.update({k: v for k, v in size.items() if k in ("d_model", "n_layers", "d_ff")})
    return MambaConfig(**kw)


class PeakSampler:
    """Polls driver memory on a thread; MPS has no peak counter of its own."""

    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, torch.mps.driver_allocated_memory())
            except RuntimeError:                      # MPS torn down
                return
            time.sleep(0.01)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def bench(name, size, vocab, scan_impl, steps):
    torch.manual_seed(SEED)
    cfg = make_cfg(size, vocab, scan_impl)
    model = TinyMamba(cfg).to("mps")
    n_params = model.count_params()
    block = mamba_block_params(cfg.d_model, model.d_inner, cfg.d_state, cfg.d_conv)
    budget = transformer_block_params(cfg.d_model, cfg.d_ff)
    print(f"[{name}] scan_impl={scan_impl}  params {n_params:,} "
          f"({n_params / 1e6:.2f}M)  d_inner {model.d_inner}  "
          f"block {block:,} vs transformer {budget:,} "
          f"({100.0 * block / budget:.1f}%)")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, vocab, (size["batch"], size["seq"] + 1),
                        generator=g).to("mps")

    def step():
        x, y = idx[:, :-1], idx[:, 1:]
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    step()                                     # warm up, allocate everything
    torch.mps.synchronize()
    torch.mps.empty_cache()
    with PeakSampler() as sampler:
        t0 = time.perf_counter()
        for _ in range(steps):
            loss = step()
        torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

    peak_gb = sampler.peak / 2**30
    toks = size["batch"] * size["seq"] * steps
    print(f"[{name}] step {steps}x{size['batch']}x{size['seq']}  loss "
          f"{loss.item():.4f}  peak MPS {peak_gb:.2f} GB  "
          f"{toks / elapsed:,.0f} tok/s  ({1000 * elapsed / steps:.0f} ms/step)")
    del model, opt
    torch.mps.empty_cache()
    return dict(params=n_params, peak_gb=peak_gb, tok_s=toks / elapsed,
                ms_per_step=1000 * elapsed / steps, loss=loss.item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=3,
                    help="train steps to time (default 3)")
    ap.add_argument("--ref", action="store_true",
                    help="also time scan_impl 'torch' at a small size")
    ap.add_argument("--only", choices=sorted(SIZES),
                    help="run just one size")
    ap.add_argument("--vocab", type=int, default=32768,
                    help="vocabulary size (default 32768)")
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        sys.exit("no MPS device -- this benchmark needs an Apple-silicon Mac")
    torch.set_num_threads(4)

    vocab = args.vocab
    sizes = [args.only] if args.only else list(SIZES)
    rows = [(n, bench(n, SIZES[n], vocab, "metal", args.steps))
            for n in sizes]
    if args.ref:
        rows.append(("ref-torch", bench("ref-torch", REF_SIZE, vocab, "torch",
                                        args.steps)))

    print()
    print(f"{'run':<10} {'params':>13} {'peak MPS':>10} {'tok/s':>10} "
          f"{'ms/step':>9}  loss")
    for name, r in rows:
        print(f"{name:<10} {r['params']:>13,} {r['peak_gb']:>9.2f}GB "
              f"{r['tok_s']:>10,.0f} {r['ms_per_step']:>9.0f}  {r['loss']:.4f}")
    for name, r in rows:
        if r["peak_gb"] >= 16:
            print(f"[gate] {name}: {r['peak_gb']:.2f} GB exceeds the 16 GB target")


if __name__ == "__main__":
    main()
