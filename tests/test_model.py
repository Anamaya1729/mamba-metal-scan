"""CPU tests for the model: the associative scan matches the plain
recurrence to 1e-5 (including non-power-of-two lengths and the chunked state
carry), the Mamba block fits the transformer's parameter budget, and the
recurrent cache produces the same logits as recomputing the whole context.
"""
import torch

from mamba_metal.model import (MambaConfig, TinyMamba, d_inner_for_budget,
                               mamba_block_params, parallel_scan, scan_recurrence,
                               transformer_block_params)

SEED = 1337
VOCAB = 200
# The 18M transformer the Mamba is sized against.
MODEL = dict(d_model=384, n_layers=10, d_ff=1536, max_seq_len=512)


def make_cfg(**over):
    kw = dict(vocab_size=VOCAB, d_model=64, n_layers=2, d_ff=128, max_seq_len=64,
              d_state=8, d_conv=4, scan_chunk=0)
    kw.update(over)
    return MambaConfig(**kw)


def mamba_cfg_from_model(**over):
    """A Mamba config sized like the production transformer."""
    kw = dict(vocab_size=VOCAB, d_model=MODEL["d_model"], n_layers=MODEL["n_layers"],
              d_ff=MODEL["d_ff"], max_seq_len=MODEL["max_seq_len"],
              d_state=16, d_conv=4, scan_chunk=0)
    kw.update(over)
    return MambaConfig(**kw)


def fixed_batch(batch=4, seq=32, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, seq), generator=g)


def test_parallel_scan_equals_recurrence():
    g = torch.Generator().manual_seed(SEED)
    a = torch.rand(2, 64, 3, 5, generator=g).clamp(0.01, 0.99)
    b = torch.randn(2, 64, 3, 5, generator=g)
    assert torch.allclose(parallel_scan(a, b), scan_recurrence(a, b), atol=1e-5)


def test_chunked_scan_equals_recurrence():
    # 40 is not a multiple of 8, so the last slice exercises the state carry.
    g = torch.Generator().manual_seed(SEED + 1)
    a = torch.rand(2, 40, 7, generator=g).clamp(0.01, 0.99)
    b = torch.randn(2, 40, 7, generator=g)
    assert torch.allclose(parallel_scan(a, b, chunk=8), scan_recurrence(a, b),
                          atol=1e-5)


def test_scan_handles_non_power_of_two_lengths():
    for T in (1, 2, 3, 5, 17, 33):
        g = torch.Generator().manual_seed(SEED + T)
        a = torch.rand(1, T, 4, generator=g).clamp(0.01, 0.99)
        b = torch.randn(1, T, 4, generator=g)
        assert torch.allclose(parallel_scan(a, b), scan_recurrence(a, b), atol=1e-5)


def test_parameter_budget_matches_the_transformer():
    target = transformer_block_params(MODEL["d_model"], MODEL["d_ff"])
    d_inner = d_inner_for_budget(target, MODEL["d_model"], 16, 4)
    # The sizing helper picks the largest d_inner that fits the budget.
    assert mamba_block_params(MODEL["d_model"], d_inner, 16, 4) <= target
    assert mamba_block_params(MODEL["d_model"], d_inner + 8, 16, 4) > target
    # And the assembled model counts exactly what the helpers say it should.
    mcfg = mamba_cfg_from_model()
    mamba = TinyMamba(mcfg)
    total = (VOCAB * mcfg.d_model                          # embedding, tied with head
             + mcfg.n_layers * mamba_block_params(mcfg.d_model, mamba.d_inner,
                                                  mcfg.d_state, mcfg.d_conv)
             + mcfg.d_model)                               # final norm
    assert mamba.count_params() == total, (mamba.count_params(), total)
    blocks = mcfg.n_layers * transformer_block_params(mcfg.d_model, mcfg.d_ff)
    ours = mcfg.n_layers * mamba_block_params(mcfg.d_model, mamba.d_inner,
                                              mcfg.d_state, mcfg.d_conv)
    assert abs(ours - blocks) / blocks < 0.01


def test_shapes_and_dtypes():
    torch.manual_seed(SEED)
    model = TinyMamba(make_cfg())
    x = fixed_batch()
    logits, loss = model(x, x)
    assert logits.shape == (4, 32, VOCAB)
    assert logits.dtype == torch.float32
    assert loss.dtype == torch.float32
    assert loss.item() > 0


def test_causal_future_tokens_do_not_leak():
    torch.manual_seed(SEED)
    model = TinyMamba(make_cfg()).eval()
    x = fixed_batch()
    x2 = x.clone()
    x2[:, 16:] = (x2[:, 16:] + 1) % VOCAB
    with torch.no_grad():
        a, _ = model(x, x)
        b, _ = model(x2, x2)
    # The conv is causal and the SSM only looks backwards, so the first 16
    # positions cannot see the change at position 16 onward.
    assert torch.equal(a[:, :16], b[:, :16])


def test_overfit_one_batch():
    torch.manual_seed(SEED)
    model = TinyMamba(make_cfg())
    x = fixed_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(200):
        _, loss = model(x, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 1.0, f"did not overfit: final loss {losses[-1]:.4f}"


def test_recurrent_cache_matches_full_recomputation():
    # The cache path decodes one token at a time against a stored conv window
    # and SSM state; the naive path reruns the whole context. Same function.
    torch.manual_seed(SEED)
    model = TinyMamba(make_cfg(scan_chunk=8)).eval()
    full = fixed_batch(seq=26)
    x, step = full[:, :20], full[:, 20:]
    with torch.no_grad():
        cache: dict = {}
        model(x, cache=cache)                    # prime: full pass, stores state
        cached = []
        for t in range(step.shape[1]):
            lg, _ = model(step[:, t:t + 1], cache=cache)
            assert lg.shape == (x.shape[0], 1, VOCAB), lg.shape
            cached.append(lg[:, -1])
        naive = []
        for t in range(step.shape[1]):
            lg, _ = model(torch.cat([x, step[:, :t + 1]], dim=1))
            naive.append(lg[:, -1])
    assert len(cached) == 6
    for i, (a, b) in enumerate(zip(cached, naive)):
        assert torch.allclose(a, b, atol=1e-5), \
            f"step {i}: recurrent state diverges from full recomputation " \
            f"(max delta {(a - b).abs().max().item():.2e})"
