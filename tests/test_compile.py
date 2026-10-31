"""PolyLoRA on FlashDeBERTa under torch.compile: CUDA graphs, no recompiles when adapters change."""
import warnings

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

from polylora import PolyLoraConfig, PolyLoraModel
from polylora.store import CpuLayerWeights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
flashdeberta = pytest.importorskip("flashdeberta")

TARGETS = ["query_proj", "key_proj", "value_proj", "pos_key_proj", "pos_query_proj", "dense"]
L, LENGTHS = 96, [96, 50, 13, 77]


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.config.error_on_recompile = False
    torch._dynamo.reset()


def make_model(max_gpu_adapters=3, num_adapters=5):
    from transformers import DebertaV2Config

    config = DebertaV2Config(
        hidden_size=128, num_attention_heads=2, intermediate_size=256, num_hidden_layers=2,
        relative_attention=True, position_buckets=64, max_relative_positions=-1, pos_att_type=["p2c", "c2p"],
        share_att_key=False, norm_rel_ebd="layer_norm", position_biased_input=False, vocab_size=200,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    torch.manual_seed(0)
    base = flashdeberta.FlashDebertaV2Model(config).cuda().to(torch.bfloat16).eval()
    model = PolyLoraModel(base, PolyLoraConfig(max_gpu_adapters=max_gpu_adapters, max_rank=8,
                                               target_modules=TARGETS)).eval()
    g = torch.Generator().manual_seed(1)
    for i in range(num_adapters):
        rank = [4, 8, 2][i % 3]
        layers = {}
        for name in model.target_module_names:
            linear = base.get_submodule(name).base_layer
            layers[name] = CpuLayerWeights(A=torch.randn(linear.in_features, rank, generator=g) * 0.1,
                                           B=torch.randn(rank, linear.out_features, generator=g) * 0.1,
                                           rank=rank, scale=0.5)
        model.adapter_store.add_adapter(f"a{i}", layers)
    return model


def make_batch():
    g = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(1, 200, (len(LENGTHS), L), device="cuda", generator=g)
    mask = (torch.arange(L, device="cuda")[None] < torch.tensor(LENGTHS, device="cuda")[:, None]).long()
    return ids, mask


def masked_rel_err(a, b, mask):
    keep = mask[..., None]
    return (((a.float() - b.float()) * keep).norm() / (b.float() * keep).norm()).item()


def test_adapter_changes_never_recompile():
    """New mixes, adapters loaded into the GPU cache and evictions all reuse one graph."""
    model = make_model()
    ids, mask = make_batch()
    compiled = torch.compile(model, mode="reduce-overhead")
    mixes = [
        ["a0", "a1", "__base__", "a0"],
        ["a1", "a1", "a1", "a1"],
        ["a3", "a4", "a3", "__base__"],  # not resident yet: loads and evictions
        ["a0", "a3", "a4", "a3"],
        ["__base__"] * 4,
    ]
    for step, adapters in enumerate(mixes):
        with torch.no_grad():
            expected = model(input_ids=ids, attention_mask=mask, adapter_ids=adapters).last_hidden_state
            for _ in range(3):  # warm-up, record, replay
                got = compiled(input_ids=ids, attention_mask=mask, adapter_ids=adapters).last_hidden_state.clone()
        assert masked_rel_err(got, expected, mask) < 2e-2, adapters
        if step == 0:
            torch._dynamo.config.error_on_recompile = True
    assert counters["stats"]["unique_graphs"] == 1
    assert counters["inductor"]["cudagraph_skips"] == 0


def test_replay_reads_reloaded_weights():
    """A recorded CUDA graph sees the weights of whichever adapter now sits in a slot."""
    model = make_model(max_gpu_adapters=1)
    ids, mask = make_batch()
    compiled = torch.compile(model, mode="reduce-overhead")
    outputs = {}
    for adapter in ["a0", "a1", "a0"]:  # one slot: every call evicts the previous adapter
        with torch.no_grad():
            for _ in range(3):
                outputs[adapter] = compiled(input_ids=ids, attention_mask=mask,
                                            adapter_ids=[adapter] * 4).last_hidden_state.clone()
            expected = model(input_ids=ids, attention_mask=mask, adapter_ids=[adapter] * 4).last_hidden_state
        assert masked_rel_err(outputs[adapter], expected, mask) < 2e-2, adapter
    assert masked_rel_err(outputs["a0"], outputs["a1"], mask) > 1e-2


def test_prepacked_inputs_with_weight_indices_fullgraph():
    from flashdeberta.packing import pack_inputs

    model = make_model()
    ids, mask = make_batch()
    packed = pack_inputs(ids, mask, device="cuda")
    compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    for step, adapters in enumerate([["a2", "a4", "__base__", "a3"], ["a0", "a0", "a1", "a1"]]):
        with torch.no_grad():
            expected = model(input_ids=ids, attention_mask=mask, adapter_ids=adapters).last_hidden_state
            for _ in range(3):
                weight_indices = model.resolve_adapters(adapters)
                got = compiled(**packed, weight_indices=weight_indices).last_hidden_state.clone()
        cu = packed["cu_seqlens"].tolist()
        for b in range(len(LENGTHS)):
            rows, ref = got[cu[b]:cu[b + 1]].float(), expected[b, :cu[b + 1] - cu[b]].float()
            assert ((rows - ref).norm() / ref.norm()).item() < 2e-2
        if step == 0:
            torch._dynamo.config.error_on_recompile = True
    assert counters["inductor"]["cudagraph_skips"] == 0


@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9),
                    reason="requires FP8 tensor cores")
def test_fp8_base_layers_with_adapters():
    pytest.importorskip("torchao")
    from flashdeberta.quantization import quantize

    model = make_model()
    ids, mask = make_batch()
    adapters = ["a0", "a1", "__base__", "a2"]
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask, adapter_ids=adapters).last_hidden_state
    # The base layers inside the multi-LoRA linears become FP8; adapters stay bf16.
    quantize(model, linear="fp8_rowwise", attention="fp8_qk")
    query = model.base_model.encoder.layer[0].attention.self.query_proj
    assert type(query.base_layer.weight).__name__ != "Parameter"
    compiled = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        for _ in range(3):
            got = compiled(input_ids=ids, attention_mask=mask, adapter_ids=adapters).last_hidden_state.clone()
    assert torch.isfinite(got).all()
    assert masked_rel_err(got, expected, mask) < 6e-2
    assert counters["inductor"]["cudagraph_skips"] == 0


def test_packed_layer_input_has_no_host_sync():
    """Under compile a packed input is never measured against the token count (a GPU read)."""
    model = make_model()
    ids, mask = make_batch()
    compiled = torch.compile(model.base_model.encoder.layer[0].attention.self.query_proj)
    weight_indices = model.resolve_adapters(["a0", "a1", "a2", "__base__"])
    seq_lens = mask.sum(-1).to(torch.int32)
    from polylora.context import LoraBatchContext
    ctx = LoraBatchContext(adapter_ids=None, weight_indices=weight_indices, seq_lens=seq_lens,
                           max_seq_len=L, seg_starts=(seq_lens.cumsum(0) - seq_lens).to(torch.int32))
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)  # 236 tokens + 20 filler rows
    model.adapter_cache.active_context = ctx
    try:
        with torch.no_grad():
            compiled(x)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("warn")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compiled(x)
    finally:
        torch.cuda.set_sync_debug_mode("default")
        model.adapter_cache.active_context = None
    assert not any("synchroniz" in str(w.message) for w in caught)


@pytest.mark.parametrize("compiled", [False, True])
def test_prepacked_inputs_with_adapter_ids_when_packed_size_equals_padded_size(compiled):
    """236 tokens round up to 256 packed rows = B * L: rows must still be read as packed."""
    from flashdeberta.packing import pack_inputs

    model = make_model()
    g = torch.Generator(device="cuda").manual_seed(4)
    lengths = [64, 60, 56, 56]
    ids = torch.randint(1, 200, (4, 64), device="cuda", generator=g)
    mask = (torch.arange(64, device="cuda")[None] < torch.tensor(lengths, device="cuda")[:, None]).long()
    packed = pack_inputs(ids, mask, device="cuda")
    assert packed["input_ids"].shape[0] == 4 * 64
    adapters = ["a0", "a1", "__base__", "a2"]
    run = torch.compile(model, mode="reduce-overhead") if compiled else model
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask, adapter_ids=adapters).last_hidden_state
        for _ in range(3 if compiled else 1):
            got = run(**packed, adapter_ids=adapters).last_hidden_state.clone()
    cu = packed["cu_seqlens"].tolist()
    for b in range(4):
        rows, ref = got[cu[b]:cu[b + 1]].float(), expected[b, :lengths[b]].float()
        assert ((rows - ref).norm() / ref.norm()).item() < 2e-2, b


def test_hf_deberta_mixed_adapters_raise_when_compiled():
    """HF DeBERTa projects shared position embeddings with a plain forward: mixed batches must raise
    under torch.compile as in eager, never silently give every row the first row's adapter."""
    from transformers import DebertaV2Config, DebertaV2Model

    config = DebertaV2Config(hidden_size=128, num_attention_heads=2, intermediate_size=256, num_hidden_layers=2,
                             relative_attention=True, position_buckets=64, max_relative_positions=-1,
                             pos_att_type=["p2c", "c2p"], share_att_key=True, norm_rel_ebd="layer_norm",
                             position_biased_input=False, vocab_size=200)
    torch.manual_seed(0)
    base = DebertaV2Model(config).cuda().eval()
    model = PolyLoraModel(base, PolyLoraConfig(max_gpu_adapters=3, max_rank=8,
                                               target_modules=["query_proj", "key_proj", "value_proj"])).eval()
    g = torch.Generator().manual_seed(1)
    for adapter_id in ("a0", "a1"):
        layers = {name: CpuLayerWeights(A=torch.randn(128, 4, generator=g) * 0.1,
                                        B=torch.randn(4, 128, generator=g) * 0.1, rank=4, scale=0.5)
                  for name in model.target_module_names}
        model.adapter_store.add_adapter(adapter_id, layers)
    ids, mask = make_batch()
    compiled = torch.compile(model)
    with torch.no_grad():
        single = compiled(input_ids=ids, attention_mask=mask, adapter_ids=["a1"] * 4).last_hidden_state
        expected = model(input_ids=ids, attention_mask=mask, adapter_ids=["a1"] * 4).last_hidden_state
        assert masked_rel_err(single, expected, mask) < 1e-2
        with pytest.raises(Exception, match="project_shared"):
            compiled(input_ids=ids, attention_mask=mask, adapter_ids=["a0", "a1", "a0", "a1"])


def test_varying_sequence_lengths_do_not_recompile():
    """New sequence lengths and adapter mixes reuse the graph; outputs match eager for all of them."""
    model = make_model()
    compiled = torch.compile(model, mode="reduce-overhead")
    g = torch.Generator(device="cuda").manual_seed(9)
    for step, L in enumerate([96, 40, 150, 77, 211, 64]):
        lengths = torch.randint(L // 2, L, (4,), device="cuda", generator=g)
        lengths[0] = L
        ids = torch.randint(1, 200, (4, L), device="cuda", generator=g)
        mask = (torch.arange(L, device="cuda")[None] < lengths[:, None]).long()
        adapters = [["a0", "a1", "__base__", "a2"], ["a1"] * 4][step % 2]
        with torch.no_grad():
            expected = model(input_ids=ids, attention_mask=mask, adapter_ids=adapters).last_hidden_state
            got = compiled(input_ids=ids, attention_mask=mask, adapter_ids=adapters).last_hidden_state
        assert masked_rel_err(got, expected, mask) < 2e-2, L
    # One graph per FlashDeBERTa layout at most: packed, and padded for a batch whose rounded-up packed
    # size would not be smaller than the padded batch. Lengths and adapter mixes add none.
    assert counters["stats"]["unique_graphs"] <= 2
