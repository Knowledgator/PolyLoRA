"""Mixed-adapter batches vs. merged-weight references across models, adapter counts and batch shapes.

Each case wraps a small encoder with PolyLoRA, runs a batch where rows pick random adapters (or
``__base__``), and checks every row against the same encoder with that row's adapter merged into its
weights (W + scale * (A @ B)^T), which is what PEFT's merge produces. Adapters differ in rank, scale and
which of the wrapped layers they cover.
"""
import copy

import pytest
import torch

from polylora import PolyLoraConfig, PolyLoraModel
from polylora.store import CpuLayerWeights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HIDDEN, HEADS, LAYERS, VOCAB = 64, 2, 2, 200
BUCKETS = 32  # relative-position span; long sequences go past it
RANKS = [1, 2, 4, 8, 16]

# backend -> (builder, target modules)
BACKENDS = ["flash-unpad", "flash-padded", "flash-noshare", "hf-deberta", "bert"]
ADAPTER_COUNTS = [1, 3, 6]
SHAPES = [  # (batch size, longest sequence)
    pytest.param(1, 17, id="B1-L17"),
    pytest.param(4, 16, id="B4-L16"),
    pytest.param(8, 128, id="B8-L128"),
    pytest.param(16, 300, id="B16-L300"),
]
PATHS = [pytest.param(False, id="reference"), pytest.param(True, id="triton")]
# Relative L2 error over real tokens. The Triton LoRA kernels run fp32 matmuls in TF32.
TOLERANCE = {False: 1e-4, True: 5e-3}


def deberta_config(share_att_key):
    from transformers import DebertaV2Config

    config = DebertaV2Config(
        hidden_size=HIDDEN, num_attention_heads=HEADS, intermediate_size=2 * HIDDEN, num_hidden_layers=LAYERS,
        relative_attention=True, position_buckets=BUCKETS, max_relative_positions=-1, pos_att_type=["p2c", "c2p"],
        share_att_key=share_att_key, norm_rel_ebd="layer_norm", position_biased_input=False, vocab_size=VOCAB,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    config._attn_implementation = "eager"
    return config


def make_base(backend):
    """Returns (model, target modules, configure(model)) for a backend; weights are seeded."""
    torch.manual_seed(0)
    deberta_targets = ["query_proj", "key_proj", "value_proj", "dense"]
    if backend.startswith("flash"):
        flashdeberta = pytest.importorskip("flashdeberta.model")
        model = flashdeberta.FlashDebertaV2Model(deberta_config(share_att_key=backend != "flash-noshare"))
        unpad = backend != "flash-padded"

        def configure(m):
            m.config.unpad_inputs = unpad
        return model, deberta_targets, configure
    if backend == "hf-deberta":
        from transformers import DebertaV2Model
        # HF DeBERTa projects shared position embeddings with a plain forward, which only
        # supports one adapter per batch when share_att_key=True (see test below).
        return DebertaV2Model(deberta_config(share_att_key=False)), deberta_targets, lambda m: None
    if backend == "bert":
        from transformers import BertConfig, BertModel

        config = BertConfig(hidden_size=HIDDEN, num_attention_heads=HEADS, num_hidden_layers=LAYERS,
                            intermediate_size=2 * HIDDEN, vocab_size=VOCAB, max_position_embeddings=512,
                            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        config._attn_implementation = "eager"
        return BertModel(config, add_pooling_layer=False), ["query", "key", "value", "dense"], lambda m: None
    raise ValueError(backend)


def make_adapters(model, targets, count, seed):
    """`count` adapters with random rank and scale, each covering a random subset of the target layers."""
    g = torch.Generator().manual_seed(seed)
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and any(n == t or n.endswith(f".{t}") for t in targets)]
    adapters = {}
    for i in range(count):
        rank = RANKS[int(torch.randint(len(RANKS), (1,), generator=g))]
        scale = float(torch.empty(1).uniform_(0.5, 2.0, generator=g))
        keep = torch.rand(len(names), generator=g) < 0.75
        keep[int(torch.randint(len(names), (1,), generator=g))] = True  # never empty
        layers = {}
        for name, use in zip(names, keep.tolist()):
            if not use:
                continue
            linear = model.get_submodule(name)
            layers[name] = CpuLayerWeights(
                A=torch.randn(linear.in_features, rank, generator=g) / linear.in_features ** 0.5,
                B=torch.randn(rank, linear.out_features, generator=g) * 0.5,
                rank=rank, scale=scale,
            )
        adapters[f"adapter{i}"] = layers
    return adapters


def merge(plain, layers):
    merged = copy.deepcopy(plain)
    for name, w in layers.items():
        linear = merged.get_submodule(name)
        linear.weight.data += (w.scale * (w.A @ w.B)).T.to(linear.weight)
    return merged


def make_batch(batch_size, max_len, adapter_names, seed):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, max_len + 1, (batch_size,), generator=g)
    lengths[0] = max_len
    if batch_size > 2:
        lengths[-1] = 1
    choices = ["__base__"] + adapter_names
    rows = torch.randint(len(choices), (batch_size,), generator=g).tolist()
    adapter_ids = [choices[i] for i in rows]
    # Every adapter appears when the batch has room, and at least one row uses an adapter.
    for i, name in enumerate(adapter_names[: max(batch_size - 1, 1)]):
        adapter_ids[(i + 1) % batch_size] = name
    ids = torch.randint(1, VOCAB, (batch_size, max_len), generator=g).cuda()
    mask = (torch.arange(max_len)[None] < lengths[:, None]).long().cuda()
    return ids, mask, adapter_ids


def relative_error(got, expected, mask):
    keep = mask.bool()[..., None]
    diff = (got - expected).masked_fill(~keep, 0).double().norm()
    return (diff / expected.masked_fill(~keep, 0).double().norm()).item()


def check_rows(model, plain, adapters, configure, ids, mask, adapter_ids, tolerance):
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, adapter_ids=adapter_ids).last_hidden_state
        base_out = plain(input_ids=ids, attention_mask=mask).last_hidden_state
        for adapter_id in dict.fromkeys(adapter_ids):
            rows = torch.tensor([i for i, a in enumerate(adapter_ids) if a == adapter_id], device=ids.device)
            reference = merge(plain, adapters[adapter_id]) if adapter_id != "__base__" else plain
            configure(reference)
            expected = reference(input_ids=ids[rows], attention_mask=mask[rows]).last_hidden_state
            err = relative_error(out[rows], expected, mask[rows])
            assert err < tolerance, f"{adapter_id} rows {rows.tolist()}: relative error {err:.2e}"
            if adapter_id != "__base__":  # the adapter must actually change the output
                assert relative_error(base_out[rows], expected, mask[rows]) > 3 * tolerance


@pytest.fixture(autouse=True)
def exact_fp32_matmuls():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("use_triton", PATHS)
@pytest.mark.parametrize("batch_size,max_len", SHAPES)
@pytest.mark.parametrize("num_adapters", ADAPTER_COUNTS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_batch_matches_merged_adapters(backend, num_adapters, batch_size, max_len, use_triton):
    plain, targets, configure = make_base(backend)
    plain = plain.cuda().eval()
    configure(plain)
    adapters = make_adapters(plain, targets, num_adapters, seed=num_adapters)
    model = PolyLoraModel(copy.deepcopy(plain), PolyLoraConfig(
        max_gpu_adapters=max(num_adapters, 1), max_rank=max(RANKS), target_modules=targets,
        use_triton_kernels=use_triton,
    )).eval()
    for adapter_id, layers in adapters.items():
        model.adapter_store.add_adapter(adapter_id, layers)
    ids, mask, adapter_ids = make_batch(batch_size, max_len, list(adapters), seed=batch_size * 1000 + max_len)
    check_rows(model, plain, adapters, configure, ids, mask, adapter_ids, TOLERANCE[use_triton])


@pytest.mark.parametrize("backend", ["flash-unpad", "flash-padded"])
def test_fp16_mixed_batch(backend):
    plain, targets, configure = make_base(backend)
    plain = plain.cuda().half().eval()
    configure(plain)
    adapters = make_adapters(plain, targets, 6, seed=7)
    model = PolyLoraModel(copy.deepcopy(plain), PolyLoraConfig(
        max_gpu_adapters=6, max_rank=max(RANKS), target_modules=targets)).eval()
    for adapter_id, layers in adapters.items():
        model.adapter_store.add_adapter(adapter_id, layers)
    ids, mask, adapter_ids = make_batch(16, 300, list(adapters), seed=11)
    check_rows(model, plain, adapters, configure, ids, mask, adapter_ids, tolerance=2e-2)


@pytest.mark.parametrize("backend", ["flash-unpad", "bert"])
def test_adapter_swapping_with_small_gpu_cache(backend):
    """More adapters than GPU slots: successive batches evict and reload slots, results stay exact."""
    plain, targets, configure = make_base(backend)
    plain = plain.cuda().eval()
    configure(plain)
    adapters = make_adapters(plain, targets, 5, seed=3)
    model = PolyLoraModel(copy.deepcopy(plain), PolyLoraConfig(
        max_gpu_adapters=2, max_rank=max(RANKS), target_modules=targets)).eval()
    for adapter_id, layers in adapters.items():
        model.adapter_store.add_adapter(adapter_id, layers)
    names = list(adapters)
    for step, pair in enumerate([names[0:2], names[2:4], [names[4], names[0]], names[1:3], [names[3], names[4]]]):
        ids, mask, _ = make_batch(6, 40, pair, seed=step)
        adapter_ids = [pair[0], "__base__", pair[1], pair[0], pair[1], "__base__"]
        check_rows(model, plain, adapters, configure, ids, mask, adapter_ids, TOLERANCE[True])


def test_batch_needing_more_adapters_than_gpu_slots_raises():
    plain, targets, configure = make_base("bert")
    adapters = make_adapters(plain, targets, 3, seed=0)
    model = PolyLoraModel(plain.cuda().eval(), PolyLoraConfig(max_gpu_adapters=2, max_rank=16, target_modules=targets))
    for adapter_id, layers in adapters.items():
        model.adapter_store.add_adapter(adapter_id, layers)
    ids, mask, _ = make_batch(3, 8, [], seed=0)
    with pytest.raises(RuntimeError, match="GPU cache capacity"):
        model(input_ids=ids, attention_mask=mask, adapter_ids=["adapter0", "adapter1", "adapter2"])


def test_hf_deberta_shared_position_projection():
    """HF DeBERTa with share_att_key projects position embeddings with a plain forward: one adapter per
    batch is exact, mixed adapters raise instead of silently giving rows the wrong adapter."""
    from transformers import DebertaV2Model

    torch.manual_seed(0)
    plain = DebertaV2Model(deberta_config(share_att_key=True)).cuda().eval()
    targets = ["query_proj", "key_proj", "value_proj", "dense"]
    adapters = make_adapters(plain, targets, 2, seed=5)
    model = PolyLoraModel(copy.deepcopy(plain), PolyLoraConfig(max_gpu_adapters=2, max_rank=16,
                                                               target_modules=targets)).eval()
    for adapter_id, layers in adapters.items():
        model.adapter_store.add_adapter(adapter_id, layers)
    ids, mask, _ = make_batch(5, 50, [], seed=1)
    check_rows(model, plain, adapters, lambda m: None, ids, mask, ["adapter1"] * 5, TOLERANCE[True])
    check_rows(model, plain, adapters, lambda m: None, ids, mask, ["__base__"] * 5, TOLERANCE[True])
    # __base__ counts as an adapter of its own: mixing it with adapter rows also needs per-row tables.
    for mixed in (["adapter0", "adapter1"] * 2 + ["adapter0"], ["adapter0", "__base__"] * 2 + ["adapter0"]):
        with pytest.raises(ValueError, match="project_shared"), torch.no_grad():
            model(input_ids=ids, attention_mask=mask, adapter_ids=mixed)
