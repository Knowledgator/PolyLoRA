"""Packed (T, features) layer inputs: the real tokens of every row end to end, as produced by unpadding."""
import pytest
import torch
from torch import nn

from polylora import PolyLoraConfig, PolyLoraModel
from polylora.context import use_lora_context
from polylora.store import CpuLayerWeights

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
IN, OUT = 48, 40


class TwoLinear(nn.Module):
    """Stand-in base model; only the wrapped linears matter here."""

    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(IN, OUT)
        self.dense = nn.Linear(OUT, IN)

    def forward(self, input_ids=None, attention_mask=None):
        return self.dense(self.query_proj(input_ids))


def make_model(device, use_triton):
    torch.manual_seed(0)
    model = PolyLoraModel(
        TwoLinear().to(device),
        PolyLoraConfig(max_gpu_adapters=3, max_rank=8, target_modules=["query_proj", "dense"],
                       use_triton_kernels=use_triton),
    ).eval()
    g = torch.Generator().manual_seed(1)
    for adapter_id, rank in [("a", 4), ("b", 8)]:
        layers = {}
        for name in ["query_proj", "dense"]:
            linear = model.base_model.get_submodule(name).base_layer
            layers[name] = CpuLayerWeights(A=torch.randn(linear.in_features, rank, generator=g),
                                           B=torch.randn(rank, linear.out_features, generator=g), rank=rank, scale=0.5)
        model.adapter_store.add_adapter(adapter_id, layers)
    return model


def padded_and_packed(model, lengths, adapter_ids, device):
    """Run the wrapped query_proj on a padded batch and on the same tokens packed."""
    B, L = len(lengths), max(lengths) + 3
    mask = (torch.arange(L, device=device)[None] < torch.tensor(lengths, device=device)[:, None]).long()
    g = torch.Generator(device=device).manual_seed(2)
    x = torch.randn(B, L, IN, device=device, generator=g)
    ctx = model.prepare_lora_context(adapter_ids, mask, B, L, device)
    layer = model.base_model.query_proj
    keep = mask.bool()
    with torch.no_grad(), use_lora_context(ctx):
        padded = layer(x)[keep]
        packed = layer(x[keep])  # (T, IN), rows in order, padding removed
    return padded, packed


DEVICES = [pytest.param("cuda", True, marks=cuda, id="triton"), pytest.param("cpu", False, id="reference")]


@pytest.mark.parametrize("device,use_triton", DEVICES)
@pytest.mark.parametrize("lengths,adapter_ids", [
    ([30, 21, 7], ["a", "b", "__base__"]),
    ([30, 20, 10], ["b", "a", "b"]),          # T divisible by B: used to be reshaped silently
    ([1, 64, 33, 17], ["a", "__base__", "b", "a"]),
])
def test_packed_matches_padded(device, use_triton, lengths, adapter_ids):
    model = make_model(device, use_triton)
    padded, packed = padded_and_packed(model, lengths, adapter_ids, device)
    assert packed.shape == padded.shape
    assert torch.allclose(packed, padded, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device,use_triton", DEVICES)
def test_packed_lora_is_applied_per_row(device, use_triton):
    model = make_model(device, use_triton)
    lengths = [12, 5, 9]
    _, mixed = padded_and_packed(model, lengths, ["a", "__base__", "b"], device)
    _, base_only = padded_and_packed(model, lengths, ["__base__"] * 3, device)
    mixed_rows, base_rows = torch.split(mixed, lengths), torch.split(base_only, lengths)
    # Adapter rows change; the __base__ row is the plain linear.
    assert not torch.allclose(mixed_rows[0], base_rows[0])
    assert torch.allclose(mixed_rows[1], base_rows[1])
    assert not torch.allclose(mixed_rows[2], base_rows[2])


def test_unknown_layout_raises():
    model = make_model("cpu", False)
    mask = torch.ones(3, 10, dtype=torch.long)
    mask[1, 6:] = 0
    ctx = model.prepare_lora_context(["a", "b", "a"], mask, 3, 10, torch.device("cpu"))
    with use_lora_context(ctx), pytest.raises(ValueError, match="Cannot map layer input"):
        model.base_model.query_proj(torch.randn(12, IN))  # neither B, B * L nor T rows


@cuda
def test_flashdeberta_packed_mode_matches_padded_mode():
    flashdeberta = pytest.importorskip("flashdeberta.model")
    from transformers import DebertaV2Config

    config = DebertaV2Config(
        hidden_size=64, num_attention_heads=2, intermediate_size=128, num_hidden_layers=2,
        relative_attention=True, position_buckets=64, max_relative_positions=-1, pos_att_type=["p2c", "c2p"],
        share_att_key=False, norm_rel_ebd="layer_norm", position_biased_input=False, vocab_size=200,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    torch.manual_seed(0)
    base = flashdeberta.FlashDebertaV2Model(config).cuda().eval()
    if not hasattr(flashdeberta, "PackedBatch"):
        pytest.skip("installed flashdeberta has no packed mode")
    targets = ["query_proj", "key_proj", "value_proj", "dense"]
    model = PolyLoraModel(base, PolyLoraConfig(max_gpu_adapters=3, max_rank=8, target_modules=targets)).eval()
    g = torch.Generator().manual_seed(1)
    for adapter_id, rank in [("a", 4), ("b", 8)]:
        layers = {}
        for name in model.target_module_names:
            linear = base.get_submodule(name).base_layer
            layers[name] = CpuLayerWeights(A=torch.randn(linear.in_features, rank, generator=g) * 0.1,
                                           B=torch.randn(rank, linear.out_features, generator=g) * 0.1,
                                           rank=rank, scale=2.0)
        model.adapter_store.add_adapter(adapter_id, layers)

    # The second batch has T divisible by B.
    for lengths in ([40, 25, 9], [30, 20, 10]):
        L = max(lengths)
        ids = torch.randint(1, 200, (len(lengths), L), device="cuda")
        mask = (torch.arange(L, device="cuda")[None] < torch.tensor(lengths, device="cuda")[:, None]).long()
        outputs = {}
        for unpad in (True, False):
            base.config.unpad_inputs = unpad
            with torch.no_grad():
                outputs[unpad] = model(input_ids=ids, attention_mask=mask, adapter_ids=["a", "b", "__base__"]).last_hidden_state
        keep = mask[..., None].bool()
        assert torch.allclose(outputs[True] * keep, outputs[False] * keep, atol=1e-4)


@pytest.mark.parametrize("device,use_triton", DEVICES)
def test_one_vector_per_row_stays_in_its_row(device, use_triton):
    """(B, features) inputs (poolers, heads): each row gets only its own adapter, even when rows are long."""
    model = make_model(device, use_triton)
    B, L = 4, 40
    mask = torch.ones(B, L, dtype=torch.long, device=device)
    x = torch.randn(B, IN, device=device, generator=torch.Generator(device=device).manual_seed(3))
    layer = model.base_model.query_proj
    with torch.no_grad():
        with use_lora_context(model.prepare_lora_context(["a", "__base__", "b", "a"], mask, B, L, device)):
            mixed = layer(x)
        plain = layer.base_layer(x)
        for i, adapter_id in enumerate(["a", "__base__", "b", "a"]):
            with use_lora_context(model.prepare_lora_context([adapter_id], mask[:1], 1, L, device)):
                solo = layer(x[i:i + 1])
            assert torch.allclose(mixed[i:i + 1], solo, atol=1e-5, rtol=1e-5)
    assert torch.equal(mixed[1], plain[1])


@cuda
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-2), (torch.float16, 2e-2)])
def test_sgmv_kernels_match_reference_on_segments(dtype, tol):
    """Segments with gaps between them, mixed ranks and a rank-0 adapter; rows outside segments are untouched.

    tl.dot runs fp32 inputs in TF32, hence the loose fp32 tolerance.
    """
    from polylora.sgmv import sgmv_lora_a_fwd, sgmv_lora_b_fwd

    torch.manual_seed(0)
    K, N, R = 96, 300, 16
    ranks = torch.tensor([0, 16, 4, 9], dtype=torch.int32, device="cuda")
    scales = torch.tensor([0.0, 0.5, 2.0, 1.0], dtype=torch.float32, device="cuda")
    A = torch.randn(4, K, R, device="cuda", dtype=dtype) * 0.1
    Bw = torch.randn(4, R, N, device="cuda", dtype=dtype) * 0.1
    for slot in range(4):  # unused rank columns hold garbage that must be ignored
        A[slot, :, ranks[slot]:] = 100
        Bw[slot, ranks[slot]:] = 100
    seq_lens = torch.tensor([17, 0, 40, 1, 33], dtype=torch.int32, device="cuda")
    seg_starts = torch.tensor([3, 20, 25, 70, 80], dtype=torch.int32, device="cuda")
    slots = torch.tensor([1, 2, 2, 3, 0], dtype=torch.int64, device="cuda")
    T = 120
    x = torch.randn(T, K, device="cuda", dtype=dtype)
    base = torch.randn(T, N, device="cuda", dtype=dtype)

    expected = base.clone().float()
    for start, n, slot in zip(seg_starts.tolist(), seq_lens.tolist(), slots.tolist()):
        r = int(ranks[slot])
        if n and r:
            rows = slice(start, start + n)
            expected[rows] += x[rows].float() @ A[slot, :, :r].float() @ Bw[slot, :r].float() * scales[slot]

    tmp = sgmv_lora_a_fwd(x, A, seq_lens, seg_starts, slots, ranks, max_seq_len=40)
    out = sgmv_lora_b_fwd(tmp, Bw, seq_lens, seg_starts, slots, ranks, scales, base.clone())
    assert torch.allclose(out.float(), expected, atol=tol, rtol=tol)


def test_shared_input_gets_one_output_per_adapter():
    """project_shared: (1, M, in) -> one projection per distinct adapter and each row's entry."""
    model = make_model("cpu", False)
    layer = model.base_model.query_proj
    x = torch.randn(1, 7, IN)
    mask = torch.ones(4, 5, dtype=torch.long)
    ids = ["b", "__base__", "b", "a"]
    with torch.no_grad(), use_lora_context(model.prepare_lora_context(ids, mask, 4, 5, torch.device("cpu"))):
        out, index = layer.project_shared(x)
        assert out.shape == (3, 7, OUT) and index.tolist() == [0, 1, 0, 2]
        for row, adapter_id in enumerate(ids):
            with use_lora_context(model.prepare_lora_context([adapter_id], mask[:1], 1, 5, torch.device("cpu"))):
                solo = layer(x)  # one row: the whole shared input gets that row's adapter
            assert torch.allclose(out[index[row]], solo[0], atol=1e-5)
        # Mixed adapters cannot share one output through the plain forward.
        with pytest.raises(ValueError, match="project_shared"):
            layer(x)
    with torch.no_grad(), use_lora_context(model.prepare_lora_context(["a", "a"], mask[:2], 2, 5, torch.device("cpu"))):
        out, index = layer.project_shared(x)
        assert out.shape == (1, 7, OUT) and index is None
        assert torch.allclose(layer(x), out, atol=1e-5)  # single adapter: every shared row gets it


FLASH_CASES = [
    pytest.param(share, unpad, use_triton, id=f"share={share}-unpad={unpad}-{'triton' if use_triton else 'reference'}")
    for share in (True, False) for unpad in (True, False) for use_triton in (False, True)
]


@cuda
@pytest.mark.parametrize("share_att_key,unpad,use_triton", FLASH_CASES)
def test_flashdeberta_rows_match_merged_adapters(share_att_key, unpad, use_triton):
    """Every row of a mixed batch equals FlashDeBERTa with that row's adapter merged into the weights.

    With share_att_key the relative position embeddings go through the LoRA query/key projections,
    so each row also needs its own adapter's position tables.
    """
    import copy

    flashdeberta = pytest.importorskip("flashdeberta.model")
    from transformers import DebertaV2Config

    config = DebertaV2Config(
        hidden_size=64, num_attention_heads=2, intermediate_size=128, num_hidden_layers=2,
        relative_attention=True, position_buckets=64, max_relative_positions=-1, pos_att_type=["p2c", "c2p"],
        share_att_key=share_att_key, norm_rel_ebd="layer_norm", position_biased_input=False, vocab_size=200,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    torch.manual_seed(0)
    base = flashdeberta.FlashDebertaV2Model(config).cuda().eval()
    plain = copy.deepcopy(base)
    targets = ["query_proj", "key_proj", "value_proj", "dense"]
    model = PolyLoraModel(base, PolyLoraConfig(max_gpu_adapters=3, max_rank=8, target_modules=targets,
                                               use_triton_kernels=use_triton)).eval()
    g = torch.Generator().manual_seed(1)
    adapters = {}
    for adapter_id, rank in [("a", 4), ("b", 8)]:
        adapters[adapter_id] = {
            name: CpuLayerWeights(A=torch.randn(base.get_submodule(name).base_layer.in_features, rank, generator=g) * 0.1,
                                  B=torch.randn(rank, base.get_submodule(name).base_layer.out_features, generator=g) * 0.1,
                                  rank=rank, scale=2.0)
            for name in model.target_module_names
        }
        model.adapter_store.add_adapter(adapter_id, adapters[adapter_id])

    def merged(adapter_id):
        m = copy.deepcopy(plain)
        m.config.unpad_inputs = unpad
        for name, w in adapters.get(adapter_id, {}).items():
            m.get_submodule(name).weight.data += (w.scale * (w.A @ w.B)).T.cuda()
        return m

    lengths, adapter_ids = [9, 40, 25, 33], ["a", "b", "__base__", "a"]
    L = max(lengths)
    ids = torch.randint(1, 200, (len(lengths), L), device="cuda")
    mask = (torch.arange(L, device="cuda")[None] < torch.tensor(lengths, device="cuda")[:, None]).long()
    base.config.unpad_inputs = unpad
    # The Triton LoRA kernels run fp32 matmuls in TF32.
    atol = 1e-2 if use_triton else 1e-4
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, adapter_ids=adapter_ids).last_hidden_state
        for row, (n, adapter_id) in enumerate(zip(lengths, adapter_ids)):
            expected = merged(adapter_id)(input_ids=ids[row:row + 1], attention_mask=mask[row:row + 1])
            assert torch.allclose(out[row, :n], expected.last_hidden_state[0, :n], atol=atol), (row, adapter_id)
