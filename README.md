# PolyLoRA

Minimal PyTorch runtime for batched LoRA inference where each row can use a different adapter.

PolyLoRA wraps an existing `torch.nn.Module`, replaces selected `nn.Linear` layers, and serves PEFT LoRA adapters from CPU, GPU, and optional disk caches.

## Install

```bash
pip install .
```

With PEFT loading support:

```bash
pip install '.[peft]'
```

## Usage

```python
from polylora import PolyLoraConfig, PolyLoraModel

model = PolyLoraModel(
    base_model,
    PolyLoraConfig(
        max_gpu_adapters=4,
        max_rank=16,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
    ),
).eval()

model.load_adapter_from_disk("legal", "./adapters/legal")
model.load_adapter_from_disk("finance", "./adapters/finance")

outputs = model(**batch, adapter_ids=["legal", "finance"])
```

Omit `adapter_ids` to run the base model. Use `__base__` for rows that should skip LoRA inside a mixed batch.

## Caches

PolyLoRA uses three adapter tiers:

- GPU cache: fixed-size adapter slots for the active batch. Slot `0` is reserved for `__base__`, so non-adapter rows share the same execution path.
- CPU cache: LRU store for loaded adapter weights. GPU evictions can reload from CPU without touching disk.
- Disk cache: optional bounded PEFT adapter directory cache. CPU misses can reload adapters from this cold layer.

This makes small hot sets fast while still allowing a larger adapter catalog than GPU memory can hold.

## Kernels

On CUDA, PolyLoRA uses Triton SGMV kernels for the LoRA `A` and `B` projections:

- Mixed batches can contain different adapter ids, including `__base__` rows.
- Different adapters may use different ranks, up to `max_rank`.
- Rank-0 rows skip adapter work, which is how base-only rows and missing layer weights are represented.
- The `B` projection fuses scaling and add-back into the base linear output.
- The implementation falls back to a PyTorch reference path on CPU or when Triton is disabled.

## Adapter Layouts

Adapters do not need to cover every wrapped layer. If a model is wrapped with a larger `target_modules` set and an adapter only contains LoRA weights for some of those layers, missing layers are treated as rank-0 no-ops for that adapter.

PolyLoRA rejects adapters with weights outside the configured module set, which keeps mixed adapters predictable when different adapters target different subsets of the model.

## Notes

- Supports standard PEFT LoRA adapters for inference.
- Does not support LoRA dropout, DoRA, RS-LoRA, or LoRA bias.
- Attention masks must be right padded when `enforce_right_padding=True`.

## Development

```bash
pip install -e '.[dev]'
pytest tests
```
