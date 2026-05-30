# mlora

`mlora` is a small PyTorch runtime for serving batches where each row can use a different LoRA adapter. It wraps an existing `torch.nn.Module`, replaces selected `nn.Linear` layers with multi-adapter LoRA layers, and keeps adapter weights in a CPU store with a bounded GPU cache.

The package can load PEFT LoRA adapters from an in-memory PEFT model or from PEFT adapter directories on disk.

## Installation

From this repository:

```bash
pip install ./mlora
```

For PEFT adapter loading examples and tests:

```bash
pip install './mlora[peft]'
```

## Quick start

```python
import torch
from transformers import AutoModel, AutoTokenizer

from mlora import CustomLoraConfig, CustomPeftModel

base_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").eval().to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

model = CustomPeftModel(
    base_model,
    CustomLoraConfig(
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        max_gpu_adapters=4,
        max_rank=16,
    ),
).eval()

model.load_adapter_from_disk("legal", "./adapters/legal")
model.load_adapter_from_disk("finance", "./adapters/finance")

batch = tokenizer(
    ["contract text", "earnings report"],
    padding=True,
    return_tensors="pt",
).to("cuda")

with torch.inference_mode():
    outputs = model(**batch, adapter_ids=["legal", "finance"])
```

Use `adapter_ids=None` or omit `adapter_ids` to run the base model without LoRA. Use the configured base adapter id, `__base__` by default, for rows that should skip LoRA inside a mixed batch.

## API

- `CustomLoraConfig`: runtime configuration for target modules, cache sizes, maximum rank, padding checks, and Triton kernel usage.
- `CustomPeftModel`: wraps a base model and handles per-row adapter dispatch.
- `CpuAdapterStore`: CPU-side adapter storage with optional eviction.
- `GpuAdapterCache`: GPU-side adapter cache used by `CustomPeftModel`.
- `MultiLoraLinear`: replacement linear layer that applies the active row-level LoRA context.
- `LoraBatchContext`: explicit context object for lower-level integrations.

## Adapter requirements

`mlora` supports standard PEFT LoRA adapters for inference. The current runtime does not support LoRA dropout, DoRA, RS-LoRA, or LoRA bias. Attention masks must be right padded when `enforce_right_padding=True`.

Triton kernels are used on CUDA when available. The implementation falls back to a PyTorch reference path on CPU or when `use_triton_kernels=False`.

## Development

Run the package tests from this repository root:

```bash
pytest mlora/tests
```

The tests use the local `peft/` checkout when present.
