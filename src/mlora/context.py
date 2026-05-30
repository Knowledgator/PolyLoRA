from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class LoraBatchContext:
    adapter_ids: list[str]
    weight_indices: torch.Tensor
    seq_lens: torch.Tensor
    attention_mask: torch.Tensor | None = None


_CURRENT_LORA_CONTEXT: ContextVar[LoraBatchContext | None] = ContextVar(
    "mlora_current_lora_context", default=None
)


def get_lora_context() -> LoraBatchContext | None:
    return _CURRENT_LORA_CONTEXT.get()


@contextmanager
def use_lora_context(ctx: LoraBatchContext | None) -> Iterator[None]:
    token = _CURRENT_LORA_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _CURRENT_LORA_CONTEXT.reset(token)


def assert_right_padded(attention_mask: torch.Tensor, seq_lens: torch.Tensor) -> None:
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must be a 2D batch mask")

    _, max_len = attention_mask.shape
    positions = torch.arange(max_len, device=attention_mask.device).expand_as(attention_mask)
    expected = positions < seq_lens[:, None]
    if not torch.equal(attention_mask.bool(), expected):
        raise ValueError("Custom multi-LoRA runtime currently requires right-padded attention_mask")
