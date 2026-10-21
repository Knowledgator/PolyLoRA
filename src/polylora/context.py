from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterator

import torch


@dataclass(frozen=True)
class LoraBatchContext:
    adapter_ids: list[str]
    weight_indices: torch.Tensor
    seq_lens: torch.Tensor
    attention_mask: torch.Tensor | None = None
    # Padded sequence length of the batch; bounds every row's length.
    max_seq_len: int | None = None
    # First row of each batch in a packed (T, ...) layer input: the real tokens of every
    # row laid end to end in row order, as produced by unpadding with the attention mask.
    seg_starts: torch.Tensor | None = None
    # Per-forward memo shared by every layer (padded segments, adapter groups).
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def padded_segments(self, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(seq_lens, seg_starts) of a (B, width, ...) layer input viewed as (B * width, ...).

        Only a token-aligned input (``width`` equal to the padded length) skips padding; any other
        per-row input, e.g. one pooled vector (width 1) or span features, gets LoRA on all ``width``
        entries, as PEFT does. Cached per width: every layer of a forward pass reuses them.
        """
        key = ("padded", width)
        cached = self._cache.get(key)
        if cached is None:
            batch_size = self.seq_lens.shape[0]
            if self.max_seq_len is None or width == self.max_seq_len:
                seq_lens = torch.clamp(self.seq_lens, max=width).to(torch.int32)
            else:
                seq_lens = torch.full_like(self.seq_lens, width, dtype=torch.int32)
            seg_starts = torch.arange(batch_size, device=self.seq_lens.device, dtype=torch.int32) * width
            cached = self._cache[key] = (seq_lens, seg_starts)
        return cached

    def adapter_groups(self, adapter_to_slot: dict[str, int]) -> tuple[list[int], torch.Tensor | None]:
        """Distinct adapter slots of the batch and each row's position among them.

        Returns ``(slots, index)``: ``slots`` lists each distinct slot once in first-use order and
        ``index`` (B,) maps every row to its entry, or is None when all rows share one adapter.
        Built from host-side ids, so it needs no GPU sync.
        """
        cached = self._cache.get("groups")
        if cached is None:
            row_slots = [adapter_to_slot.get(adapter_id, 0) for adapter_id in self.adapter_ids]
            slots = list(dict.fromkeys(row_slots))
            index = None
            if len(slots) > 1:
                position = {slot: i for i, slot in enumerate(slots)}
                index = torch.tensor([position[slot] for slot in row_slots], dtype=torch.long,
                                     device=self.weight_indices.device)
            cached = self._cache["groups"] = (slots, index)
        return cached

    @cached_property
    def num_tokens(self) -> int:
        """Real tokens in the batch (T of a packed input). Reads one value back from the GPU, once."""
        return int(self.seq_lens.sum())


_CURRENT_LORA_CONTEXT: ContextVar[LoraBatchContext | None] = ContextVar(
    "polylora_current_lora_context", default=None
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
