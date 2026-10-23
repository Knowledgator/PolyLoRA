from __future__ import annotations

import math

import torch
from torch import nn

from .cache import GpuAdapterCache
from .context import get_lora_context


try:
    from .sgmv import sgmv_lora_a as _SGMV_LORA_A
    from .sgmv import sgmv_lora_b_ as _SGMV_LORA_B_
except Exception:
    _SGMV_LORA_A = None
    _SGMV_LORA_B_ = None


class MultiLoraLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, module_name: str, cache: GpuAdapterCache) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.module_name = module_name
        self.cache = cache

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def _context(self):
        # PolyLoraModel.forward publishes the batch on the shared cache (traceable by torch.compile);
        # code that sets a context with `use_lora_context` reaches the layers through the ContextVar.
        # The cache attribute is shared by all threads: one PolyLoraModel runs one forward at a time.
        lora_ctx = self.cache.active_context
        if lora_ctx is None and not torch.compiler.is_compiling():
            # Not under torch.compile: a ContextVar read is not traceable (a graph break per layer).
            lora_ctx = get_lora_context()
        return lora_ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora_ctx = self._context()
        if lora_ctx is None:
            return base

        original_shape = base.shape
        x_in, base_in, seq_lens, seg_starts, weight_indices, max_seq_len = self._to_segments(x, base, lora_ctx)

        if self._can_use_triton(x_in):
            tmp = _SGMV_LORA_A(
                x_in,
                self.cache.A[self.module_name],
                seq_lens,
                seg_starts,
                weight_indices,
                self.cache.ranks[self.module_name],
                max_seq_len,
            )
            out = base_in  # updated in place: base + scale * (x A) B on every segment row
            _SGMV_LORA_B_(
                tmp,
                self.cache.B[self.module_name],
                seq_lens,
                seg_starts,
                weight_indices,
                self.cache.ranks[self.module_name],
                self.cache.scales[self.module_name],
                out,
            )
        else:
            out = self._reference_forward(x_in, base_in, seq_lens, seg_starts, weight_indices)

        return out.reshape(original_shape)

    @staticmethod
    def _is_packed(x: torch.Tensor, lora_ctx) -> bool:
        """Is ``x`` a packed (T, features) input: the real tokens of all rows, end to end?

        A packed input may hold filler rows after the real tokens (a fixed-size packed batch, as
        FlashDeBERTa builds under torch.compile); they are outside every segment and stay as they are.
        """
        if x.dim() != 2 or lora_ctx.seg_starts is None:
            return False
        batch_size = lora_ctx.weight_indices.shape[0]
        if x.shape[0] == batch_size:
            return False  # one vector per row (e.g. a pooler)
        if lora_ctx.packed:
            # Pre-packed model inputs: token rows are packed even when their count happens to equal
            # B * max_seq_len (a rounded-up packed size can).
            return True
        padded_len = lora_ctx.padded_len
        if padded_len is not None and x.shape[0] == batch_size * padded_len:
            return False  # flattened padded batch; equals the packed layout when there is no padding
        if torch.compiler.is_compiling():
            # The token count must not be read from the GPU here: any other row count is a packed
            # batch of up to that many rows (FlashDeBERTa keeps a packed batch below B * L rows).
            return True
        # Eager: the real tokens, possibly followed by filler rows (a packed batch rounded up to a
        # bucket, as FlashDeBERTa does under torch.compile), and fewer rows than the padded batch.
        return x.shape[0] >= lora_ctx.num_tokens and (padded_len is None or x.shape[0] < batch_size * padded_len)

    def project_shared(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project an input shared by every row, once per distinct adapter in the batch.

        ``x`` is (1, ..., in_features), e.g. DeBERTa's relative position embeddings. Returns
        ``(out, index)``: ``out`` is (S, ..., out_features), the projection under each of the S
        distinct adapters of the batch, and ``index`` (B,) picks each row's entry, or is None when
        every row uses the same adapter (S == 1).
        """
        if x.shape[0] != 1:
            raise ValueError(f"project_shared expects a (1, ..., features) input, got {tuple(x.shape)}")
        base = self.base_layer(x)
        lora_ctx = self._context()
        if lora_ctx is None:
            return base, None
        if torch.compiler.is_compiling():
            return self._project_shared_static(x, base, lora_ctx)
        slots, index = lora_ctx.adapter_groups(self.cache.adapter_to_slot)
        if slots == [0]:
            return base, None  # every row is __base__

        slot_ids = torch.tensor(slots, dtype=torch.long, device=x.device)
        A = self.cache.A[self.module_name].index_select(0, slot_ids)         # (S, in, max_rank)
        B = self.cache.B[self.module_name].index_select(0, slot_ids)         # (S, max_rank, out)
        scales = self.cache.scales[self.module_name].index_select(0, slot_ids)
        # Unused rank columns of A / rows of B are zero, so the full max_rank needs no rank lookup.
        rows = x.reshape(-1, x.shape[-1])                                     # (M, in)
        delta = torch.matmul(torch.matmul(rows, A), B) * scales.to(base.dtype)[:, None, None]
        out = base.reshape(1, -1, base.shape[-1]) + delta                     # (S, M, out)
        return out.reshape(len(slots), *base.shape[1:]), index

    def _project_shared_static(self, x, base, lora_ctx):
        """``project_shared`` with shapes that do not depend on which adapters the batch uses.

        The eager path projects once per distinct adapter, found on the host, so its output size
        changes with the batch. Under torch.compile that would mean a new graph per adapter mix.
        Instead, project under every cache slot (index = each row's slot) or under every row's
        adapter (index = row), whichever is fewer. Rows on the base adapter get a zero delta.
        """
        weight_indices = lora_ctx.weight_indices
        batch_size = weight_indices.shape[0]
        A_all, B_all = self.cache.A[self.module_name], self.cache.B[self.module_name]
        scales_all = self.cache.scales[self.module_name]
        if self.cache.num_slots <= batch_size:
            A, B, scales, index = A_all, B_all, scales_all, weight_indices
        else:
            A, B = A_all.index_select(0, weight_indices), B_all.index_select(0, weight_indices)
            scales = scales_all.index_select(0, weight_indices)
            index = torch.arange(batch_size, device=weight_indices.device)
        rows = x.reshape(-1, x.shape[-1]).to(A.dtype)                        # (M, in)
        delta = torch.matmul(torch.matmul(rows, A), B) * scales.to(A.dtype)[:, None, None]
        out = base.reshape(1, -1, base.shape[-1]) + delta.to(base.dtype)    # (S, M, out)
        return out.reshape(A.shape[0], *base.shape[1:]), index

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        return (
            self.cache.config.use_triton_kernels
            and x.is_cuda
            and _SGMV_LORA_A is not None
            and _SGMV_LORA_B_ is not None
        )

    def _to_segments(self, x, base, lora_ctx):
        """Map a layer input to the kernels' packed layout.

        Returns ``(x (T, in), base (T, out), seq_lens, seg_starts, weight_indices, max_seq_len)``:
        segment ``b`` is rows ``[seg_starts[b], seg_starts[b] + seq_lens[b])`` and uses adapter
        ``weight_indices[b]``. Padded layouts become packed ones with fixed-stride segments.
        """
        weight_indices = lora_ctx.weight_indices
        batch_size = weight_indices.shape[0]

        if self._is_packed(x, lora_ctx):
            return (x.contiguous(), base.contiguous(), lora_ctx.seq_lens, lora_ctx.seg_starts,
                    weight_indices, lora_ctx.padded_len)

        if x.dim() < 2:
            raise ValueError("MultiLoraLinear expects a batched tensor")

        if x.shape[0] == batch_size:
            # (B, ...): one row per batch entry; (B, features) is a single vector per row.
            width = math.prod(x.shape[1:-1])
            seq_lens, seg_starts = lora_ctx.padded_segments(width)
            return self._flat(x), self._flat(base), seq_lens, seg_starts, weight_indices, width

        if x.dim() == 3 and x.shape[0] == 1:
            # (1, M, features) shared by every row, e.g. relative position embeddings. One output can
            # only carry one adapter; models that need one per row call `project_shared` instead.
            # Whether rows mix adapters is host-side data, found when the batch was built (lora_ctx.mixed):
            # under torch.compile it is a constant of the graph, so this check holds there too.
            mixed = lora_ctx.mixed
            if mixed is None:
                mixed = len(lora_ctx.adapter_groups(self.cache.adapter_to_slot)[0]) > 1
            if mixed:
                raise ValueError(
                    f"{self.module_name}: input of shape {tuple(x.shape)} is shared by rows with different "
                    "adapters; the model must call MultiLoraLinear.project_shared to get one output per adapter"
                )
            width = x.shape[1]
            seq_lens = torch.full((1,), width, dtype=torch.int32, device=x.device)
            seg_starts = torch.zeros(1, dtype=torch.int32, device=x.device)
            return self._flat(x), self._flat(base), seq_lens, seg_starts, weight_indices[:1], width

        if x.dim() == 2 and lora_ctx.padded_len is not None and x.shape[0] == batch_size * lora_ctx.padded_len:
            # Flattened padded batch (B * L, features). Packed inputs are handled before this.
            width = lora_ctx.padded_len
            seq_lens, seg_starts = lora_ctx.padded_segments(width)
            return x.contiguous(), base.contiguous(), seq_lens, seg_starts, weight_indices, width

        raise ValueError(
            f"Cannot map layer input of shape {tuple(x.shape)} to the batch rows: expected (B, ...), "
            f"(1, M, features), (B * L, features) or packed (T, features) with B={batch_size}, "
            f"L={lora_ctx.max_seq_len}"
        )

    @staticmethod
    def _flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[-1]).contiguous()

    def _reference_forward(
        self,
        x: torch.Tensor,
        base: torch.Tensor,
        seq_lens: torch.Tensor,
        seg_starts: torch.Tensor,
        weight_indices: torch.Tensor,
    ) -> torch.Tensor:
        out = base.clone()
        A = self.cache.A[self.module_name]
        B = self.cache.B[self.module_name]
        ranks = self.cache.ranks[self.module_name]
        scales = self.cache.scales[self.module_name]
        seq_lens_cpu = seq_lens.detach().to("cpu").tolist()
        starts_cpu = seg_starts.detach().to("cpu").tolist()
        slots_cpu = weight_indices.detach().to("cpu").tolist()

        for start, seq_len, slot in zip(starts_cpu, seq_lens_cpu, slots_cpu):
            rank = int(ranks[slot].item())
            if rank == 0 or seq_len == 0:
                continue
            rows = slice(start, start + seq_len)
            delta = x[rows].matmul(A[slot, :, :rank]).matmul(B[slot, :rank, :])
            out[rows] += delta * scales[slot].to(dtype=out.dtype)
        return out
