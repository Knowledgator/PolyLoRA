from __future__ import annotations

import torch
from torch import nn

from .cache import GpuAdapterCache
from .context import get_lora_context


try:
    from .sgmv import sgmv_lora_a_fwd as _SGMV_LORA_A_FWD
    from .sgmv import sgmv_lora_b_fwd as _SGMV_LORA_B_FWD
except Exception:
    _SGMV_LORA_A_FWD = None
    _SGMV_LORA_B_FWD = None


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora_ctx = get_lora_context()
        if lora_ctx is None:
            return base

        if x.shape[0] != lora_ctx.weight_indices.shape[0]:
            raise ValueError("LoRA batch context batch size does not match layer input batch size")

        original_shape = base.shape
        x_3d = self._as_3d(x).contiguous()
        base_3d = self._as_3d(base).contiguous()

        if self._can_use_triton(x_3d):
            tmp = _SGMV_LORA_A_FWD(
                x_3d,
                self.cache.A[self.module_name],
                lora_ctx.seq_lens,
                lora_ctx.weight_indices,
                self.cache.ranks,
            )
            out = _SGMV_LORA_B_FWD(
                tmp,
                self.cache.B[self.module_name],
                lora_ctx.seq_lens,
                lora_ctx.weight_indices,
                self.cache.ranks,
                self.cache.scales,
                base_3d,
            )
        else:
            out = self._reference_forward(x_3d, base_3d, lora_ctx.seq_lens, lora_ctx.weight_indices)

        return out.reshape(original_shape)

    def _can_use_triton(self, x: torch.Tensor) -> bool:
        return (
            self.cache.config.use_triton_kernels
            and x.is_cuda
            and _SGMV_LORA_A_FWD is not None
            and _SGMV_LORA_B_FWD is not None
        )

    @staticmethod
    def _as_3d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        if x.dim() > 3:
            return x.reshape(x.shape[0], -1, x.shape[-1])
        raise ValueError("MultiLoraLinear expects a batched tensor")

    def _reference_forward(
        self,
        x: torch.Tensor,
        base: torch.Tensor,
        seq_lens: torch.Tensor,
        weight_indices: torch.Tensor,
    ) -> torch.Tensor:
        out = base.clone()
        A = self.cache.A[self.module_name]
        B = self.cache.B[self.module_name]
        ranks = self.cache.ranks
        scales = self.cache.scales
        seq_lens_cpu = seq_lens.detach().to("cpu").tolist()
        slots_cpu = weight_indices.detach().to("cpu").tolist()

        for batch_idx, slot in enumerate(slots_cpu):
            rank = int(ranks[slot].item())
            seq_len = int(seq_lens_cpu[batch_idx])
            if rank == 0 or seq_len == 0:
                continue
            delta = x[batch_idx, :seq_len].matmul(A[slot, :, :rank]).matmul(B[slot, :rank, :])
            out[batch_idx, :seq_len] += delta * scales[slot].to(dtype=out.dtype)
        return out
