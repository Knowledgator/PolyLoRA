from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from .config import CustomLoraConfig
from .store import CpuAdapterStore


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    in_features: int
    out_features: int


class GpuAdapterCache:
    def __init__(
        self,
        store: CpuAdapterStore,
        module_specs: list[ModuleSpec],
        config: CustomLoraConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.store = store
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_slots = config.max_gpu_adapters + 1
        self.adapter_to_slot: dict[str, int] = {config.base_adapter_id: 0}
        self.slot_to_adapter: list[str | None] = [config.base_adapter_id] + [None] * config.max_gpu_adapters
        self.lru: OrderedDict[str, None] = OrderedDict()

        self.A: dict[str, torch.Tensor] = {}
        self.B: dict[str, torch.Tensor] = {}
        self.module_specs = {spec.name: spec for spec in module_specs}
        for spec in module_specs:
            self.A[spec.name] = torch.zeros(
                self.num_slots,
                spec.in_features,
                config.max_rank,
                device=self.device,
                dtype=self.dtype,
            )
            self.B[spec.name] = torch.zeros(
                self.num_slots,
                config.max_rank,
                spec.out_features,
                device=self.device,
                dtype=self.dtype,
            )
        self.ranks = torch.zeros(self.num_slots, device=self.device, dtype=torch.int32)
        self.scales = torch.zeros(self.num_slots, device=self.device, dtype=torch.float32)

    def ensure_resident(self, adapter_ids: list[str]) -> torch.Tensor:
        needed = {adapter_id for adapter_id in adapter_ids if adapter_id != self.config.base_adapter_id}
        unknown = [adapter_id for adapter_id in needed if adapter_id not in self.store]
        if unknown:
            raise KeyError(f"Unknown LoRA adapter id(s): {unknown}")
        if len(needed) > self.config.max_gpu_adapters:
            raise RuntimeError("Batch needs more adapters than GPU cache capacity")

        for adapter_id in sorted(needed):
            if adapter_id not in self.adapter_to_slot:
                slot = self._allocate_or_evict(needed)
                self._load_adapter_into_slot(adapter_id, slot)

        for adapter_id in needed:
            self._touch(adapter_id)
        self._sync_cpu_eviction_exclusions()

        return torch.tensor(
            [
                0 if adapter_id == self.config.base_adapter_id else self.adapter_to_slot[adapter_id]
                for adapter_id in adapter_ids
            ],
            dtype=torch.int64,
            device=self.device,
        )

    def to(self, device: torch.device, dtype: torch.dtype) -> None:
        device = torch.device(device)
        if self.device == device and self.dtype == dtype:
            return
        self.device = device
        self.dtype = dtype
        self.A = {name: tensor.to(device=device, dtype=dtype) for name, tensor in self.A.items()}
        self.B = {name: tensor.to(device=device, dtype=dtype) for name, tensor in self.B.items()}
        self.ranks = self.ranks.to(device=device)
        self.scales = self.scales.to(device=device)

    def _allocate_or_evict(self, needed: set[str]) -> int:
        for slot in range(1, self.num_slots):
            if self.slot_to_adapter[slot] is None:
                return slot

        for adapter_id in list(self.lru.keys()):
            if adapter_id in needed:
                continue
            slot = self.adapter_to_slot.pop(adapter_id)
            self.lru.pop(adapter_id, None)
            self.slot_to_adapter[slot] = None
            self.ranks[slot].zero_()
            self.scales[slot].zero_()
            for tensor in self.A.values():
                tensor[slot].zero_()
            for tensor in self.B.values():
                tensor[slot].zero_()
            return slot
        raise RuntimeError("No evictable LoRA adapter slot is available")

    def _load_adapter_into_slot(self, adapter_id: str, slot: int) -> None:
        adapter = self.store.get(adapter_id)
        rank: int | None = None
        scale: float | None = None
        for module_name, spec in self.module_specs.items():
            try:
                weights = adapter.layers[module_name]
            except KeyError as exc:
                raise ValueError(f"Adapter {adapter_id!r} is missing weights for {module_name!r}") from exc
            if weights.rank > self.config.max_rank:
                raise ValueError(
                    f"Adapter {adapter_id!r} rank {weights.rank} exceeds max_rank {self.config.max_rank}"
                )
            if weights.A.shape != (spec.in_features, weights.rank):
                raise ValueError(f"LoRA A shape mismatch for adapter {adapter_id!r}, module {module_name!r}")
            if weights.B.shape != (weights.rank, spec.out_features):
                raise ValueError(f"LoRA B shape mismatch for adapter {adapter_id!r}, module {module_name!r}")
            if rank is None:
                rank = weights.rank
                scale = weights.scale
            elif rank != weights.rank or scale != weights.scale:
                raise ValueError("Per-module rank/alpha patterns are not supported in the first runtime")

            self.A[module_name][slot].zero_()
            self.B[module_name][slot].zero_()
            self.A[module_name][slot, :, : weights.rank].copy_(
                weights.A.to(device=self.device, dtype=self.dtype), non_blocking=True
            )
            self.B[module_name][slot, : weights.rank, :].copy_(
                weights.B.to(device=self.device, dtype=self.dtype), non_blocking=True
            )

        self.ranks[slot] = int(rank or 0)
        self.scales[slot] = float(scale or 0.0)
        self.adapter_to_slot[adapter_id] = slot
        self.slot_to_adapter[slot] = adapter_id
        self._touch(adapter_id)
        self._sync_cpu_eviction_exclusions()

    def _touch(self, adapter_id: str) -> None:
        if adapter_id == self.config.base_adapter_id:
            return
        self.lru.pop(adapter_id, None)
        self.lru[adapter_id] = None
        self.store.touch(adapter_id)

    def _sync_cpu_eviction_exclusions(self) -> None:
        self.store.set_eviction_exclusions(
            {adapter_id for adapter_id in self.slot_to_adapter[1:] if adapter_id is not None}
        )
