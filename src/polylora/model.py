from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .cache import GpuAdapterCache, ModuleSpec
from .config import PolyLoraConfig
from .context import LoraBatchContext, assert_right_padded
from .layers import MultiLoraLinear
from .store import CpuAdapterStore, DiskAdapterCache


class PolyLoraModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        config: PolyLoraConfig,
        adapter_store: CpuAdapterStore | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.module_specs = self._collect_module_specs(base_model, config.target_modules)
        if not self.module_specs:
            raise ValueError("No nn.Linear modules matched the PolyLoRA module selection")
        disk_cache = (
            DiskAdapterCache(config.disk_cache_dir, config.max_disk_adapters)
            if config.disk_cache_dir is not None
            else None
        )
        self.adapter_store = adapter_store or CpuAdapterStore(
            config.base_adapter_id,
            config.max_cpu_adapters,
            disk_cache=disk_cache,
            module_names=self.target_module_names,
        )
        self.adapter_store.set_module_names(self.target_module_names)

        device, dtype = self._infer_device_dtype(base_model)
        self.adapter_cache = GpuAdapterCache(self.adapter_store, self.module_specs, config, device, dtype)
        self._replace_target_modules(self.base_model)

    @property
    def target_module_names(self) -> list[str]:
        return [spec.name for spec in self.module_specs]

    def add_adapter_from_peft_model(
        self,
        adapter_id: str,
        peft_model: nn.Module,
        peft_adapter_name: str | None = None,
    ) -> None:
        self.adapter_store.add_adapter_from_peft_model(
            adapter_id=adapter_id,
            peft_model=peft_model,
            module_names=self.target_module_names,
            peft_adapter_name=peft_adapter_name,
        )

    def load_adapter_from_disk(
        self,
        adapter_id: str,
        adapter_path: str,
        peft_adapter_name: str = "default",
    ) -> None:
        self.adapter_store.load_adapter_from_disk(
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            module_names=self.target_module_names,
            peft_adapter_name=peft_adapter_name,
        )

    # Not traced itself (recursive=False: the base model it calls still is). Tracing it would make
    # torch.compile guard on the adapter-id strings and recompile for every new mix of adapters.
    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        *args: Any,
        adapter_ids: list[str] | None = None,
        weight_indices: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the base model with one LoRA adapter per batch row.

        ``adapter_ids`` (one id per row; the base adapter id for no adapter) are made resident in
        the GPU cache first; ``weight_indices`` from ``resolve_adapters`` can be passed instead.
        Under ``torch.compile(model)`` this method runs eagerly and the base model is compiled, so
        neither the ids nor the adapter mix ever cause a recompile. Loading new adapters into the
        cache does not either: weights change in place in fixed-size slots.

        Pre-packed FlashDeBERTa inputs (``cu_seqlens`` and ``max_seqlen`` keyword arguments) are
        supported: rows are the sequences of ``cu_seqlens``.
        """
        if adapter_ids is None and weight_indices is None:
            return self.base_model(*args, **kwargs)
        if adapter_ids is not None:
            lora_ctx = self._context_from_ids(adapter_ids, args, kwargs)
        else:
            lora_ctx = self._build_context(None, weight_indices, args, kwargs)
        # Published on the shared cache for the MultiLoraLinear layers; plain attribute writes are
        # traceable by torch.compile, unlike a ContextVar.
        self.adapter_cache.active_context = lora_ctx
        try:
            return self.base_model(*args, **kwargs)
        finally:
            self.adapter_cache.active_context = None

    @torch.compiler.disable  # host-side bookkeeping; never part of a graph
    def resolve_adapters(self, adapter_ids: list[str], device: torch.device | str | None = None) -> torch.Tensor:
        """Make ``adapter_ids`` resident in the GPU cache; returns the (B,) slot of every row.

        Pass the result as ``weight_indices`` to ``forward``. It stays valid until another call
        evicts one of the adapters, so resolve right before every forward pass.
        """
        device, dtype = self._infer_device_dtype(self.base_model) if device is None else (
            torch.device(device), self._infer_device_dtype(self.base_model)[1])
        self.adapter_cache.to(device, dtype)
        weight_indices = self.adapter_cache.ensure_resident(list(adapter_ids))
        weight_indices._polylora_mixed = len(set(adapter_ids)) > 1  # read by forward, on the host
        return weight_indices

    @torch.compiler.disable  # host-side bookkeeping; never part of a graph
    def _context_from_ids(self, adapter_ids, args, kwargs) -> LoraBatchContext:
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None:  # pre-packed inputs: rows are the sequences of cu_seqlens
            batch_size, device = cu_seqlens.numel() - 1, cu_seqlens.device
        else:
            batch_size, _, device, _ = self._infer_forward_shape(*args, **kwargs)
        if len(adapter_ids) != batch_size:
            raise ValueError("adapter_ids length must match batch size")
        weight_indices = self.resolve_adapters(adapter_ids, device)
        return self._build_context(adapter_ids, weight_indices, args, kwargs, check_padding=True)

    @torch.compiler.disable  # host-side bookkeeping; never part of a graph
    def _build_context(self, adapter_ids, weight_indices, args, kwargs, check_padding=False) -> LoraBatchContext:
        """Batch context from tensors only, for padded or pre-packed inputs."""
        # Sequence lengths change from batch to batch: let the compiled base model treat the length as
        # symbolic from its first compile (marks only apply outside traced code, as here).
        for tensor in (kwargs.get("attention_mask"), kwargs.get("input_ids"), kwargs.get("token_type_ids"),
                       *(a for a in args if torch.is_tensor(a))):
            if tensor is not None and tensor.dim() >= 2:
                torch._dynamo.maybe_mark_dynamic(tensor, 1)
        if adapter_ids is not None:
            mixed = len(set(adapter_ids)) > 1
        else:
            mixed = getattr(weight_indices, "_polylora_mixed", None)
            if mixed is None:  # slots built by the caller: one read of the (B,) slots
                mixed = bool((weight_indices != weight_indices[:1]).any())
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
            return LoraBatchContext(
                adapter_ids=None if adapter_ids is None else list(adapter_ids),
                weight_indices=weight_indices,
                seq_lens=seq_lens,
                max_seq_len=int(kwargs["max_seqlen"]),
                seg_starts=cu_seqlens[:-1].to(torch.int32),
                packed=True,
                mixed=mixed,
            )
        batch_size, seq_len, device, _ = self._infer_forward_shape(*args, **kwargs)
        if weight_indices.shape[0] != batch_size:
            raise ValueError("weight_indices length must match batch size")
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        else:
            seq_lens = attention_mask.to(device=device, dtype=torch.int32).sum(dim=-1)
            # A data-dependent check: eager only.
            if check_padding and self.config.enforce_right_padding and not torch.compiler.is_compiling():
                assert_right_padded(attention_mask.to(device), seq_lens)
        seq_lens = seq_lens.to(torch.int32)
        seg_starts = torch.cumsum(seq_lens, dim=0, dtype=torch.int32) - seq_lens
        return LoraBatchContext(
            adapter_ids=None if adapter_ids is None else list(adapter_ids),
            weight_indices=weight_indices,
            seq_lens=seq_lens,
            attention_mask=attention_mask,
            max_seq_len=seq_len,
            seg_starts=seg_starts,
            mixed=mixed,
        )

    def prepare_lora_context(
        self,
        adapter_ids: list[str],
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> LoraBatchContext:
        """Context for a padded (B, L) batch, for use with ``use_lora_context``."""
        if len(adapter_ids) != batch_size:
            raise ValueError("adapter_ids length must match batch size")
        weight_indices = self.adapter_cache.ensure_resident(adapter_ids)
        if attention_mask is None:
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        else:
            seq_lens = attention_mask.to(device=device, dtype=torch.int32).sum(dim=-1)
            if self.config.enforce_right_padding:
                assert_right_padded(attention_mask.to(device), seq_lens)

        seq_lens = seq_lens.to(torch.int32)
        seg_starts = torch.cumsum(seq_lens, dim=0, dtype=torch.int32) - seq_lens
        return LoraBatchContext(
            adapter_ids=list(adapter_ids),
            weight_indices=weight_indices,
            seq_lens=seq_lens,
            attention_mask=attention_mask,
            max_seq_len=seq_len,
            seg_starts=seg_starts,
        )

    @staticmethod
    def _matches_target(name: str, target_modules: list[str]) -> bool:
        return any(name == target or name.endswith(f".{target}") for target in target_modules)

    @classmethod
    def _collect_module_specs(cls, model: nn.Module, target_modules: list[str] | None) -> list[ModuleSpec]:
        specs: list[ModuleSpec] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and (target_modules is None or cls._matches_target(name, target_modules)):
                specs.append(ModuleSpec(name, module.in_features, module.out_features))
        return specs

    def _replace_target_modules(self, model: nn.Module) -> None:
        for module_name in self.target_module_names:
            parent, child_name = self._get_parent(model, module_name)
            child = getattr(parent, child_name)
            if not isinstance(child, nn.Linear):
                raise TypeError(f"Expected nn.Linear at {module_name!r}, found {type(child).__name__}")
            setattr(parent, child_name, MultiLoraLinear(child, module_name, self.adapter_cache))

    @staticmethod
    def _get_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
        parts = module_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    @staticmethod
    def _infer_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
        for param in model.parameters():
            return param.device, param.dtype
        return torch.device("cpu"), torch.float32

    @staticmethod
    def _infer_forward_shape(*args: Any, **kwargs: Any) -> tuple[int, int, torch.device, torch.dtype]:
        tensor = kwargs.get("input_ids")
        if tensor is None:
            tensor = kwargs.get("inputs_embeds")
        if tensor is None:
            for arg in args:
                if torch.is_tensor(arg):
                    tensor = arg
                    break
        if tensor is None:
            raise ValueError("Cannot infer batch size without input_ids, inputs_embeds, or a tensor positional arg")
        if tensor.dim() < 2:
            raise ValueError("Input tensor must have batch and sequence dimensions")
        batch_size, seq_len = int(tensor.shape[0]), int(tensor.shape[1])
        dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
        return batch_size, seq_len, tensor.device, dtype


CustomPeftModel = PolyLoraModel
