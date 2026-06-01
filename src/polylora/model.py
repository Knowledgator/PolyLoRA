from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .cache import GpuAdapterCache, ModuleSpec
from .config import CustomLoraConfig
from .context import LoraBatchContext, assert_right_padded, use_lora_context
from .layers import MultiLoraLinear
from .store import CpuAdapterStore, DiskAdapterCache


class CustomPeftModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        config: CustomLoraConfig,
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

    def forward(self, *args: Any, adapter_ids: list[str] | None = None, **kwargs: Any) -> Any:
        if adapter_ids is None:
            return self.base_model(*args, **kwargs)

        batch_size, seq_len, device, _ = self._infer_forward_shape(*args, **kwargs)
        _, dtype = self._infer_device_dtype(self.base_model)
        self.adapter_cache.to(device, dtype)
        attention_mask = kwargs.get("attention_mask")
        lora_ctx = self.prepare_lora_context(adapter_ids, attention_mask, batch_size, seq_len, device)
        with use_lora_context(lora_ctx):
            return self.base_model(*args, **kwargs)

    def prepare_lora_context(
        self,
        adapter_ids: list[str],
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> LoraBatchContext:
        if len(adapter_ids) != batch_size:
            raise ValueError("adapter_ids length must match batch size")

        weight_indices = self.adapter_cache.ensure_resident(adapter_ids)
        if attention_mask is None:
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        else:
            seq_lens = attention_mask.to(device=device, dtype=torch.int32).sum(dim=-1)
            if self.config.enforce_right_padding:
                assert_right_padded(attention_mask.to(device), seq_lens)

        return LoraBatchContext(
            adapter_ids=list(adapter_ids),
            weight_indices=weight_indices,
            seq_lens=seq_lens,
            attention_mask=attention_mask,
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
