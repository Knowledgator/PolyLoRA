from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CpuLayerWeights:
    A: torch.Tensor
    B: torch.Tensor
    rank: int
    scale: float


@dataclass(frozen=True)
class CpuAdapter:
    adapter_id: str
    config: dict[str, Any]
    layers: dict[str, CpuLayerWeights]


def _pin_if_possible(tensor: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor
    return tensor


class CpuAdapterStore:
    def __init__(self, base_adapter_id: str = "__base__", max_adapters: int | None = None) -> None:
        if max_adapters is not None and max_adapters < 1:
            raise ValueError("max_adapters must be at least 1")
        self.base_adapter_id = base_adapter_id
        self.max_adapters = max_adapters
        self.adapters: OrderedDict[str, CpuAdapter] = OrderedDict()
        self.eviction_exclusions: set[str] = set()

    def __contains__(self, adapter_id: str) -> bool:
        return adapter_id == self.base_adapter_id or adapter_id in self.adapters

    def get(self, adapter_id: str) -> CpuAdapter:
        try:
            adapter = self.adapters[adapter_id]
        except KeyError as exc:
            raise KeyError(f"Unknown LoRA adapter id: {adapter_id}") from exc
        self.touch(adapter_id)
        return adapter

    def touch(self, adapter_id: str) -> None:
        if adapter_id in self.adapters:
            self.adapters.move_to_end(adapter_id)

    def set_eviction_exclusions(self, adapter_ids: set[str]) -> None:
        self.eviction_exclusions = {adapter_id for adapter_id in adapter_ids if adapter_id != self.base_adapter_id}

    def add_adapter(
        self,
        adapter_id: str,
        layers: dict[str, CpuLayerWeights],
        config: dict[str, Any] | None = None,
    ) -> None:
        if adapter_id == self.base_adapter_id:
            raise ValueError(f"{self.base_adapter_id!r} is reserved for base-only rows")
        if not layers:
            raise ValueError("Adapter must contain at least one target layer")
        self.adapters[adapter_id] = CpuAdapter(adapter_id, config or {}, layers)
        self.adapters.move_to_end(adapter_id)
        self._evict_if_needed(exclude={adapter_id})

    def add_adapter_from_peft_state_dict(
        self,
        adapter_id: str,
        state_dict: dict[str, torch.Tensor],
        module_names: list[str],
        lora_alpha: float,
        peft_adapter_name: str = "default",
        config: dict[str, Any] | None = None,
    ) -> None:
        layers: dict[str, CpuLayerWeights] = {}
        for module_name in module_names:
            lora_a_key = self._find_lora_key(state_dict, module_name, "lora_A", peft_adapter_name)
            lora_b_key = self._find_lora_key(state_dict, module_name, "lora_B", peft_adapter_name)
            if lora_a_key is None or lora_b_key is None:
                raise ValueError(f"Adapter {adapter_id!r} is missing LoRA weights for {module_name!r}")

            lora_a = state_dict[lora_a_key].detach().to("cpu").contiguous()
            lora_b = state_dict[lora_b_key].detach().to("cpu").contiguous()
            rank = int(lora_a.shape[0])
            if rank <= 0:
                raise ValueError(f"Adapter {adapter_id!r} has non-positive rank for {module_name!r}")
            if int(lora_b.shape[1]) != rank:
                raise ValueError(f"LoRA A/B rank mismatch for adapter {adapter_id!r}, module {module_name!r}")

            layers[module_name] = CpuLayerWeights(
                A=_pin_if_possible(lora_a.t().contiguous()),
                B=_pin_if_possible(lora_b.t().contiguous()),
                rank=rank,
                scale=float(lora_alpha) / rank,
            )

        self.add_adapter(adapter_id, layers, config)

    def add_adapter_from_peft_model(
        self,
        adapter_id: str,
        peft_model: torch.nn.Module,
        module_names: list[str],
        peft_adapter_name: str | None = None,
    ) -> None:
        peft_adapter_name = peft_adapter_name or adapter_id
        peft_config = getattr(peft_model, "peft_config", {})
        adapter_config = peft_config.get(peft_adapter_name)
        if adapter_config is None:
            raise ValueError(f"PEFT model does not contain adapter {peft_adapter_name!r}")
        self._validate_peft_config(adapter_config)
        lora_alpha = float(getattr(adapter_config, "lora_alpha"))
        self.add_adapter_from_peft_state_dict(
            adapter_id=adapter_id,
            state_dict=peft_model.state_dict(),
            module_names=module_names,
            lora_alpha=lora_alpha,
            peft_adapter_name=peft_adapter_name,
            config=self._config_to_dict(adapter_config),
        )

    def load_adapter_from_disk(
        self,
        adapter_id: str,
        adapter_path: str | Path,
        module_names: list[str],
        peft_adapter_name: str = "default",
    ) -> None:
        adapter_path = Path(adapter_path)
        config_path = adapter_path / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"PEFT adapter config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        self._validate_peft_config_dict(config)
        lora_alpha = float(config["lora_alpha"])
        state_dict = self._load_peft_state_dict(adapter_path)
        self.add_adapter_from_peft_state_dict(
            adapter_id=adapter_id,
            state_dict=state_dict,
            module_names=module_names,
            lora_alpha=lora_alpha,
            peft_adapter_name=peft_adapter_name,
            config=config,
        )

    @staticmethod
    def _find_lora_key(
        state_dict: dict[str, torch.Tensor],
        module_name: str,
        side: str,
        peft_adapter_name: str,
    ) -> str | None:
        suffix = f"{module_name}.{side}.{peft_adapter_name}.weight"
        suffix_without_adapter = f"{module_name}.{side}.weight"
        for key in state_dict:
            if key.endswith(suffix) or key.endswith(suffix_without_adapter):
                return key
        return None

    @staticmethod
    def _load_peft_state_dict(adapter_path: Path) -> dict[str, torch.Tensor]:
        safetensors_path = adapter_path / "adapter_model.safetensors"
        bin_path = adapter_path / "adapter_model.bin"
        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError("safetensors is required to load adapter_model.safetensors") from exc
            return load_file(str(safetensors_path), device="cpu")
        if bin_path.exists():
            return torch.load(bin_path, map_location="cpu")
        raise FileNotFoundError(f"No PEFT adapter weights found in {adapter_path}")

    @staticmethod
    def _validate_peft_config(config: Any) -> None:
        unsupported = {
            "use_dora": False,
            "lora_bias": False,
            "use_rslora": False,
        }
        for name, supported_value in unsupported.items():
            if hasattr(config, name) and getattr(config, name) != supported_value:
                raise ValueError(f"Unsupported PEFT LoRA option: {name}={getattr(config, name)!r}")
        if getattr(config, "lora_dropout", 0.0) not in (0, 0.0):
            raise ValueError("LoRA dropout is not supported by the inference runtime")

    @staticmethod
    def _validate_peft_config_dict(config: dict[str, Any]) -> None:
        if config.get("peft_type") not in (None, "LORA"):
            raise ValueError(f"Unsupported PEFT adapter type: {config.get('peft_type')!r}")
        if config.get("use_dora", False):
            raise ValueError("DoRA adapters are not supported")
        if config.get("use_rslora", False):
            raise ValueError("RS-LoRA adapters are not supported")
        if config.get("lora_bias", False):
            raise ValueError("LoRA bias is not supported")
        if float(config.get("lora_dropout", 0.0)) != 0.0:
            raise ValueError("LoRA dropout is not supported by the inference runtime")
        if "lora_alpha" not in config:
            raise ValueError("adapter_config.json is missing lora_alpha")

    @staticmethod
    def _config_to_dict(config: Any) -> dict[str, Any]:
        if hasattr(config, "to_dict"):
            return dict(config.to_dict())
        if hasattr(config, "__dict__"):
            return dict(config.__dict__)
        return {}

    def _evict_if_needed(self, exclude: set[str] | None = None) -> None:
        if self.max_adapters is None:
            return
        exclude = (exclude or set()) | self.eviction_exclusions
        while len(self.adapters) > self.max_adapters:
            for adapter_id in list(self.adapters.keys()):
                if adapter_id in exclude:
                    continue
                self.adapters.pop(adapter_id)
                break
            else:
                raise RuntimeError("No evictable CPU LoRA adapter is available")
