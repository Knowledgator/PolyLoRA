from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CustomLoraConfig:
    max_gpu_adapters: int
    max_rank: int
    target_modules: list[str] | None = None
    max_cpu_adapters: int | None = None
    disk_cache_dir: str | Path | None = None
    max_disk_adapters: int | None = None
    base_adapter_id: str = "__base__"
    enforce_right_padding: bool = True
    use_triton_kernels: bool = True

    def __post_init__(self) -> None:
        if self.max_gpu_adapters < 1:
            raise ValueError("max_gpu_adapters must be at least 1")
        if self.max_cpu_adapters is not None and self.max_cpu_adapters < self.max_gpu_adapters:
            raise ValueError("max_cpu_adapters must be at least max_gpu_adapters")
        if self.max_disk_adapters is not None and self.max_disk_adapters < 1:
            raise ValueError("max_disk_adapters must be at least 1")
        if self.max_disk_adapters is not None and self.disk_cache_dir is None:
            raise ValueError("disk_cache_dir must be set when max_disk_adapters is set")
        if self.max_rank < 1:
            raise ValueError("max_rank must be at least 1")
        if self.target_modules is not None and not self.target_modules:
            raise ValueError("target_modules must not be empty when provided")
