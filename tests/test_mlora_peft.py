import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch_spec = importlib.util.find_spec("torch")
transformers_spec = importlib.util.find_spec("transformers")
pytestmark = pytest.mark.skipif(
    torch_spec is None or transformers_spec is None,
    reason="torch and transformers are required for mlora tests",
)

import torch
from transformers import DebertaV2Config, DebertaV2Model

from mlora import CustomLoraConfig, CustomPeftModel


def _import_local_peft():
    peft_src = Path(__file__).resolve().parents[2] / "peft" / "src"
    if not peft_src.exists():
        pytest.skip("local PEFT checkout is not available")
    sys.path.insert(0, str(peft_src))
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except Exception as exc:
        pytest.skip(f"PEFT import failed: {exc}")
    return LoraConfig, PeftModel, TaskType, get_peft_model


def test_mlora_mixed_deberta_matches_peft_multiple_adapters(tmp_path):
    LoraConfig, _, TaskType, get_peft_model = _import_local_peft()
    torch.manual_seed(0)

    deberta_config = DebertaV2Config(
        vocab_size=97,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=32,
        relative_attention=False,
        type_vocab_size=0,
    )
    base_model = DebertaV2Model(deberta_config).eval()

    lora_a_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=2,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        init_lora_weights=False,
    )
    lora_b_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=3,
        lora_alpha=6,
        lora_dropout=0.0,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        init_lora_weights=False,
    )

    peft_model = get_peft_model(copy.deepcopy(base_model), lora_a_config, adapter_name="a").eval()
    peft_model.add_adapter("b", lora_b_config)
    peft_model.eval()

    mlora_model = CustomPeftModel(
        copy.deepcopy(base_model),
        CustomLoraConfig(
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            max_gpu_adapters=2,
            max_rank=3,
            use_triton_kernels=False,
        ),
    ).eval()
    mlora_model.add_adapter_from_peft_model("a", peft_model, peft_adapter_name="a")

    adapter_b_path = tmp_path / "adapter_b"
    adapter_b_path.mkdir()
    with (adapter_b_path / "adapter_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "peft_type": "LORA",
                "r": 3,
                "lora_alpha": 6,
                "lora_dropout": 0.0,
                "target_modules": ["query_proj", "key_proj", "value_proj", "dense"],
            },
            handle,
        )
    torch.save(peft_model.state_dict(), adapter_b_path / "adapter_model.bin")
    mlora_model.load_adapter_from_disk("b", str(adapter_b_path), peft_adapter_name="b")

    sample = {
        "input_ids": torch.tensor(
            [
                [1, 5, 6, 7, 2],
                [1, 8, 9, 4, 2],
                [1, 3, 3, 3, 2],
                [1, 11, 12, 13, 2],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(4, 5, dtype=torch.long),
    }
    adapter_ids = ["a", "b", "__base__", "a"]

    with torch.inference_mode():
        peft_outputs = peft_model(**sample, adapter_names=adapter_ids).last_hidden_state
        mlora_outputs = mlora_model(**sample, adapter_ids=adapter_ids).last_hidden_state
        base_outputs = base_model(**sample).last_hidden_state

    assert torch.allclose(mlora_outputs, peft_outputs, atol=1e-5, rtol=1e-5)
    assert torch.allclose(mlora_outputs[2], base_outputs[2], atol=1e-5, rtol=1e-5)


def test_mlora_loads_three_peft_adapters_from_disk_and_batches_predictions(tmp_path):
    LoraConfig, PeftModel, TaskType, get_peft_model = _import_local_peft()
    torch.manual_seed(3)

    deberta_config = DebertaV2Config(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=32,
        relative_attention=False,
        type_vocab_size=0,
    )
    base_model = DebertaV2Model(deberta_config).eval()
    target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
    adapter_specs = [
        ("adapter_0", 2, 4),
        ("adapter_1", 3, 9),
        ("adapter_2", 4, 8),
    ]

    adapter_paths: dict[str, Path] = {}
    for seed, (adapter_id, rank, alpha) in enumerate(adapter_specs, start=10):
        torch.manual_seed(seed)
        peft_model = get_peft_model(
            copy.deepcopy(base_model),
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.0,
                target_modules=target_modules,
                init_lora_weights=False,
            ),
        ).eval()
        adapter_path = tmp_path / adapter_id
        peft_model.save_pretrained(str(adapter_path), safe_serialization=False)
        adapter_paths[adapter_id] = adapter_path

    batch = {
        "input_ids": torch.tensor(
            [
                [1, 5, 6, 7, 2, 0],
                [1, 8, 9, 4, 3, 2],
                [1, 11, 12, 13, 14, 2],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
    }
    adapter_ids = [adapter_id for adapter_id, _, _ in adapter_specs]

    peft_outputs = []
    with torch.inference_mode():
        for batch_idx, adapter_id in enumerate(adapter_ids):
            peft_model = PeftModel.from_pretrained(
                copy.deepcopy(base_model),
                str(adapter_paths[adapter_id]),
            ).eval()
            single_sample = {key: value[batch_idx : batch_idx + 1] for key, value in batch.items()}
            peft_outputs.append(peft_model(**single_sample).last_hidden_state)
        expected = torch.cat(peft_outputs, dim=0)

    mlora_model = CustomPeftModel(
        copy.deepcopy(base_model),
        CustomLoraConfig(
            target_modules=target_modules,
            max_gpu_adapters=3,
            max_rank=4,
            use_triton_kernels=False,
        ),
    ).eval()
    for adapter_id in adapter_ids:
        mlora_model.load_adapter_from_disk(adapter_id, str(adapter_paths[adapter_id]))

    with torch.inference_mode():
        actual = mlora_model(**batch, adapter_ids=adapter_ids).last_hidden_state

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_mlora_gpu_cache_reloads_from_cpu_cache():
    LoraConfig, _, TaskType, get_peft_model = _import_local_peft()
    torch.manual_seed(1)

    deberta_config = DebertaV2Config(
        vocab_size=97,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=32,
        relative_attention=False,
        type_vocab_size=0,
    )
    base_model = DebertaV2Model(deberta_config).eval()
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        init_lora_weights=False,
    )
    peft_model = get_peft_model(copy.deepcopy(base_model), lora_config, adapter_name="a").eval()
    peft_model.add_adapter("b", lora_config)

    mlora_model = CustomPeftModel(
        copy.deepcopy(base_model),
        CustomLoraConfig(
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            max_gpu_adapters=1,
            max_cpu_adapters=2,
            max_rank=2,
            use_triton_kernels=False,
        ),
    ).eval()
    mlora_model.add_adapter_from_peft_model("a", peft_model, peft_adapter_name="a")
    mlora_model.add_adapter_from_peft_model("b", peft_model, peft_adapter_name="b")

    sample = {
        "input_ids": torch.tensor([[1, 5, 6, 2]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

    with torch.inference_mode():
        out_a_1 = mlora_model(**sample, adapter_ids=["a"]).last_hidden_state
        assert mlora_model.adapter_cache.slot_to_adapter[1] == "a"
        out_b = mlora_model(**sample, adapter_ids=["b"]).last_hidden_state
        assert mlora_model.adapter_cache.slot_to_adapter[1] == "b"
        out_a_2 = mlora_model(**sample, adapter_ids=["a"]).last_hidden_state
        assert mlora_model.adapter_cache.slot_to_adapter[1] == "a"

    assert "a" in mlora_model.adapter_store
    assert "b" in mlora_model.adapter_store
    assert not torch.allclose(out_a_1, out_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_a_1, out_a_2, atol=1e-5, rtol=1e-5)


def test_mlora_base_adapter_uses_slot_zero_rank_zero():
    LoraConfig, _, TaskType, get_peft_model = _import_local_peft()
    torch.manual_seed(2)

    deberta_config = DebertaV2Config(
        vocab_size=97,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=32,
        relative_attention=False,
        type_vocab_size=0,
    )
    base_model = DebertaV2Model(deberta_config).eval()
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        init_lora_weights=False,
    )
    peft_model = get_peft_model(copy.deepcopy(base_model), lora_config, adapter_name="a").eval()

    mlora_model = CustomPeftModel(
        copy.deepcopy(base_model),
        CustomLoraConfig(
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            max_gpu_adapters=1,
            max_rank=2,
            use_triton_kernels=False,
        ),
    ).eval()
    mlora_model.add_adapter_from_peft_model("a", peft_model, peft_adapter_name="a")

    sample = {
        "input_ids": torch.tensor([[1, 5, 6, 2], [1, 8, 9, 2]], dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }

    with torch.inference_mode():
        mixed = mlora_model(**sample, adapter_ids=["__base__", "a"]).last_hidden_state
        base = base_model(**sample).last_hidden_state

    assert mlora_model.adapter_cache.adapter_to_slot["__base__"] == 0
    assert int(mlora_model.adapter_cache.ranks[0].item()) == 0
    assert float(mlora_model.adapter_cache.scales[0].item()) == 0.0
    assert torch.allclose(mixed[0], base[0], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(mixed[1], base[1], atol=1e-5, rtol=1e-5)
