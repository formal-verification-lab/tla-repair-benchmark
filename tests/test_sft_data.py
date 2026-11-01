import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmark import finetune
from benchmark.sft_data import cases, dump, initial_messages, records, sha


def test_synthetic_families_and_parameter_variants_stay_in_one_split():
    examples = cases()
    assert len({case.name for case in examples}) == len(examples)
    groups = {}
    for case in examples:
        groups.setdefault(case.family, set()).add(case.split)
    assert all(len(splits) == 1 for splits in groups.values())
    assert {
        split: sum(c.split == split for c in examples) for split in ("train", "valid", "test")
    } == {"train": 24, "valid": 4, "test": 4}


def test_initial_generation_never_receives_the_reference_answer():
    case = replace(cases()[0], correct="SECRET_REFERENCE", broken="BROKEN_REFERENCE")
    prompt = json.dumps(initial_messages(case))
    assert "SECRET_REFERENCE" not in prompt and "BROKEN_REFERENCE" not in prompt
    examples = records(case, "tool error")
    assert examples[0]["messages"][-1]["content"] == "SECRET_REFERENCE"
    explanation = examples[-1]["messages"]
    assert "SECRET_REFERENCE" not in json.dumps(explanation[:-1])
    assert "BROKEN_REFERENCE" in json.dumps(explanation[:-1])


def test_token_audit_counts_ids_even_if_tokenizer_returns_a_mapping(tmp_path, monkeypatch):
    row = {
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    for split in ("train", "valid", "test"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    fake = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: {
            "input_ids": [1] * 12,
            "attention_mask": [1] * 12,
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: fake)
        ),
    )
    config = {
        "model_dir": str(tmp_path),
        "dataset_dir": str(tmp_path),
        "training": {"max_seq_length": 10},
    }
    with pytest.raises(ValueError, match="Refusing token truncation"):
        finetune.tokenizer_audit(config, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()
    config["training"]["max_seq_length"] = 12
    report = finetune.tokenizer_audit(config, tmp_path / "audit.json")
    assert report["splits"]["train"]["max"] == 12


@pytest.fixture
def prepared(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights").write_bytes(b"test weights")
    dump(
        tmp_path / "model-manifest.json",
        {"model_id": "test", "revision": "fixed", "files": finetune.model_files(model)},
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "cases.json").write_text("[]")
    splits = {}
    for split in ("train", "valid", "test"):
        path = dataset / f"{split}.jsonl"
        path.write_text("{}\n")
        splits[split] = {"families": [split], "sha256": sha(path)}
    dump(dataset / "manifest.json", {"splits": splits})
    jar = tmp_path / "tools.jar"
    jar.write_bytes(b"test jar")
    config = {
        "model_id": "test",
        "model_revision": "fixed",
        "model_dir": str(model),
        "dataset_dir": str(dataset),
        "jar_path": str(jar),
        "jar_sha256": sha(jar),
    }
    finetune.seal_preparation(config)
    finetune.validate(config)
    return config


def test_preflight_rejects_changed_evaluation_fixtures(prepared):
    (finetune.resolve(prepared["dataset_dir"]) / "cases.json").write_text('["changed"]')
    with pytest.raises(ValueError, match="evaluation fixtures changed"):
        finetune.validate(prepared)


def test_preflight_rejects_changed_model_weights(prepared):
    (finetune.resolve(prepared["model_dir"]) / "weights").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model files changed"):
        finetune.validate(prepared)


def test_preflight_rejects_changed_configuration(prepared):
    prepared["seed"] = 123
    with pytest.raises(ValueError, match="Configuration changed"):
        finetune.validate(prepared)


@pytest.mark.parametrize("line", ["Train loss: nan", "    10    nan ▼", "  10    val inf"])
def test_training_stops_on_nonfinite_loss_in_text_or_table(tmp_path, line):
    with pytest.raises(ValueError, match="Nonfinite training loss"):
        finetune.run_process([sys.executable, "-c", f"print({line!r})"], tmp_path, "training")
    assert line in (tmp_path / "training.log").read_text()
