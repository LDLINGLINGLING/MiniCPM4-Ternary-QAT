"""Tiny CPU tests for the unchanged FP32 master HF conversion intermediate."""
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import export_bitnet_hf as exporter


def make_master():
    generator = torch.Generator().manual_seed(7)
    state = {
        "model.embed_tokens.weight": torch.randn(5, 7, generator=generator),
        "model.norm.weight": torch.randn(7, generator=generator),
    }
    for layer in range(exporter.NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        state[f"{prefix}.input_layernorm.weight"] = torch.randn(7, generator=generator)
        state[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(7, generator=generator)
        for module in exporter.PROJECTION_MODULES:
            # Values outside [-1, 1] make accidental ternarization obvious.
            state[f"{prefix}.{module}.weight"] = 3 * torch.randn(3, 5, generator=generator)
    assert sum(name.endswith(exporter.PROJECTION_SUFFIXES) for name in state) == 168
    return state


def make_base(path):
    path.mkdir()
    for name in exporter.HF_ASSET_NAMES:
        (path / name).write_text(f"asset:{name}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"base weights must not be copied")
    (path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")


def run_export(tmp_path, master, config, granularity):
    checkpoint = tmp_path / "state.pt"
    run_config = tmp_path / "resolved_config.json"
    base = tmp_path / "base"
    output = tmp_path / "output"
    torch.save({"model": master, "optimizer": {"step": 1}}, checkpoint)
    run_config.write_text(json.dumps(config), encoding="utf-8")
    make_base(base)
    result = exporter.main([
        "--checkpoint", str(checkpoint),
        "--run-config", str(run_config),
        "--base-model", str(base),
        "--output-dir", str(output),
        "--granularity", granularity,
    ])
    assert result == 0
    return checkpoint, run_config, base, output


@pytest.mark.parametrize("granularity,config", [
    ("row", {}),
    ("row", {"quant_group": None}),
    ("g32", {"quant_group": 32}),
])
def test_export_preserves_every_fp32_master_tensor(tmp_path, granularity, config):
    master = make_master()
    checkpoint, run_config, base, output = run_export(tmp_path, master, config, granularity)

    with safe_open(str(output / "model.safetensors"), framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(master)
        assert len(handle.keys()) == len(master)
        assert handle.metadata() == {"format": "pt"}
        for name, expected in master.items():
            actual = handle.get_tensor(name)
            assert actual.dtype == torch.float32
            assert actual.is_contiguous()
            assert torch.equal(actual, expected), name

    for name in exporter.HF_ASSET_NAMES:
        assert (output / name).read_bytes() == (base / name).read_bytes()
    assert (output / "model.safetensors").read_bytes() != (base / "model.safetensors").read_bytes()
    assert not (output / "model.safetensors.index.json").exists()

    manifest = json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_kind"] == "fp32_master_conversion_intermediate"
    assert manifest["quantization_applied"] is False
    assert manifest["granularity"] == granularity
    assert manifest["quant_group"] == (32 if granularity == "g32" else None)
    assert manifest["projection_tensor_count"] == 168
    assert manifest["source_checkpoint"] == str(checkpoint.resolve())
    assert manifest["run_config"] == str(run_config.resolve())

    marker = output / "do_not_overwrite"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        exporter.export_checkpoint(checkpoint, run_config, base, output, granularity)
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("granularity,config", [
    ("row", {"quant_group": 32}),
    ("row", {"quant_group": 0}),
    ("g32", {}),
    ("g32", {"quant_group": None}),
    ("g32", {"quant_group": 16}),
    ("g32", {"quant_group": 32.0}),
])
def test_granularity_must_match_resolved_config(tmp_path, granularity, config):
    checkpoint = tmp_path / "state.pt"
    checkpoint.write_bytes(b"must not be loaded for an invalid granularity")
    run_config = tmp_path / "resolved_config.json"
    run_config.write_text(json.dumps(config), encoding="utf-8")
    base = tmp_path / "base"
    make_base(base)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="granularity"):
        exporter.export_checkpoint(checkpoint, run_config, base, output, granularity)
    assert not output.exists()


@pytest.mark.parametrize("defect", ["missing_projection", "bf16_tensor"])
def test_invalid_master_is_rejected_before_output_creation(tmp_path, defect):
    master = make_master()
    if defect == "missing_projection":
        del master["model.layers.23.mlp.down_proj.weight"]
    else:
        master["model.norm.weight"] = master["model.norm.weight"].bfloat16()
    checkpoint = tmp_path / "state.pt"
    torch.save({"model": master}, checkpoint)
    run_config = tmp_path / "resolved_config.json"
    run_config.write_text("{}", encoding="utf-8")
    base = tmp_path / "base"
    make_base(base)
    output = tmp_path / "output"
    match = "exactly 168" if defect == "missing_projection" else "FP32 master"
    with pytest.raises(ValueError, match=match):
        exporter.export_checkpoint(checkpoint, run_config, base, output, "row")
    assert not output.exists()
