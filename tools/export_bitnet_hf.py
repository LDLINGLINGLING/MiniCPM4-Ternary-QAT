#!/usr/bin/env python3
"""Export an unchanged FP32 master checkpoint as a Hugging Face intermediate.

The output is intentionally *not* ternarized. The approved deployment path is
FP32 master -> F32 GGUF -> native absmean I2_S_ROW/G32; applying absmean here
and again in the native converter would incorrectly rescale the weights.

Only use --checkpoint with a trusted local state.pt: loading it requires
weights_only=False because the training checkpoint contains more than tensors.
"""

import argparse
from collections.abc import Mapping
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


NUM_LAYERS = 24
PROJECTION_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
PROJECTION_SUFFIXES = tuple(f"{name.rsplit('.', 1)[-1]}.weight" for name in PROJECTION_MODULES)
EXPECTED_PROJECTIONS = frozenset(
    f"model.layers.{layer}.{module}.weight"
    for layer in range(NUM_LAYERS)
    for module in PROJECTION_MODULES
)

# MiniCPM's two Python files are needed by config.json's auto_map. The fixed
# allowlist deliberately excludes all base-model weight files and indexes.
HF_ASSET_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "configuration_minicpm.py",
    "modeling_minicpm.py",
)
REQUIRED_HF_ASSETS = ("config.json", "tokenizer_config.json")
TOKENIZER_DATA_NAMES = ("tokenizer.json", "tokenizer.model")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export unchanged FP32 master weights as the Hugging Face intermediate for "
            "FP32 master -> F32 GGUF -> native absmean I2_S_ROW/G32. "
            "This command does not ternarize or otherwise change tensor values."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Trusted local state.pt containing the FP32 master under 'model' (uses weights_only=False).",
    )
    parser.add_argument(
        "--run-config",
        required=True,
        type=Path,
        help="The run's resolved_config.json; its quant_group must match --granularity.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        type=Path,
        help="Local MiniCPM HF directory supplying config/tokenizer/remote-code files, never weights.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="NEW output directory; an existing path is rejected and never overwritten.",
    )
    parser.add_argument(
        "--granularity",
        required=True,
        choices=("row", "g32"),
        help=(
            "Granularity for the later native quantizer: row requires a missing/null quant_group; "
            "g32 requires exactly 32. No quantization is applied by this export."
        ),
    )
    return parser.parse_args(argv)


def _validate_granularity(run_config_path, granularity):
    with run_config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Run config must contain a JSON object: {run_config_path}")

    group = config.get("quant_group")
    if granularity == "row":
        if group is not None:
            raise ValueError(
                f"--granularity row requires missing quant_group or null, got {group!r}"
            )
    elif type(group) is not int or group != 32:  # bool and 32.0 are not exactly integer 32.
        raise ValueError(f"--granularity g32 requires quant_group exactly 32, got {group!r}")
    return group


def _validate_base_model(base_model):
    if not base_model.is_dir():
        raise FileNotFoundError(f"Base model directory does not exist: {base_model}")
    missing = [name for name in REQUIRED_HF_ASSETS if not (base_model / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Base model is missing required files {missing}: {base_model}")
    if not any((base_model / name).is_file() for name in TOKENIZER_DATA_NAMES):
        raise FileNotFoundError(
            f"Base model needs one of {TOKENIZER_DATA_NAMES}: {base_model}"
        )


def _validate_master(master):
    if not isinstance(master, Mapping):
        raise ValueError("Checkpoint 'model' must be a state-dict mapping")
    for name, tensor in master.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise ValueError("Checkpoint 'model' must map string names to tensors")
        if tensor.dtype != torch.float32:
            raise ValueError(f"Expected FP32 master tensor {name}, got {tensor.dtype}")

    names = set(master)
    targets = {name for name in names if name.endswith(PROJECTION_SUFFIXES)}
    if targets != EXPECTED_PROJECTIONS:
        missing = sorted(EXPECTED_PROJECTIONS - targets)
        unexpected = sorted(targets - EXPECTED_PROJECTIONS)
        raise ValueError(
            f"Expected exactly {len(EXPECTED_PROJECTIONS)} projection tensors across "
            f"{NUM_LAYERS} layers, found {len(targets)}; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _atomic_save_weights(tensors, destination):
    fd, temporary_name = tempfile.mkstemp(
        prefix=".model.safetensors.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(destination, text):
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_saved_weights(destination, expected):
    """Check every saved projection and nonprojection against the FP32 master."""
    with safe_open(str(destination), framework="pt", device="cpu") as handle:
        saved_names = set(handle.keys())
        if saved_names != set(expected):
            raise RuntimeError(
                f"Saved tensor count/names differ: expected {len(expected)}, found {len(saved_names)}"
            )
        if handle.metadata() != {"format": "pt"}:
            raise RuntimeError(f"Unexpected safetensors metadata: {handle.metadata()!r}")
        for name in sorted(expected):
            master_value = expected[name]
            saved_value = handle.get_tensor(name)
            if saved_value.dtype != torch.float32:
                raise RuntimeError(f"Saved tensor is not FP32: {name} ({saved_value.dtype})")
            if not saved_value.is_contiguous():
                raise RuntimeError(f"Saved tensor is not contiguous: {name}")
            if not torch.equal(saved_value, master_value):
                category = "projection" if name in EXPECTED_PROJECTIONS else "nonprojection"
                raise RuntimeError(f"Saved {category} tensor differs from FP32 master: {name}")
    return len(saved_names)


def export_checkpoint(checkpoint, run_config, base_model, output_dir, granularity):
    """Export and verify one trusted FP32 state.pt; output_dir must not exist."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    run_config = Path(run_config).expanduser().resolve()
    base_model = Path(base_model).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if granularity not in ("row", "g32"):
        raise ValueError(f"Unknown granularity: {granularity!r}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output path already exists; refusing to overwrite: {output_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not run_config.is_file():
        raise FileNotFoundError(f"Run config does not exist: {run_config}")
    _validate_base_model(base_model)
    _validate_granularity(run_config, granularity)

    # Trusted local training checkpoints only: weights_only=False is required by
    # their optimizer/config objects. map_location avoids allocating GPU memory.
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_state, Mapping) or "model" not in checkpoint_state:
        raise ValueError("Checkpoint must contain a 'model' state dict")
    exported = dict(checkpoint_state["model"])
    del checkpoint_state
    gc.collect()  # Release the optimizer before making contiguous export copies.
    _validate_master(exported)

    # Copy one tensor at a time, releasing each original where possible. Cloning
    # also prevents safetensors failures on shared storage while preserving bits.
    for name in list(exported):
        exported[name] = exported[name].detach().cpu().contiguous().clone()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)  # exist_ok=False is the no-overwrite guarantee.
    for name in HF_ASSET_NAMES:
        source = base_model / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)

    weights_path = output_dir / "model.safetensors"
    _atomic_save_weights(exported, weights_path)
    tensor_count = _verify_saved_weights(weights_path, exported)

    manifest = {
        "export_kind": "fp32_master_conversion_intermediate",
        "description": "Unchanged FP32 master weights; no ternarization applied during HF export.",
        "conversion_pipeline": "FP32 master -> F32 GGUF -> native absmean I2_S_ROW/G32",
        "quantization_applied": False,
        "granularity": granularity,
        "quant_group": None if granularity == "row" else 32,
        "tensor_count": tensor_count,
        "projection_tensor_count": len(EXPECTED_PROJECTIONS),
        "num_hidden_layers": NUM_LAYERS,
        "source_checkpoint": str(checkpoint),
        "run_config": str(run_config),
        "base_model": str(base_model),
    }
    _atomic_write_text(
        output_dir / "export_manifest.json", json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main(argv=None):
    args = parse_args(argv)
    manifest = export_checkpoint(
        checkpoint=args.checkpoint,
        run_config=args.run_config,
        base_model=args.base_model,
        output_dir=args.output_dir,
        granularity=args.granularity,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
