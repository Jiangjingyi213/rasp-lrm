from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local envs.
    yaml = None


DEFAULT_TEMPLATE_CANDIDATES = (
    "external_code/GISP/script/GISP/c4/GISPv2_0_llama3_8b_c4.yml",
    "script/GISP/c4/GISPv2_0_llama3_8b_c4.yml",
    "configs/GISP/c4/GISPv2_0_llama3_8b_c4.yml",
)


def _load_template(repo_dir: Path) -> tuple[dict[str, Any], str | None]:
    if yaml is None:
        return {}, None
    for relative in DEFAULT_TEMPLATE_CANDIDATES:
        path = repo_dir / relative
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise TypeError(f"Official GISP template is not a YAML mapping: {path}")
            return data, str(path)
    return {}, None


def _patch_existing_keys(data: Any, replacements: dict[str, Any]) -> None:
    if isinstance(data, dict):
        for key, value in list(data.items()):
            if key in replacements:
                data[key] = replacements[key]
            else:
                _patch_existing_keys(value, replacements)
    elif isinstance(data, list):
        for value in data:
            _patch_existing_keys(value, replacements)


def _set_default(data: dict[str, Any], key: str, value: Any) -> None:
    if key not in data:
        data[key] = value


def build_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_dir = Path(args.gisp_repo_dir).resolve()
    template, template_path = _load_template(repo_dir)
    cfg = deepcopy(template)
    replacements = {
        "model": args.model,
        "model_name": args.model,
        "model_name_or_path": args.model,
        "pretrained_model": args.model,
        "pretrained_model_name_or_path": args.model,
        "dataset": "c4",
        "data": "c4",
        "calibration_dataset": "c4",
        "calibration_path": args.calibration_path,
        "seq_len": int(args.seq_len),
        "seqlen": int(args.seq_len),
        "n_samples": int(args.samples),
        "nsamples": int(args.samples),
        "samples": int(args.samples),
        "pruning_ratio": float(args.pruning_ratio),
        "prune_ratio": float(args.pruning_ratio),
        "sparsity_ratio": float(args.pruning_ratio),
        "iteration": int(args.iterations),
        "iterations": int(args.iterations),
        "iterative": True,
        "iterative_scheduling": "linear",
        "prune_skip": True,
        "prune_modules": "all",
        "prune_separate": False,
        "prune_metric": "grad_sp_global",
        "func_name": "global_grad_sp",
        "taylor": "param_first",
        "real_metrics": "first",
        "save_model": True,
        "save_model_path": args.output_model_dir,
        "output_dir": args.output_model_dir,
        "output_path": args.output_model_dir,
    }
    _patch_existing_keys(cfg, replacements)

    # These defaults make the generated YAML self-describing if the official
    # template layout changes or is not present in the cloned repository.
    _set_default(cfg, "model", args.model)
    _set_default(cfg, "dataset", "c4")
    _set_default(cfg, "calibration_path", args.calibration_path)
    _set_default(cfg, "seq_len", int(args.seq_len))
    _set_default(cfg, "n_samples", int(args.samples))
    _set_default(cfg, "pruning_ratio", float(args.pruning_ratio))
    _set_default(cfg, "prune_metric", "grad_sp_global")
    _set_default(cfg, "func_name", "global_grad_sp")
    _set_default(cfg, "prune_modules", "all")
    _set_default(cfg, "prune_separate", False)
    _set_default(cfg, "taylor", "param_first")
    _set_default(cfg, "real_metrics", "first")
    _set_default(cfg, "iterative", True)
    _set_default(cfg, "iteration", int(args.iterations))
    _set_default(cfg, "iterative_scheduling", "linear")
    _set_default(cfg, "prune_skip", True)
    _set_default(cfg, "save_model", True)
    _set_default(cfg, "save_model_path", args.output_model_dir)

    manifest = {
        "schema": "official_gisp_qwen3_8b_config_manifest_v1",
        "gisp_repo_dir": str(repo_dir),
        "template_path": template_path,
        "template_found": template_path is not None,
        "model": args.model,
        "calibration_dataset": "c4",
        "calibration_path": args.calibration_path,
        "downstream_contamination_policy": (
            "clean: C4 calibration only; no GSM8K train/test examples are used for pruning"
        ),
        "pruning_ratio": float(args.pruning_ratio),
        "iterations": int(args.iterations),
        "seq_len": int(args.seq_len),
        "samples": int(args.samples),
        "output_model_dir": args.output_model_dir,
        "notes": [
            "The launcher still validates the official GISP entrypoint before running pruning.",
            "If the upstream CLI uses a non-standard argument name, set GISP_PRUNE_CMD explicitly.",
        ],
    }
    return cfg, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Qwen3-8B C4 official-GISP config.")
    parser.add_argument("--gisp-repo-dir", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--calibration-path", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--pruning-ratio", type=float, default=0.20)
    parser.add_argument("--iterations", type=int, default=112)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    cfg, manifest = build_config(args)
    output_config = Path(args.output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as handle:
        if yaml is None:
            json.dump(cfg, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
        else:
            yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=False)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
