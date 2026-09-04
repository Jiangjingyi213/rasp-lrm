from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
    from yaml.nodes import MappingNode, ScalarNode, SequenceNode
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local envs.
    yaml = None
    MappingNode = None
    ScalarNode = None
    SequenceNode = None


DEFAULT_TEMPLATE_CANDIDATES = (
    "external_code/GISP/script/GISP/c4/GISPv2_0_llama3_8b_c4.yml",
    "script/GISP/c4/GISPv2_0_llama3_8b_c4.yml",
    "configs/GISP/c4/GISPv2_0_llama3_8b_c4.yml",
)


if yaml is not None:
    class GispYamlLoader(yaml.SafeLoader):
        pass

    def _construct_join(loader: yaml.Loader, node: Any) -> str:
        if isinstance(node, SequenceNode):
            return "".join(str(value) for value in loader.construct_sequence(node))
        if isinstance(node, ScalarNode):
            return str(loader.construct_scalar(node))
        if isinstance(node, MappingNode):
            return "".join(f"{key}{value}" for key, value in loader.construct_mapping(node).items())
        return ""

    def _construct_unknown(loader: yaml.Loader, tag_suffix: str, node: Any) -> Any:
        if isinstance(node, SequenceNode):
            return loader.construct_sequence(node)
        if isinstance(node, MappingNode):
            return loader.construct_mapping(node)
        if isinstance(node, ScalarNode):
            return loader.construct_scalar(node)
        return None

    GispYamlLoader.add_constructor("!join", _construct_join)
    GispYamlLoader.add_multi_constructor("!", _construct_unknown)
else:
    GispYamlLoader = None


def _load_template(repo_dir: Path) -> tuple[dict[str, Any], str | None]:
    if yaml is None:
        return {}, None
    for relative in DEFAULT_TEMPLATE_CANDIDATES:
        path = repo_dir / relative
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.load(handle, Loader=GispYamlLoader) or {}
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


def _as_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _parse_pipeline_nodes(raw: str) -> list[int]:
    if not raw.strip():
        return []
    nodes = []
    for item in raw.replace(",", " ").split():
        nodes.append(int(item))
    return nodes


def _patch_model_config(model_config: dict[str, Any], model_name: str) -> None:
    model_key_fragments = (
        "model",
        "tokenizer",
        "pretrained",
        "checkpoint",
        "path",
        "name_or_path",
    )
    known_model_values = ("llama", "meta-llama", "qwen", "mistral", "deepseek")
    for key, value in list(model_config.items()):
        if isinstance(value, dict):
            _patch_model_config(value, model_name)
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _patch_model_config(item, model_name)
            continue
        if not isinstance(value, str):
            continue
        key_l = str(key).lower()
        value_l = value.lower()
        if any(fragment in key_l for fragment in model_key_fragments) or any(
            fragment in value_l for fragment in known_model_values
        ):
            if key_l not in {"struct", "type", "model_type", "dtype", "torch_dtype", "device", "device_map"}:
                model_config[key] = model_name


def _replace_model_name_in_strings(data: Any, model_name: str) -> None:
    if isinstance(data, dict):
        for key, value in list(data.items()):
            if isinstance(value, str) and any(token in value for token in ("Llama", "llama", "Meta-Llama")):
                data[key] = value.replace("meta-llama/Meta-Llama-3-8B", model_name)
                data[key] = data[key].replace("Meta-Llama-3-8B", model_name.split("/")[-1])
                data[key] = data[key].replace("llama3_8b", "qwen3_8b")
            else:
                _replace_model_name_in_strings(value, model_name)
    elif isinstance(data, list):
        for value in data:
            _replace_model_name_in_strings(value, model_name)


def _apply_official_gisp_overrides(cfg: dict[str, Any], args: argparse.Namespace, repo_dir: Path) -> None:
    enable_pipeline = bool(getattr(args, "enable_pipeline", False))
    pipeline_nodes_raw = str(getattr(args, "pipeline_nodes", ""))

    system = _as_mapping(cfg, "system")
    system["device"] = "cuda"
    pipeline = _as_mapping(system, "pipeline")
    pipeline["enable_pipeline"] = enable_pipeline
    pipeline_nodes = _parse_pipeline_nodes(pipeline_nodes_raw)
    if pipeline_nodes:
        pipeline["pipeline_nodes"] = pipeline_nodes
    pipeline["size_mbs"] = int(pipeline.get("size_mbs", 2))

    if isinstance(cfg.get("model"), dict):
        _patch_model_config(cfg["model"], args.model)
    else:
        cfg["model"] = {
            "struct": "hf",
            "name_or_path": args.model,
            "model_name_or_path": args.model,
            "tokenizer_name_or_path": args.model,
        }
    model_config = cfg["model"]
    model_config["custom_modeling"] = False
    model_config["trust_remote_code"] = True

    task = _as_mapping(cfg, "task")
    task["task_mode"] = "prune"
    task["seed"] = int(task.get("seed", 0))
    task["project"] = str(task.get("project", "GISPv2"))
    task["datasets_folder"] = str(repo_dir / "data")
    prune = _as_mapping(task, "prune")
    prune_dataset = _as_mapping(prune, "prune_dataset")
    prune_dataset["type"] = str(prune_dataset.get("type", "open_domain"))
    prune_dataset["name"] = "c4"
    prune_dataset["path"] = args.calibration_path
    prune_dataset["seq_len"] = int(args.seq_len)
    prune_dataset["n_samples"] = int(args.samples)
    # The official GISP templates name this field `ratio` and use values like
    # 0.7 for a 30% prune run, so for a T20 run we set the keep ratio to 0.8.
    prune["ratio"] = 1.0 - float(args.pruning_ratio)
    prune["target_pruning_ratio"] = float(args.pruning_ratio)
    prune["batch_size"] = int(args.batch_size)
    prune["prune_metric"] = "grad_sp_global"
    prune["func_name"] = "global_grad_sp"
    prune["prune_modules"] = "all"
    prune["prune_separate"] = False
    prune["prune_skip"] = True
    prune["taylor"] = "param_first"
    prune["real_metrics"] = "first"
    prune["iterative"] = True
    prune["iteration"] = int(args.iterations)
    prune["iterative_scheduling"] = "linear"
    custom_config = _as_mapping(prune, "custom_config")
    pruner_dir = repo_dir / "external_code" / "GISP" / "pruners"
    custom_config["custom_package_location"] = str(pruner_dir)
    prune["custom_pruner"] = True

    safe_model_slug = args.model.rstrip("/").split("/")[-1].replace("/", "_")
    task["name"] = (
        f"official_gisp_{safe_model_slug}_c4_prune{float(args.pruning_ratio):.2f}_"
        f"iter{int(args.iterations)}_seq{int(args.seq_len)}_n{int(args.samples)}"
    )
    task["output_folder"] = str(Path(args.output_model_dir).parent / "00_official_gisp" / "upstream_outputs")

    evaluation = _as_mapping(cfg, "evaluation")
    evaluation["lm_eval"] = False
    evaluation["commonsense_eval"] = False
    evaluation["ppl"] = False
    for key in ("lm_eval_options", "commonsense_eval_options", "ppl_options"):
        options = _as_mapping(evaluation, key)
        options["output_path"] = args.output_model_dir

    report = _as_mapping(cfg, "report")
    report["use_wandb"] = False
    logger = _as_mapping(report, "logger")
    logger["log_file_path"] = str(Path(task["output_folder"]) / f"{task['name']}.txt")

    cfg["dataset"] = "c4"
    cfg["calibration_path"] = args.calibration_path
    cfg["seq_len"] = int(args.seq_len)
    cfg["n_samples"] = int(args.samples)
    cfg["pruning_ratio"] = float(args.pruning_ratio)
    cfg["official_keep_ratio"] = 1.0 - float(args.pruning_ratio)
    cfg["save_model"] = True
    cfg["save_model_path"] = args.output_model_dir
    _replace_model_name_in_strings(cfg, args.model)


def _validate_generated_config(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    model_config = cfg.get("model")
    if not isinstance(model_config, dict):
        raise TypeError("Generated official GISP config must keep `model` as a mapping, not a scalar.")
    if str(model_config.get("struct", "hf")) != "hf":
        raise ValueError(f"Generated official GISP config expected model.struct='hf', got {model_config.get('struct')!r}")
    if bool(model_config.get("custom_modeling", False)):
        raise ValueError(
            "Generated official GISP config must disable model.custom_modeling for Qwen3. "
            "The upstream custom modeling package used by the Llama template does not provide Qwen classes."
        )
    task = cfg.get("task")
    if not isinstance(task, dict):
        raise TypeError("Generated official GISP config must contain task mapping.")
    prune = task.get("prune")
    if not isinstance(prune, dict):
        raise TypeError("Generated official GISP config must contain task.prune mapping.")
    prune_dataset = prune.get("prune_dataset")
    if not isinstance(prune_dataset, dict):
        raise TypeError("Generated official GISP config must contain task.prune.prune_dataset mapping.")
    expected_keep_ratio = 1.0 - float(args.pruning_ratio)
    actual_keep_ratio = float(prune.get("ratio"))
    if abs(actual_keep_ratio - expected_keep_ratio) > 1e-9:
        raise ValueError(
            "Generated official GISP config has wrong task.prune.ratio: "
            f"expected keep ratio {expected_keep_ratio}, got {actual_keep_ratio}"
        )
    if str(prune_dataset.get("path")) != str(args.calibration_path):
        raise ValueError(
            "Generated official GISP config has wrong C4 calibration path: "
            f"expected {args.calibration_path}, got {prune_dataset.get('path')}"
        )
    if bool(cfg.get("evaluation", {}).get("lm_eval", False)):
        raise ValueError("Generated official GISP config should disable upstream lm_eval; downstream eval is local.")
    system = cfg.get("system")
    if not isinstance(system, dict):
        raise TypeError("Generated official GISP config must contain system mapping.")
    pipeline = system.get("pipeline")
    if not isinstance(pipeline, dict):
        raise TypeError("Generated official GISP config must contain system.pipeline mapping.")
    if bool(getattr(args, "enable_pipeline", False)) and len(pipeline.get("pipeline_nodes", [])) < 2:
        raise ValueError("GISP pipeline mode requires at least two logical CUDA pipeline nodes.")


def build_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_dir = Path(args.gisp_repo_dir).resolve()
    template, template_path = _load_template(repo_dir)
    cfg = deepcopy(template)
    replacements = {
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
    if not template:
        _patch_existing_keys(cfg, replacements)
    _apply_official_gisp_overrides(cfg, args, repo_dir)
    _validate_generated_config(cfg, args)

    # These defaults make the generated YAML self-describing if the official
    # template layout changes or is not present in the cloned repository.
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
        "official_task_prune_ratio_keep": 1.0 - float(args.pruning_ratio),
        "iterations": int(args.iterations),
        "seq_len": int(args.seq_len),
        "samples": int(args.samples),
        "batch_size": int(args.batch_size),
        "output_model_dir": args.output_model_dir,
        "pipeline_enabled": bool(getattr(args, "enable_pipeline", False)),
        "pipeline_nodes": _parse_pipeline_nodes(str(getattr(args, "pipeline_nodes", ""))),
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--enable-pipeline", action="store_true")
    parser.add_argument("--pipeline-nodes", default="")
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
