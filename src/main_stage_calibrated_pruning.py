from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

from src.data.format_prompt import build_prompt, forced_assistant_prefix
from src.data.load_gsm8k import load_tasks
from src.metrics.answer_match import answer_match, extract_answer, math_verify_available
from src.models.load_model import load_model_bundle
from src.baselines.sparsegpt_official_qwen3 import (
    prepare_sparsegpt_official_qwen3_artifact,
    summary_to_dict as sparsegpt_summary_to_dict,
)
from src.stage_calibration.artifacts import manifest_hash, stable_hash
from src.stage_calibration.calibrate import collect_stage_statistics
from src.stage_calibration.evaluate import evaluate_method, method_requires_mask_bank, uniform_ratios
from src.stage_calibration.final_shards import (
    aggregate_final_summaries,
    annotate_final_eval_indices,
    infer_shard_count,
    merge_final_shards,
    shard_dataset_dir,
    shard_summary_path,
    shard_tasks,
    validate_shard,
)
from src.stage_calibration.mask_bank import build_mask_bank, load_mask_bank, save_mask_bank
from src.stage_calibration.policy_selection import load_downstream_methods_from_selection
from src.stage_calibration.prefill import tokenize_prompt_with_prefill
from src.stage_calibration.pool import (
    allocate_source_targets,
    decontaminate,
    normalize_big_math_row,
    normalize_calibration_pool_row,
    source_allowed,
    source_counts,
    stratified_split,
)
from src.stage_calibration.protocol import (
    STAGES,
    analyze_generated_ids,
    decoded_text_has_complete_stage_answer,
)
from src.utils.io import append_jsonl, ensure_dir, read_json, read_jsonl, read_yaml, write_json, write_jsonl
from src.utils.seed import set_seed


PHASES = {
    "preflight": "00_preflight",
    "build_pool": "01_pool",
    "generate_trajectories": "02_trajectories",
    "merge_trajectory_shards": "02_trajectories",
    "select_trajectories": "03_selected",
    "calibrate_masks": "04_masks",
    "validate_masks": "04_masks",
    "evaluate_dev": "05_dev",
    "evaluate_final": "06_final",
    "merge_final_shards": "06_final",
    "prepare_sparsegpt": "07_sparsegpt",
    "summarize": ".",
}


class CompleteStageAnswerStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, protocol_start_index: int) -> None:
        self.tokenizer = tokenizer
        self.protocol_start_index = int(protocol_start_index)
        self.triggered = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated = input_ids[0, self.protocol_start_index :]
        if generated.numel() == 0:
            return False
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.triggered = decoded_text_has_complete_stage_answer(decoded)
        return self.triggered


def _load_dataset_token_kwargs(token_value: Any) -> dict[str, Any]:
    if token_value in (None, False, "false", "False", "0"):
        return {}
    parameters = inspect.signature(load_dataset).parameters
    if "token" in parameters:
        return {"token": token_value}
    if "use_auth_token" in parameters:
        return {"use_auth_token": token_value}
    return {}


def _resolve_hf_token(pool_cfg: dict[str, Any], *, default_gated: bool) -> Any:
    configured = pool_cfg.get("token", None)
    if isinstance(configured, str) and configured.startswith("$"):
        return os.environ.get(configured[1:])
    if configured is not None:
        return configured

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token

    # SynthLabsAI/Big-Math-RL-Verified is gated. Passing True lets datasets use
    # the cached huggingface-cli token when present and fails loudly when absent.
    return True if default_gated else None


def _dataset_load_error_message(pool_cfg: dict[str, Any], error: Exception) -> str:
    name = pool_cfg.get("name_or_path", "<missing>")
    endpoint = os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_BASE_URL") or "<default>"
    data_files_hint = (
        "  - 或者先把数据集 parquet 下载到服务器本地，然后在 config 的 "
        "`calibration_pool.data_files` 指向这些 parquet 文件；代码会用 "
        "`load_dataset('parquet', data_files=...)` 读取。\n"
    )
    mirror_hint = ""
    if "hf-mirror" in str(endpoint):
        mirror_hint = (
            "  - 你当前的 HF endpoint 看起来是镜像站。gated 数据集在镜像站上"
            "经常会对 dataset_infos.json 返回 403；建议本次运行显式加 "
            "`HF_ENDPOINT=https://huggingface.co`。\n"
        )
    return (
        f"Failed to load calibration dataset `{name}`.\n"
        "这通常不是 workflow 逻辑错误，而是 Big-Math-RL-Verified 是 Hugging Face "
        "gated dataset，当前服务器没有通过官方端点携带有效 token，或镜像站拒绝了元数据请求。\n"
        f"Current HF endpoint: {endpoint}\n"
        "Recommended fixes:\n"
        "  - 先在浏览器打开 https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified "
        "并接受数据集访问条款。\n"
        "  - 在远程服务器运行 `huggingface-cli login`，使用有访问权限的 token 登录。\n"
        f"{mirror_hint}"
        "  - 重新运行：`HF_ENDPOINT=https://huggingface.co PROFILE=smoke "
        "PYTHON=/home/cike/jjy/envs/rasp_qwen3_eval/bin/python "
        "bash scripts/run_stage_calibrated_pruning.sh`。\n"
        f"{data_files_hint}"
        f"Original error: {type(error).__name__}: {error}"
    )


def _load_calibration_pool_dataset(pool_cfg: dict[str, Any]):
    split = pool_cfg.get("split", "train")
    data_files = pool_cfg.get("data_files")
    name_or_path = pool_cfg["name_or_path"]
    dataset_config = pool_cfg.get("dataset_config") or pool_cfg.get("config_name") or pool_cfg.get("subset")
    default_gated = str(name_or_path).lower() == "synthlabsai/big-math-rl-verified"
    token_value = _resolve_hf_token(pool_cfg, default_gated=default_gated)

    try:
        if data_files:
            return load_dataset(
                str(pool_cfg.get("file_format", "parquet")),
                data_files=data_files,
                split=split,
                streaming=True,
                **_load_dataset_token_kwargs(token_value),
            )
        args = [name_or_path]
        if dataset_config:
            args.append(str(dataset_config))
        return load_dataset(
            *args,
            split=split,
            streaming=True,
            **_load_dataset_token_kwargs(token_value),
        )
    except Exception as exc:
        raise RuntimeError(_dataset_load_error_message(pool_cfg, exc)) from exc


def paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(
        str(cfg["workflow"]["root"]).format(
            profile=cfg["workflow"].get("profile", "smoke"),
            seed=cfg.get("seed", 1),
        )
    )
    return {
        "root": root,
        "preflight": root / "00_preflight" / "preflight.json",
        "pool": root / "01_pool" / "candidate_pool.jsonl",
        "excluded": root / "01_pool" / "excluded.jsonl",
        "pool_summary": root / "01_pool" / "summary.json",
        "candidates": root / "02_trajectories" / "candidate_trajectories.jsonl",
        "trajectory_summary": root / "02_trajectories" / "summary.json",
        "calibration": root / "03_selected" / "calibration.jsonl",
        "dev": root / "03_selected" / "dev.jsonl",
        "selection_summary": root / "03_selected" / "summary.json",
        "expansion_request": root / "03_selected" / "expansion_request.json",
        "bank": root / "04_masks" / "mask_bank.pt",
        "bank_summary": root / "04_masks" / "summary.json",
        "bank_validation": root / "04_masks" / "validation.json",
        "dev_dir": root / "05_dev",
        "dev_summary": root / "05_dev" / "summary.json",
        "frozen": root / "05_dev" / "frozen_policy.json",
        "final_dir": root / "06_final",
        "final_summary": root / "06_final" / "summary.json",
        "sparsegpt_dir": root / "07_sparsegpt",
        "sparsegpt_summary": root / "07_sparsegpt" / "summary.json",
        "workflow_summary": root / "final_summary.json",
        "workflow_gate": root / "workflow_gate.json",
    }


def profile(cfg: dict[str, Any]) -> dict[str, Any]:
    name = str(cfg["workflow"].get("profile", "smoke"))
    return cfg["profiles"][name]


def metadata(cfg: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "config_hash": stable_hash(cfg),
        "model_name": cfg["model"]["name_or_path"],
        "profile": cfg["workflow"].get("profile", "smoke"),
        **extra,
    }


def expected_bank_metadata(cfg: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    rows = read_jsonl(p["calibration"])
    return metadata(
        cfg,
        calibration_manifest_hash=manifest_hash(rows),
        tokenizer_name=cfg["model"]["name_or_path"],
        prompt_hash=stable_hash(cfg["prompt"]["structured"]),
    )


def command_preflight(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    import transformers

    checks = {
        "math_verify_available": math_verify_available(),
        "cuda_available": torch.cuda.is_available(),
        "explicit_four_stage_protocol": tuple(cfg["stages"]) == STAGES,
        "test_sets_not_calibration_sources": not {
            "gsm8k",
            "math500",
        } & {str(value).lower() for value in cfg["calibration_pool"]["allowed_sources"]},
        "ratio_grid_valid": all(0.0 <= float(value) < 1.0 for value in cfg["masks"]["ratios"]),
    }
    result = {
        "schema": "stage_calibration_preflight_v1",
        **metadata(cfg),
        "checks": checks,
        "passed": all(checks.values()),
        "environment": {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
    }
    write_json(p["preflight"], result)
    if not result["passed"]:
        raise RuntimeError(f"Preflight failed: {checks}")


def _final_dataset_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    configured = cfg.get("evaluation", {}).get("final_datasets")
    if configured:
        return [dict(row) for row in configured]
    return [
        {"dataset": "gsm8k", "split": "test"},
        {
            "dataset": "math500",
            "name_or_path": cfg["evaluation"]["math500_name_or_path"],
            "split": "test",
        },
    ]


def _final_dataset_name(dataset_cfg: dict[str, Any]) -> str:
    raw = str(dataset_cfg.get("dataset_label") or dataset_cfg.get("dataset"))
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return name or "dataset"


def _filter_final_datasets_for_env(cfg: dict[str, Any]) -> None:
    env_value = os.environ.get("STAGE_FINAL_DATASET_NAME")
    if not env_value:
        return
    requested = {value.strip() for value in env_value.split(",") if value.strip()}
    if not requested:
        return
    datasets = _final_dataset_configs(cfg)
    selected = []
    for row in datasets:
        candidates = {
            _final_dataset_name(row),
            str(row.get("dataset", "")),
            str(row.get("dataset_label", "")),
        }
        if candidates & requested:
            selected.append(row)
    if not selected:
        available = sorted(_final_dataset_name(row) for row in datasets)
        raise RuntimeError(
            f"STAGE_FINAL_DATASET_NAME={env_value!r} did not match any configured final dataset. "
            f"Available: {available}"
        )
    cfg.setdefault("evaluation", {})["final_datasets"] = selected


def _protected_final_tasks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    skip_env = os.environ.get("STAGE_PROTECTED_SKIP_DATASETS", "")
    skip_names = {value.strip() for value in skip_env.split(",") if value.strip()}
    for dataset_cfg in _final_dataset_configs(cfg):
        candidates = {
            _final_dataset_name(dataset_cfg),
            str(dataset_cfg.get("dataset", "")),
            str(dataset_cfg.get("dataset_label", "")),
        }
        if candidates & skip_names:
            continue
        tasks.extend(load_tasks(dataset_cfg))
    return tasks


def _source_name_from_cfg(source_cfg: dict[str, Any]) -> str:
    return str(
        source_cfg.get("source")
        or source_cfg.get("default_source")
        or source_cfg.get("name")
        or source_cfg.get("name_or_path")
        or "unknown_source"
    )


def _merged_source_pool_cfg(pool_cfg: dict[str, Any], source_cfg: dict[str, Any]) -> dict[str, Any]:
    inherited_keys = {
        "allowed_sources",
        "excluded_sources",
        "near_duplicate_threshold",
        "reservoir_multiplier",
        "stop_after_reservoir_target",
        "token",
        "use_auth_token",
        "file_format",
    }
    merged = {key: pool_cfg[key] for key in inherited_keys if key in pool_cfg}
    merged.update(source_cfg)
    return merged


def _collect_source_reservoir(
    *,
    source_cfg: dict[str, Any],
    target: int,
    pcfg: dict[str, Any],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = _load_calibration_pool_dataset(source_cfg)
    source_name = _source_name_from_cfg(source_cfg)
    max_scan = int(source_cfg.get("max_scan", pcfg.get("pool_max_scan", 100000)))
    reservoir_multiplier = int(source_cfg.get("reservoir_multiplier", pcfg.get("pool_reservoir_multiplier", 4)))
    reservoir_target = max(target * reservoir_multiplier, target)
    stop_after_reservoir = bool(
        source_cfg.get(
            "stop_after_reservoir_target",
            pcfg.get("pool_stop_after_reservoir_target", False),
        )
    )
    reservoir: list[dict[str, Any]] = []
    seen_allowed = 0
    invalid = 0
    scanned_rows = 0
    for index, raw in enumerate(dataset):
        scanned_rows = index + 1
        if index >= max_scan:
            break
        try:
            row = normalize_calibration_pool_row(dict(raw), index, source_cfg)
        except ValueError:
            invalid += 1
            continue
        if not source_allowed(
            row["source"],
            source_cfg.get("allowed_sources", ("*",)),
            source_cfg.get("excluded_sources", ()),
        ):
            continue
        seen_allowed += 1
        if len(reservoir) < reservoir_target:
            reservoir.append(row)
        else:
            replacement = rng.randint(0, seen_allowed - 1)
            if replacement < reservoir_target:
                reservoir[replacement] = row
        if stop_after_reservoir and len(reservoir) >= reservoir_target:
            break
    return reservoir, {
        "source": source_name,
        "target": target,
        "reservoir_rows": len(reservoir),
        "reservoir_target": reservoir_target,
        "scanned_rows": scanned_rows,
        "seen_allowed_rows": seen_allowed,
        "invalid_rows": invalid,
        "stop_after_reservoir_target": stop_after_reservoir,
    }


def _build_multi_source_pool(
    *,
    cfg: dict[str, Any],
    pcfg: dict[str, Any],
    pool_cfg: dict[str, Any],
    candidate_target: int,
    protected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sources = list(pool_cfg.get("sources") or [])
    targets = allocate_source_targets(candidate_target, sources)
    rng = random.Random(int(cfg["seed"]))
    reservoir: list[dict[str, Any]] = []
    source_summaries = []
    source_targets = {}
    for source_cfg, target in zip(sources, targets):
        merged = _merged_source_pool_cfg(pool_cfg, source_cfg)
        rows, summary = _collect_source_reservoir(
            source_cfg=merged,
            target=target,
            pcfg=pcfg,
            rng=rng,
        )
        reservoir.extend(rows)
        source_summaries.append(summary)
        source_targets[summary["source"]] = target
    kept, excluded = decontaminate(
        reservoir,
        protected,
        threshold=float(pool_cfg.get("near_duplicate_threshold", 0.80)),
    )
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        by_source[str(row["source"])].append(row)
    selected = []
    for source, target in source_targets.items():
        rows = by_source.get(source, [])
        if len(rows) < target:
            raise RuntimeError(
                f"Insufficient decontaminated candidates for source `{source}`: {len(rows)}<{target}; "
                "increase max_scan or reservoir_multiplier"
            )
        rng.shuffle(rows)
        selected.extend(rows[:target])
    rng.shuffle(selected)
    return selected, excluded, {
        "multi_source": True,
        "source_targets": source_targets,
        "source_summaries": source_summaries,
        "reservoir_rows": len(reservoir),
    }


def command_build_pool(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    pcfg = profile(cfg)
    pool_cfg = cfg["calibration_pool"]
    expansion_round = (
        int(read_json(p["expansion_request"]).get("next_round", 0))
        if p["expansion_request"].exists()
        else 0
    )
    candidate_target = int(pcfg["candidate_problems"]) + (
        expansion_round * int(pcfg.get("expansion_problems", 0))
    )
    protected = _protected_final_tasks(cfg)
    if pool_cfg.get("sources"):
        kept, excluded, extra_summary = _build_multi_source_pool(
            cfg=cfg,
            pcfg=pcfg,
            pool_cfg=pool_cfg,
            candidate_target=candidate_target,
            protected=protected,
        )
        if len(kept) < candidate_target:
            raise RuntimeError("Insufficient decontaminated mixed calibration candidates; increase max_scan")
        write_jsonl(p["pool"], kept)
        write_jsonl(p["excluded"], excluded)
        write_json(
            p["pool_summary"],
            {
                "schema": "stage_calibration_pool_v1",
                **metadata(cfg, pool_manifest_hash=manifest_hash(kept)),
                "candidate_rows": len(kept),
                "candidate_target": candidate_target,
                "expansion_round": expansion_round,
                "excluded_rows": len(excluded),
                "source_counts": source_counts(kept),
                "answer_type_counts": dict(Counter(str(row.get("answer_type", "math")) for row in kept)),
                "protected_rows": len(protected),
                **extra_summary,
            },
        )
        return
    dataset = _load_calibration_pool_dataset(pool_cfg)
    rng = random.Random(int(cfg["seed"]))
    reservoir: list[dict[str, Any]] = []
    seen_allowed = 0
    invalid = 0
    max_scan = int(pcfg.get("pool_max_scan", pool_cfg.get("max_scan", 100000)))
    reservoir_multiplier = int(pcfg.get("pool_reservoir_multiplier", pool_cfg.get("reservoir_multiplier", 4)))
    reservoir_target = max(candidate_target * reservoir_multiplier, candidate_target)
    stop_after_reservoir = bool(
        pcfg.get(
            "pool_stop_after_reservoir_target",
            pool_cfg.get("stop_after_reservoir_target", False),
        )
    )
    scanned_rows = 0
    for index, raw in enumerate(dataset):
        scanned_rows = index + 1
        if index >= max_scan:
            break
        try:
            row = normalize_big_math_row(dict(raw), index)
        except ValueError:
            invalid += 1
            continue
        if not row.get("source") and pool_cfg.get("default_source"):
            row["source"] = str(pool_cfg["default_source"])
        if row.get("domain") == "unknown" and pool_cfg.get("default_domain"):
            row["domain"] = str(pool_cfg["default_domain"])
        if not source_allowed(
            row["source"],
            pool_cfg["allowed_sources"],
            pool_cfg["excluded_sources"],
        ):
            continue
        seen_allowed += 1
        if len(reservoir) < reservoir_target:
            reservoir.append(row)
        else:
            replacement = rng.randint(0, seen_allowed - 1)
            if replacement < reservoir_target:
                reservoir[replacement] = row
        if stop_after_reservoir and len(reservoir) >= reservoir_target:
            break
    kept, excluded = decontaminate(
        reservoir,
        protected,
        threshold=float(pool_cfg.get("near_duplicate_threshold", 0.80)),
    )
    rng.shuffle(kept)
    kept = kept[:candidate_target]
    if len(kept) < candidate_target:
        raise RuntimeError("Insufficient decontaminated Big-Math candidates; increase max_scan")
    write_jsonl(p["pool"], kept)
    write_jsonl(p["excluded"], excluded)
    write_json(
        p["pool_summary"],
        {
            "schema": "stage_calibration_pool_v1",
            **metadata(cfg, pool_manifest_hash=manifest_hash(kept)),
            "candidate_rows": len(kept),
            "candidate_target": candidate_target,
            "expansion_round": expansion_round,
            "scanned_rows": scanned_rows,
            "seen_allowed_rows": seen_allowed,
            "reservoir_target": reservoir_target,
            "pool_stop_after_reservoir_target": stop_after_reservoir,
            "excluded_rows": len(excluded),
            "invalid_rows": invalid,
            "source_counts": source_counts(kept),
            "protected_rows": len(protected),
        },
    )


@torch.no_grad()
def command_generate_trajectories(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    all_rows = read_jsonl(p["pool"])
    generation_shard = _trajectory_shard_from_env()
    indexed_rows = [
        (problem_index, row)
        for problem_index, row in enumerate(all_rows)
        if generation_shard is None or problem_index % generation_shard["count"] == generation_shard["index"]
    ]
    rows = [row for _, row in indexed_rows]
    pcfg = profile(cfg)
    repeats = int(pcfg["generations_per_problem"])
    bundle = load_model_bundle(cfg["model"])
    output = _trajectory_candidate_path(p, generation_shard)
    existing_rows = read_jsonl(output) if output.exists() else []
    existing_keys = {
        (str(row["id"]), int(row["sample_index"])) for row in existing_rows
    }
    generation = cfg["generation"]
    prompt_cfg = cfg["prompt"]["structured"]
    eos = bundle.tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    desc = "generate-stage-calibration-trajectories"
    if generation_shard is not None:
        desc += f"-shard-{generation_shard['index']}-of-{generation_shard['count']}"
    for problem_index, row in tqdm(indexed_rows, desc=desc):
        prompt = build_prompt(row["question"], bundle.tokenizer, prompt_cfg)
        prefill = forced_assistant_prefix(prompt_cfg)
        inputs, prompt_ids, prefill_ids = tokenize_prompt_with_prefill(
            bundle.tokenizer,
            prompt,
            prefill,
            max_input_tokens=int(generation["max_input_tokens"]),
            device=bundle.device,
        )
        input_length = len(prompt_ids) + len(prefill_ids)
        for sample_index in range(repeats):
            if (str(row["id"]), sample_index) in existing_keys:
                continue
            sample_seed = int(cfg["seed"]) + problem_index * repeats + sample_index
            torch.manual_seed(sample_seed)
            stopping_criteria = CompleteStageAnswerStoppingCriteria(
                bundle.tokenizer,
                len(prompt_ids),
            )
            out = bundle.model.generate(
                **inputs,
                max_new_tokens=int(generation["max_new_tokens"]),
                do_sample=True,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                top_k=int(generation["top_k"]),
                pad_token_id=bundle.tokenizer.pad_token_id,
                eos_token_id=bundle.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([stopping_criteria]),
            )
            continuation = [int(value) for value in out[0, input_length:].cpu().tolist()]
            generated = [*prefill_ids, *continuation]
            completion = bundle.tokenizer.decode(generated, skip_special_tokens=True).strip()
            stage_protocol = analyze_generated_ids(bundle.tokenizer, generated)
            ended_with_eos = bool(generated and generated[-1] in eos_ids)
            append_jsonl(
                output,
                {
                    **row,
                    "problem_index": problem_index,
                    "sample_index": sample_index,
                    "sample_seed": sample_seed,
                    "generation_shard": generation_shard,
                    "prompt": prompt,
                    "prompt_token_ids": prompt_ids,
                    "generated_token_ids": generated,
                    "completion": completion,
                    "prediction": extract_answer(completion),
                    "correct": answer_match(completion, row["gold"], answer_type=row.get("answer_type")),
                    "ended_with_eos": ended_with_eos,
                    "stopped_after_complete_stage_answer": stopping_criteria.triggered,
                    "truncated": not ended_with_eos and len(generated) >= int(generation["max_new_tokens"]),
                    "stage_protocol": stage_protocol,
                },
            )
    current_ids = {str(row["id"]) for row in rows}
    generated_rows = [row for row in read_jsonl(output) if str(row["id"]) in current_ids]
    summary_path = _trajectory_summary_path(p, generation_shard)
    write_json(
        summary_path,
        {
            "schema": "stage_calibration_candidate_trajectories_v1",
            **metadata(cfg, pool_manifest_hash=manifest_hash(all_rows)),
            "rows": len(generated_rows),
            "problems": len(rows),
            "full_problems": len(all_rows),
            "generation_shard": generation_shard,
            "correct_rate": _safe_rate(sum(int(row["correct"]) for row in generated_rows), len(generated_rows)),
            "valid_stage_rate": _safe_rate(
                sum(int(row["stage_protocol"]["valid"]) for row in generated_rows),
                len(generated_rows),
            ),
            "truncation_rate": _safe_rate(sum(int(row["truncated"]) for row in generated_rows), len(generated_rows)),
        },
    )


def command_merge_trajectory_shards(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    pool_rows = read_jsonl(p["pool"])
    pcfg = profile(cfg)
    repeats = int(pcfg["generations_per_problem"])
    shard_count = _trajectory_shard_count_from_env_or_files(p)
    id_to_problem_index = {str(row["id"]): index for index, row in enumerate(pool_rows)}
    merged: list[dict[str, Any]] = []
    shard_summaries = []
    for shard_index in range(shard_count):
        shard = {"index": shard_index, "count": shard_count}
        candidate_path = _trajectory_candidate_path(p, shard)
        summary_path = _trajectory_summary_path(p, shard)
        if not candidate_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing trajectory shard {shard_index}/{shard_count}: {candidate_path}")
        shard_rows = [
            row
            for row in read_jsonl(candidate_path)
            if str(row.get("id")) in id_to_problem_index
            and id_to_problem_index[str(row["id"])] % shard_count == shard_index
        ]
        expected_rows = sum(1 for index in range(len(pool_rows)) if index % shard_count == shard_index) * repeats
        if len(shard_rows) != expected_rows:
            raise RuntimeError(
                f"Trajectory shard {shard_index}/{shard_count} has {len(shard_rows)} rows; "
                f"expected {expected_rows}. Check shard log before merging."
            )
        merged.extend(shard_rows)
        shard_summaries.append(read_json(summary_path))
    seen_keys = set()
    duplicates = []
    for row in merged:
        key = (str(row.get("id")), int(row.get("sample_index", -1)))
        if key in seen_keys:
            duplicates.append(key)
        seen_keys.add(key)
    if duplicates:
        raise RuntimeError(f"Duplicate trajectory rows found while merging shards: {duplicates[:5]}")
    merged.sort(key=lambda row: (id_to_problem_index[str(row["id"])], int(row.get("sample_index", 0))))
    expected_total = len(pool_rows) * repeats
    if len(merged) != expected_total:
        raise RuntimeError(f"Merged trajectory rows {len(merged)} != expected {expected_total}")
    write_jsonl(p["candidates"], merged)
    write_json(
        p["trajectory_summary"],
        {
            "schema": "stage_calibration_candidate_trajectories_v1",
            **metadata(cfg, pool_manifest_hash=manifest_hash(pool_rows)),
            "rows": len(merged),
            "problems": len(pool_rows),
            "correct_rate": _safe_rate(sum(int(row["correct"]) for row in merged), len(merged)),
            "valid_stage_rate": _safe_rate(sum(int(row["stage_protocol"]["valid"]) for row in merged), len(merged)),
            "truncation_rate": _safe_rate(sum(int(row["truncated"]) for row in merged), len(merged)),
            "trajectory_sharded": True,
            "trajectory_shard_count": shard_count,
            "trajectory_shard_summaries": shard_summaries,
        },
    )


def command_select_trajectories(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    pool_ids = {str(row["id"]) for row in read_jsonl(p["pool"])}
    rows = [row for row in read_jsonl(p["candidates"]) if str(row["id"]) in pool_ids]
    rows = _refresh_candidate_metadata(cfg, rows)
    pcfg = profile(cfg)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)
    eligible = []
    rejection = Counter()
    for values in grouped.values():
        correct_count = sum(int(row["correct"]) for row in values)
        if not _passes_difficulty_filter(correct_count, len(values), cfg, pcfg):
            rejection["difficulty_filter"] += 1
            continue
        candidates = [
            row
            for row in values
            if row["correct"] and row["stage_protocol"]["valid"] and not row["truncated"]
        ]
        if not candidates:
            rejection["no_correct_valid_complete_trajectory"] += 1
            continue
        lengths = sorted(len(row["generated_token_ids"]) for row in candidates)
        median = lengths[len(lengths) // 2]
        selected = min(candidates, key=lambda row: (abs(len(row["generated_token_ids"]) - median), row["sample_index"]))
        selected["selection_reason"] = "correct_valid_complete_closest_to_problem_median_length"
        eligible.append(selected)
    requested_calibration = int(pcfg["calibration_problems"])
    requested_dev = int(pcfg["dev_problems"])
    required = requested_calibration + requested_dev
    calibration_size = requested_calibration
    dev_size = requested_dev
    adaptive_selection = False
    if len(eligible) < required:
        adapted = _adaptive_smoke_selection_sizes(cfg, len(eligible), requested_calibration, requested_dev)
        if adapted is None:
            _write_selection_failure_summary(cfg, p, rows, grouped, eligible, rejection, required)
            _request_expansion_or_fail(
                cfg,
                p,
                pcfg,
                reason=f"eligible_problems:{len(eligible)}<{required}",
            )
        calibration_size, dev_size = adapted
        adaptive_selection = True
    min_stage_tokens = int(pcfg["min_stage_tokens"])
    min_stage_tokens_by_stage = _stage_token_thresholds(pcfg, min_stage_tokens)
    selection_strategy = "stratified"
    calibration, dev = stratified_split(
        eligible,
        calibration_size,
        dev_size,
        int(cfg["seed"]),
    )
    stage_tokens = _stage_token_totals(calibration)
    if not _stage_token_gate_passed(stage_tokens, min_stage_tokens_by_stage) and bool(
        pcfg.get("stage_token_balanced_selection", False)
    ):
        calibration, dev = _stage_token_balanced_split(
            eligible,
            calibration_size,
            dev_size,
            int(cfg["seed"]),
            min_stage_tokens_by_stage,
        )
        stage_tokens = _stage_token_totals(calibration)
        selection_strategy = "stage_token_balanced"
    token_gate_passed = all(
        stage_tokens[stage] >= min_stage_tokens_by_stage[stage] for stage in STAGES
    )
    token_gate_relaxed = (
        cfg["workflow"].get("profile") == "smoke"
        and not token_gate_passed
        and all(stage_tokens[stage] > 0 for stage in STAGES)
    )
    passed = token_gate_passed or token_gate_relaxed
    write_jsonl(p["calibration"], calibration)
    write_jsonl(p["dev"], dev)
    write_json(
        p["selection_summary"],
        {
            "schema": "stage_calibration_selection_v1",
            **metadata(
                cfg,
                calibration_manifest_hash=manifest_hash(calibration),
                dev_manifest_hash=manifest_hash(dev),
            ),
            "eligible_problems": len(eligible),
            "requested_calibration_problems": requested_calibration,
            "requested_dev_problems": requested_dev,
            "adaptive_selection": adaptive_selection,
            "calibration_problems": len(calibration),
            "dev_problems": len(dev),
            "stage_content_tokens": dict(stage_tokens),
            "minimum_stage_tokens": min_stage_tokens,
            "minimum_stage_tokens_by_stage": min_stage_tokens_by_stage,
            "selection_strategy": selection_strategy,
            "stage_token_gate_passed": token_gate_passed,
            "stage_token_gate_relaxed": token_gate_relaxed,
            "difficulty_filter": _difficulty_filter_mode(cfg, pcfg),
            "rejection_counts": dict(rejection),
            "passed": passed,
        },
    )
    if not passed:
        _request_expansion_or_fail(
            cfg,
            p,
            pcfg,
            reason="selected calibration trajectories do not meet per-stage token gate",
        )
    p["expansion_request"].unlink(missing_ok=True)


def _stage_token_thresholds(pcfg: dict[str, Any], default_value: int) -> dict[str, int]:
    overrides = pcfg.get("min_stage_tokens_by_stage", {})
    return {stage: int(overrides.get(stage, default_value)) for stage in STAGES}


def _stage_token_counts(row: dict[str, Any]) -> Counter[str]:
    stages = row.get("stage_protocol", {}).get("token_stages", [])
    return Counter(stage for stage in stages if stage in STAGES)


def _stage_token_totals(rows: list[dict[str, Any]]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for row in rows:
        totals.update(_stage_token_counts(row))
    return totals


def _stage_token_gate_passed(
    stage_tokens: Counter[str],
    min_stage_tokens_by_stage: dict[str, int],
) -> bool:
    return all(stage_tokens[stage] >= min_stage_tokens_by_stage[stage] for stage in STAGES)


def _stage_token_balanced_split(
    rows: list[dict[str, Any]],
    calibration_size: int,
    dev_size: int,
    seed: int,
    min_stage_tokens_by_stage: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if calibration_size + dev_size > len(rows):
        raise ValueError("Not enough eligible rows for disjoint calibration/dev split")
    rng = random.Random(seed)
    remaining = list(rows)
    rng.shuffle(remaining)
    calibration: list[dict[str, Any]] = []
    stage_tokens: Counter[str] = Counter()

    while remaining and len(calibration) < calibration_size:
        deficits = {
            stage: max(0, int(min_stage_tokens_by_stage[stage]) - stage_tokens[stage])
            for stage in STAGES
        }
        if any(deficits.values()):
            best_index = max(
                range(len(remaining)),
                key=lambda index: _stage_token_balance_score(remaining[index], deficits),
            )
        else:
            best_index = 0
        row = remaining.pop(best_index)
        calibration.append(row)
        stage_tokens.update(_stage_token_counts(row))

    dev = stratified_split(remaining, 0, dev_size, seed + 7919)[1]
    if {row["id"] for row in calibration} & {row["id"] for row in dev}:
        raise ValueError("Calibration/dev split overlap")
    return calibration, dev


def _stage_token_balance_score(row: dict[str, Any], deficits: dict[str, int]) -> tuple[float, int, int, int]:
    counts = _stage_token_counts(row)
    normalized_gain = sum(
        min(counts[stage], deficits[stage]) / max(1, deficits[stage])
        for stage in STAGES
        if deficits[stage] > 0
    )
    raw_gain = sum(min(counts[stage], deficits[stage]) for stage in STAGES)
    scarce_stage_gain = counts["verify"] + counts["final"]
    total_tokens = sum(counts.values())
    return (normalized_gain, raw_gain, scarce_stage_gain, total_tokens)


def _adaptive_smoke_selection_sizes(
    cfg: dict[str, Any],
    eligible_count: int,
    requested_calibration: int,
    requested_dev: int,
) -> tuple[int, int] | None:
    if cfg["workflow"].get("profile") != "smoke" or eligible_count < 2:
        return None
    dev_size = min(requested_dev, max(1, eligible_count // 3))
    calibration_size = min(requested_calibration, eligible_count - dev_size)
    if calibration_size < 1 or dev_size < 1:
        return None
    return calibration_size, dev_size


def _passes_difficulty_filter(
    correct_count: int,
    attempts: int,
    cfg: dict[str, Any],
    pcfg: dict[str, Any],
) -> bool:
    mode = _difficulty_filter_mode(cfg, pcfg)
    if mode == "moderate":
        return 1 <= correct_count < attempts
    if mode == "any_correct":
        return correct_count >= 1
    raise ValueError(f"Unknown difficulty_filter mode: {mode}")


def _difficulty_filter_mode(cfg: dict[str, Any], pcfg: dict[str, Any]) -> str:
    return str(
        pcfg.get(
            "difficulty_filter",
            cfg.get("selection", {}).get("difficulty_filter", "moderate"),
        )
    )


def _refresh_candidate_metadata(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"],
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    refreshed = []
    for row in rows:
        row = dict(row)
        generated = [int(value) for value in row.get("generated_token_ids", [])]
        if generated:
            row["stage_protocol"] = analyze_generated_ids(tokenizer, generated)
        completion = str(row.get("completion", ""))
        gold = str(row.get("gold", ""))
        row["prediction"] = extract_answer(completion)
        row["correct"] = answer_match(completion, gold, answer_type=row.get("answer_type"))
        refreshed.append(row)
    return refreshed


def _write_selection_failure_summary(
    cfg: dict[str, Any],
    p: dict[str, Path],
    rows: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    eligible: list[dict[str, Any]],
    rejection: Counter,
    required: int,
) -> None:
    correct_counts = Counter(str(sum(int(row["correct"]) for row in values)) for values in grouped.values())
    fallback_reasons = Counter(
        str(row.get("stage_protocol", {}).get("fallback_reason") or "valid")
        for row in rows
    )
    detected_by = Counter(str(row.get("stage_protocol", {}).get("detected_by", "unknown")) for row in rows)
    row_count = len(rows)
    write_json(
        p["selection_summary"],
        {
            "schema": "stage_calibration_selection_v1",
            **metadata(cfg),
            "passed": False,
            "failure_reason": f"eligible_problems:{len(eligible)}<{required}",
            "eligible_problems": len(eligible),
            "required_problems": required,
            "candidate_rows": row_count,
            "candidate_problems": len(grouped),
            "row_correct_rate": (
                sum(int(row["correct"]) for row in rows) / row_count if row_count else None
            ),
            "row_valid_stage_rate": (
                sum(int(row.get("stage_protocol", {}).get("valid")) for row in rows) / row_count
                if row_count
                else None
            ),
            "row_truncation_rate": (
                sum(int(row.get("truncated", False)) for row in rows) / row_count
                if row_count
                else None
            ),
            "problem_correct_count_histogram": dict(correct_counts),
            "stage_fallback_reasons": dict(fallback_reasons),
            "stage_detected_by": dict(detected_by),
            "difficulty_filter": _difficulty_filter_mode(cfg, profile(cfg)),
            "rejection_counts": dict(rejection),
        },
    )


def _request_expansion_or_fail(
    cfg: dict[str, Any],
    p: dict[str, Path],
    pcfg: dict[str, Any],
    *,
    reason: str,
) -> None:
    expansion = int(pcfg.get("expansion_problems", 0))
    if cfg["workflow"]["profile"] != "formal" or expansion <= 0:
        raise RuntimeError(reason)
    current_round = (
        int(read_json(p["expansion_request"]).get("next_round", 0))
        if p["expansion_request"].exists()
        else 0
    )
    write_json(
        p["expansion_request"],
        {
            "schema": "stage_calibration_expansion_request_v1",
            "next_round": current_round + 1,
            "add_candidate_problems": expansion,
            "reason": reason,
        },
    )
    print(f"{reason}; formal workflow requested another {expansion} candidate problems")
    raise SystemExit(42)


def _safe_rate(numerator: int | float, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def _trajectory_shard_from_env() -> dict[str, int] | None:
    count = os.environ.get("STAGE_GENERATE_SHARD_COUNT")
    index = os.environ.get("STAGE_GENERATE_SHARD_INDEX")
    if count is None and index is None:
        return None
    if count is None or index is None:
        raise RuntimeError("Set both STAGE_GENERATE_SHARD_INDEX and STAGE_GENERATE_SHARD_COUNT")
    shard = {"index": int(index), "count": int(count)}
    validate_shard(shard_index=shard["index"], shard_count=shard["count"])
    return shard


def _trajectory_shards_dir(p: dict[str, Path]) -> Path:
    return p["root"] / "02_trajectories" / "shards"


def _trajectory_candidate_path(p: dict[str, Path], shard: dict[str, int] | None) -> Path:
    if shard is None:
        return p["candidates"]
    return (
        _trajectory_shards_dir(p)
        / f"candidate_trajectories_shard_{shard['index']:05d}_of_{shard['count']:05d}.jsonl"
    )


def _trajectory_summary_path(p: dict[str, Path], shard: dict[str, int] | None) -> Path:
    if shard is None:
        return p["trajectory_summary"]
    return (
        _trajectory_shards_dir(p)
        / f"summary_shard_{shard['index']:05d}_of_{shard['count']:05d}.json"
    )


def _trajectory_shard_count_from_env_or_files(p: dict[str, Path]) -> int:
    env_count = os.environ.get("STAGE_GENERATE_SHARD_COUNT")
    if env_count is not None:
        return int(env_count)
    counts = set()
    for path in _trajectory_shards_dir(p).glob("summary_shard_*_of_*.json"):
        match = re.match(r"summary_shard_\d+_of_(\d+)\.json$", path.name)
        if match:
            counts.add(int(match.group(1)))
    if not counts:
        raise FileNotFoundError(f"No trajectory shard summaries found in {_trajectory_shards_dir(p)}")
    if len(counts) != 1:
        raise ValueError(f"Conflicting trajectory shard counts found: {sorted(counts)}")
    return counts.pop()


def command_calibrate_masks(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    rows = read_jsonl(p["calibration"])
    bundle = load_model_bundle(cfg["model"])
    c4_samples = _effective_c4_samples(cfg)
    metrics, means, output_norms, stats_summary = collect_stage_statistics(
        bundle.model,
        bundle.tokenizer,
        rows,
        c4_samples=c4_samples,
        max_input_tokens=int(cfg["generation"]["max_input_tokens"]),
        forward_chunk_tokens=int(cfg.get("calibration", {}).get("forward_chunk_tokens", 1024)),
    )
    stats_summary["effective_c4_samples"] = c4_samples
    bank_metadata = metadata(
        cfg,
        calibration_manifest_hash=manifest_hash(rows),
        tokenizer_name=cfg["model"]["name_or_path"],
        prompt_hash=stable_hash(cfg["prompt"]["structured"]),
    )
    bank = build_mask_bank(
        metadata=bank_metadata,
        metrics=metrics,
        means=means,
        ratios=[float(value) for value in cfg["masks"]["ratios"]],
        output_norms=output_norms,
    )
    save_mask_bank(p["bank"], bank)
    write_json(p["bank_summary"], {"schema": "stage_calibrated_mask_summary_v1", **bank_metadata, **stats_summary})


def _effective_c4_samples(cfg: dict[str, Any]) -> int:
    env_override = os.environ.get("STAGE_C4_SAMPLES")
    if env_override is not None:
        return int(env_override)
    pcfg = profile(cfg)
    if "c4_samples" in pcfg:
        return int(pcfg["c4_samples"])
    return int(cfg["masks"]["c4_samples"])


def _jaccard_masks(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.bool()
    right = right.bool()
    union = left | right
    return float((left & right).sum().item() / max(1, union.sum().item()))


def command_validate_masks(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    bank = load_mask_bank(p["bank"], expected_bank_metadata(cfg, p))
    c4_enabled = _effective_c4_samples(cfg) > 0
    overlaps = {}
    for ratio in bank["ratios"]:
        key = f"{float(ratio):.4f}"
        overlaps[key] = {}
        for left_index, left in enumerate(STAGES):
            for right in STAGES[left_index + 1 :]:
                values = [
                    _jaccard_masks(
                        bank["policies"]["stage_specific"][left][layer_id]["masks"][key],
                        bank["policies"]["stage_specific"][right][layer_id]["masks"][key],
                    )
                    for layer_id in bank["layers"]
                ]
                overlaps[key][f"{left}:{right}"] = sum(values) / len(values)
    rank_correlations = {}
    rank_pairs = [
        ("trajectory_global", "prompt_only_global"),
        ("trajectory_global", "stage_balanced_global"),
    ]
    if c4_enabled:
        rank_pairs.insert(0, ("trajectory_global", "c4_global"))
    for left, right in rank_pairs:
        values = []
        for layer_id in bank["layers"]:
            x = bank["policies"][left]["reasoning"][layer_id]["metric"].float()
            y = bank["policies"][right]["reasoning"][layer_id]["metric"].float()
            x_rank = torch.argsort(torch.argsort(x)).float()
            y_rank = torch.argsort(torch.argsort(y)).float()
            values.append(float(torch.corrcoef(torch.stack([x_rank, y_rank]))[0, 1].item()))
        rank_correlations[f"{left}:{right}"] = sum(values) / len(values)
    write_json(
        p["bank_validation"],
        {
            "schema": "stage_calibrated_mask_validation_v1",
            **metadata(cfg, calibration_manifest_hash=manifest_hash(read_jsonl(p["calibration"]))),
            "passed": True,
            "policies": sorted(bank["policies"]),
            "ratios": bank["ratios"],
            "layers": len(bank["layers"]),
            "c4_evaluated": c4_enabled,
            "stage_mask_mean_jaccard": overlaps,
            "mean_layerwise_spearman_rank_correlation": rank_correlations,
        },
    )


def structured_prompt(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg["prompt"]["structured"])


def ordinary_prompt(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg["prompt"]["ordinary"])


def method(
    name: str,
    policy: str,
    ratios: dict[str, float],
    prompt: dict[str, Any],
    bias: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "policy": policy,
        "stage_ratios": ratios,
        "prompt": prompt,
        "bias_compensation": bias,
        **extra,
    }


def _pruning_target_fields(row: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if "target_pruning_ratio" in row:
        fields["target_pruning_ratio"] = float(row["target_pruning_ratio"])
    if "min_target_pruning_ratio" in row:
        fields["min_target_pruning_ratio"] = float(row["min_target_pruning_ratio"])
    if "max_target_pruning_ratio" in row:
        fields["max_target_pruning_ratio"] = float(row["max_target_pruning_ratio"])
    if "target_pruning_label" in row:
        fields["target_pruning_label"] = str(row["target_pruning_label"])
    return fields


def _adaptive_griffin_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg.get("adaptive_griffin", {}))


def _adaptive_griffin_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_adaptive_griffin_cfg(cfg).get("enabled", False))


def _griffin_prompt_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg.get("griffin_prompt", {}))


def _griffin_prompt_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_griffin_prompt_cfg(cfg).get("enabled", False))


def _wanda_official_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg.get("wanda_official", {}))


def _wanda_official_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_wanda_official_cfg(cfg).get("enabled", False))


def _sparsegpt_official_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg.get("sparsegpt_official", {}))


def _sparsegpt_official_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_sparsegpt_official_cfg(cfg).get("enabled", False))


def _shortgpt_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg.get("shortgpt", {}))


def _shortgpt_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_shortgpt_cfg(cfg).get("enabled", False))


def _limits_layer_pruning_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg.get("limits_layer_pruning", {}))


def _limits_layer_pruning_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_limits_layer_pruning_cfg(cfg).get("enabled", False))


def _flap_mlp_official_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg.get("flap_mlp_official", {}))


def _flap_mlp_official_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_flap_mlp_official_cfg(cfg).get("enabled", False))


def _llm_pruner_mlp_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg.get("llm_pruner_mlp_static_width", {}))


def _llm_pruner_mlp_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_llm_pruner_mlp_cfg(cfg).get("enabled", False))


def _griffin_prompt_method_from_cfg(row: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    prune_ratio = float(row.get("prune_ratio", row.get("ratio", 0.0)))
    name = str(row.get("method_name", row.get("name", f"griffin_prompt_{prune_ratio:.4f}".replace(".", "p"))))
    return method(
        name,
        "griffin_prompt",
        uniform_ratios(prune_ratio),
        prompt,
        bias=False,
        prune_ratio=prune_ratio,
        density=1.0 - prune_ratio,
        selection_method=str(row.get("selection_method", "topk")),
        baseline_type="prompt_prompted_dynamic",
        variant_role=str(row.get("variant_role", name)),
        selection_note=str(row.get("selection_note", "")),
        **(
            {"target_pruning_ratio": float(row["target_pruning_ratio"])}
            if "target_pruning_ratio" in row
            else {}
        ),
        **(
            {"target_pruning_label": str(row["target_pruning_label"])}
            if "target_pruning_label" in row
            else {}
        ),
    )


def griffin_prompt_methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    gcfg = _griffin_prompt_cfg(cfg)
    variants = gcfg.get("variants")
    if variants is None:
        if not gcfg:
            return []
        variants = [gcfg]
    if not isinstance(variants, list) or not variants:
        raise ValueError("griffin_prompt.variants must be a non-empty list when provided")
    methods = []
    for index, row in enumerate(variants):
        if not isinstance(row, dict):
            raise ValueError(f"griffin_prompt.variants[{index}] must be a mapping")
        merged = deepcopy(gcfg)
        merged.pop("variants", None)
        merged.update(deepcopy(row))
        methods.append(_griffin_prompt_method_from_cfg(merged, structured_prompt(cfg)))
    names = [row["name"] for row in methods]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"GRIFFIN prompt method names must be unique: {duplicates}")
    return methods


def _wanda_calibration_path(row: dict[str, Any]) -> str:
    if row.get("calibration_path"):
        return str(row["calibration_path"])
    source_root = os.environ.get(
        "SOURCE_ROOT",
        "runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3",
    )
    return str(Path(source_root) / "03_selected" / "calibration.jsonl")


def _wanda_official_method_from_cfg(row: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    sparsity = float(row.get("sparsity_ratio", row.get("ratio", 0.0)))
    name = str(row.get("method_name", row.get("name", f"wanda_official_{sparsity:.4f}".replace(".", "p"))))
    return method(
        name,
        "wanda_official",
        uniform_ratios(sparsity),
        prompt,
        bias=False,
        sparsity_ratio=sparsity,
        density=1.0 - sparsity,
        calibration_path=_wanda_calibration_path(row),
        calibration_samples=int(row.get("calibration_samples", 128)),
        calibration_max_input_tokens=int(row.get("calibration_max_input_tokens", 2048)),
        calibration_prompt_mode=str(row.get("calibration_prompt_mode", "structured_prompt")),
        calibration_text_field=str(row.get("calibration_text_field", "text")),
        **(
            {"calibration_seed": int(row["calibration_seed"])}
            if row.get("calibration_seed") is not None
            else {}
        ),
        target_modules=list(row.get("target_modules") or []) or None,
        baseline_type="official_style_wanda",
        pruning_granularity="weight_unstructured",
        matched_rasp_reference=str(row.get("matched_rasp_reference", "")),
        variant_role=str(row.get("variant_role", name)),
        selection_note=str(row.get("selection_note", "")),
        **(
            {"target_pruning_ratio": float(row["target_pruning_ratio"])}
            if "target_pruning_ratio" in row
            else {}
        ),
        **(
            {"target_pruning_label": str(row["target_pruning_label"])}
            if "target_pruning_label" in row
            else {}
        ),
    )


def wanda_official_methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    wcfg = _wanda_official_cfg(cfg)
    variants = wcfg.get("variants")
    if variants is None:
        if not wcfg:
            return []
        variants = [wcfg]
    if not isinstance(variants, list) or not variants:
        raise ValueError("wanda_official.variants must be a non-empty list when provided")
    methods = []
    for index, row in enumerate(variants):
        if not isinstance(row, dict):
            raise ValueError(f"wanda_official.variants[{index}] must be a mapping")
        merged = deepcopy(wcfg)
        merged.pop("variants", None)
        merged.update(deepcopy(row))
        methods.append(_wanda_official_method_from_cfg(merged, structured_prompt(cfg)))
    names = [row["name"] for row in methods]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Wanda official method names must be unique: {duplicates}")
    return methods


def _sparsegpt_calibration_path(row: dict[str, Any]) -> str:
    if row.get("calibration_path"):
        return str(row["calibration_path"])
    source_root = os.environ.get(
        "SOURCE_ROOT",
        "runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3",
    )
    return str(Path(source_root) / "03_selected" / "calibration.jsonl")


def _sparsegpt_artifact_root(cfg: dict[str, Any]) -> Path:
    env_root = os.environ.get("SPARSEGPT_ARTIFACT_ROOT")
    if env_root:
        return Path(env_root)
    scfg = _sparsegpt_official_cfg(cfg)
    if scfg.get("artifact_root"):
        return Path(str(scfg["artifact_root"]))
    model_name = str(cfg["model"]["name_or_path"]).rstrip("/").split("/")[-1].lower().replace("-", "_")
    return Path("runs/08_stage_calibrated_pruning/sparsegpt_artifacts") / model_name


def _sparsegpt_artifact_path(cfg: dict[str, Any], row: dict[str, Any], name: str) -> str:
    if row.get("artifact_path"):
        return str(row["artifact_path"])
    return str(_sparsegpt_artifact_root(cfg) / name)


def _sparsegpt_official_method_from_cfg(
    cfg: dict[str, Any],
    row: dict[str, Any],
    prompt: dict[str, Any],
) -> dict[str, Any]:
    sparsity = float(row.get("sparsity_ratio", row.get("ratio", 0.0)))
    name = str(row.get("method_name", row.get("name", f"sparsegpt_official_{sparsity:.4f}".replace(".", "p"))))
    return method(
        name,
        "sparsegpt_official",
        uniform_ratios(sparsity),
        prompt,
        bias=False,
        sparsity_ratio=sparsity,
        density=1.0 - sparsity,
        calibration_path=_sparsegpt_calibration_path(row),
        calibration_samples=int(row.get("calibration_samples", 128)),
        calibration_max_input_tokens=int(row.get("calibration_max_input_tokens", 2048)),
        calibration_batch_size=int(
            os.environ.get(
                "SPARSEGPT_CALIBRATION_BATCH_SIZE",
                row.get("calibration_batch_size", 1),
            )
        ),
        target_modules=list(row.get("target_modules") or []) or None,
        blocksize=int(row.get("blocksize", 128)),
        percdamp=float(row.get("percdamp", 0.01)),
        artifact_path=_sparsegpt_artifact_path(cfg, row, name),
        artifact_save_dtype=str(row.get("artifact_save_dtype", "float16")),
        baseline_type="official_style_sparsegpt",
        pruning_granularity="weight_unstructured",
        matched_rasp_reference=str(row.get("matched_rasp_reference", "")),
        variant_role=str(row.get("variant_role", name)),
        selection_note=str(row.get("selection_note", "")),
        **(
            {"target_pruning_ratio": float(row["target_pruning_ratio"])}
            if "target_pruning_ratio" in row
            else {}
        ),
        **(
            {"target_pruning_label": str(row["target_pruning_label"])}
            if "target_pruning_label" in row
            else {}
        ),
    )


def sparsegpt_official_methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    scfg = _sparsegpt_official_cfg(cfg)
    variants = scfg.get("variants")
    if variants is None:
        if not scfg:
            return []
        variants = [scfg]
    if not isinstance(variants, list) or not variants:
        raise ValueError("sparsegpt_official.variants must be a non-empty list when provided")
    methods = []
    for index, row in enumerate(variants):
        if not isinstance(row, dict):
            raise ValueError(f"sparsegpt_official.variants[{index}] must be a mapping")
        merged = deepcopy(scfg)
        merged.pop("variants", None)
        merged.update(deepcopy(row))
        methods.append(_sparsegpt_official_method_from_cfg(cfg, merged, structured_prompt(cfg)))
    names = [row["name"] for row in methods]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"SparseGPT official method names must be unique: {duplicates}")
    return methods


def _shortgpt_calibration_path(row: dict[str, Any]) -> str:
    if row.get("calibration_path"):
        return str(row["calibration_path"])
    source_root = os.environ.get(
        "SOURCE_ROOT",
        "runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3",
    )
    return str(Path(source_root) / "03_selected" / "calibration.jsonl")


def _shortgpt_method_from_cfg(row: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    prune_ratio = float(row.get("prune_ratio", row.get("ratio", 0.0)))
    name = str(row.get("method_name", row.get("name", f"shortgpt_{prune_ratio:.4f}".replace(".", "p"))))
    extra: dict[str, Any] = {}
    for key in (
        "candidate_layers",
        "protected_first_layers",
        "protected_last_layers",
        "pruned_layers",
        "selection_method",
    ):
        if key in row:
            extra[key] = deepcopy(row[key])
    if "pruned_layers_path" in row and "pruned_layers" not in extra:
        data = read_json(row["pruned_layers_path"])
        extra["pruned_layers"] = data.get("pruned_layers", data)
    return method(
        name,
        "shortgpt",
        uniform_ratios(prune_ratio),
        prompt,
        bias=False,
        prune_ratio=prune_ratio,
        density=1.0 - prune_ratio,
        calibration_path=_shortgpt_calibration_path(row),
        calibration_samples=int(row.get("calibration_samples", 128)),
        calibration_max_input_tokens=int(row.get("calibration_max_input_tokens", 2048)),
        baseline_type="shortgpt_depth_pruning",
        pruning_granularity="decoder_layer_logical_skip",
        matched_rasp_reference=str(row.get("matched_rasp_reference", "")),
        variant_role=str(row.get("variant_role", name)),
        selection_note=str(row.get("selection_note", "")),
        **extra,
        **(
            {"target_pruning_ratio": float(row["target_pruning_ratio"])}
            if "target_pruning_ratio" in row
            else {}
        ),
        **(
            {"target_pruning_label": str(row["target_pruning_label"])}
            if "target_pruning_label" in row
            else {}
        ),
    )


def shortgpt_methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    scfg = _shortgpt_cfg(cfg)
    variants = scfg.get("variants")
    if variants is None:
        if not scfg:
            return []
        variants = [scfg]
    if not isinstance(variants, list) or not variants:
        raise ValueError("shortgpt.variants must be a non-empty list when provided")
    methods = []
    for index, row in enumerate(variants):
        if not isinstance(row, dict):
            raise ValueError(f"shortgpt.variants[{index}] must be a mapping")
        merged = deepcopy(scfg)
        merged.pop("variants", None)
        merged.update(deepcopy(row))
        methods.append(_shortgpt_method_from_cfg(merged, structured_prompt(cfg)))
    names = [row["name"] for row in methods]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"ShortGPT method names must be unique: {duplicates}")
    return methods


def _limits_layer_pruning_method_from_cfg(row: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    prune_ratio = float(row.get("prune_ratio", row.get("ratio", 0.0)))
    name = str(row.get("method_name", row.get("name", f"limits_reverse_{prune_ratio:.4f}".replace(".", "p"))))
    extra: dict[str, Any] = {}
    for key in (
        "candidate_layers",
        "protected_first_layers",
        "protected_last_layers",
        "pruned_layers",
        "selection_method",
    ):
        if key in row:
            extra[key] = deepcopy(row[key])
    return method(
        name,
        "limits_layer_pruning",
        uniform_ratios(prune_ratio),
        prompt,
        bias=False,
        prune_ratio=prune_ratio,
        density=1.0 - prune_ratio,
        calibration_path=_shortgpt_calibration_path(row),
        calibration_samples=int(row.get("calibration_samples", 0)),
        calibration_max_input_tokens=int(row.get("calibration_max_input_tokens", 2048)),
        baseline_type=str(row.get("baseline_type", "limits_reverse_depth_pruning")),
        pruning_granularity="decoder_layer_logical_skip",
        matched_rasp_reference=str(row.get("matched_rasp_reference", "")),
        variant_role=str(row.get("variant_role", name)),
        selection_note=str(row.get("selection_note", "")),
        **extra,
        **(
            {"target_pruning_ratio": float(row["target_pruning_ratio"])}
            if "target_pruning_ratio" in row
            else {}
        ),
        **(
            {"target_pruning_label": str(row["target_pruning_label"])}
            if "target_pruning_label" in row
            else {}
        ),
    )


def limits_layer_pruning_methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    lcfg = _limits_layer_pruning_cfg(cfg)
    variants = lcfg.get("variants")
    if variants is None:
        if not lcfg:
            return []
        variants = [lcfg]
    if not isinstance(variants, list) or not variants:
        raise ValueError("limits_layer_pruning.variants must be a non-empty list when provided")
    methods = []
    for index, row in enumerate(variants):
        if not isinstance(row, dict):
            raise ValueError(f"limits_layer_pruning.variants[{index}] must be a mapping")
        merged = deepcopy(lcfg)
        merged.pop("variants", None)
        merged.setdefault("selection_method", "reverse")
        merged.update(deepcopy(row))
        methods.append(_limits_layer_pruning_method_from_cfg(merged, structured_prompt(cfg)))
    names = [row["name"] for row in methods]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Limits layer pruning method names must be unique: {duplicates}")
    return methods


def _flap_mlp_calibration_path(row: dict[str, Any]) -> str:
    if row.get("calibration_path"):
        return str(row["calibration_path"])
    source_root = os.environ.get(
        "SOURCE_ROOT",
        "runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3",
    )
    return str(Path(source_root) / "03_selected" / "calibration.jsonl")


def _flap_mlp_official_method_from_cfg(row: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    prune_ratio = float(row.get("prune_ratio", row.get("ratio", 0.0)))
    name = str(row.get("method_name", row.get("name", f"flap_mlp_official_{prune_ratio:.4f}".replace(".", "p"))))
    extra: dict[str, Any] = {}
    if "layers" in row:
        extra["layers"] = deepcopy(row["layers"])
    return method(
        name,
        "flap_mlp_official",
        uniform_ratios(prune_ratio),
        prompt,
        bias=bool(row.get("bias_compensation", False)),
        prune_ratio=prune_ratio,
        density=1.0 - prune_ratio,
        calibration_path=_flap_mlp_calibration_path(row),
        calibration_dataset=str(row.get("calibration_dataset", "mixed_calibration")),
        calibration_samples=int(row.get("calibration_samples", 128)),
        calibration_max_input_tokens=int(row.get("calibration_max_input_tokens", 2048)),
        metric=str(row.get("metric", "WIFV")),
        structure=str(row.get("structure", "AL-AM")),
        baseline_type="official_style_flap_mlp_qwen3_port",
        pruning_granularity="mlp_channel_structured",
        matched_rasp_reference=str(row.get("matched_rasp_reference", "")),
        variant_role=str(row.get("variant_role", name)),
        selection_note=str(row.get("selection_note", "")),
        **extra,
        **(
            {"target_pruning_ratio": float(row["target_pruning_ratio"])}
            if "target_pruning_ratio" in row
            else {}
        ),
        **(
            {"target_pruning_label": str(row["target_pruning_label"])}
            if "target_pruning_label" in row
            else {}
        ),
    )


def flap_mlp_official_methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    fcfg = _flap_mlp_official_cfg(cfg)
    variants = fcfg.get("variants")
    if variants is None:
        if not fcfg:
            return []
        variants = [fcfg]
    if not isinstance(variants, list) or not variants:
        raise ValueError("flap_mlp_official.variants must be a non-empty list when provided")
    methods = []
    for index, row in enumerate(variants):
        if not isinstance(row, dict):
            raise ValueError(f"flap_mlp_official.variants[{index}] must be a mapping")
        merged = deepcopy(fcfg)
        merged.pop("variants", None)
        merged.update(deepcopy(row))
        methods.append(_flap_mlp_official_method_from_cfg(merged, structured_prompt(cfg)))
    names = [row["name"] for row in methods]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"FLAP MLP official method names must be unique: {duplicates}")
    return methods


def _llm_pruner_mlp_method_from_cfg(row: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    prune_ratio = float(row.get("prune_ratio", row.get("ratio", 0.0)))
    name = str(row.get("method_name", row.get("name", f"llm_pruner_mlp_{prune_ratio:.4f}".replace(".", "p"))))
    extra: dict[str, Any] = {}
    if "layers" in row:
        extra["layers"] = deepcopy(row["layers"])
    return method(
        name,
        "llm_pruner_mlp_static_width",
        uniform_ratios(prune_ratio),
        prompt,
        bias=False,
        prune_ratio=prune_ratio,
        density=1.0 - prune_ratio,
        importance=str(row.get("importance", "l2")),
        structure=str(row.get("structure", "UL-UM")),
        physical_pruning=bool(row.get("physical_pruning", True)),
        baseline_type="llm_pruner_style_static_width_qwen3_mlp_no_recovery",
        pruning_granularity="mlp_channel_structured",
        matched_rasp_reference=str(row.get("matched_rasp_reference", "")),
        variant_role=str(row.get("variant_role", name)),
        selection_note=str(row.get("selection_note", "")),
        **extra,
        **(
            {"target_pruning_ratio": float(row["target_pruning_ratio"])}
            if "target_pruning_ratio" in row
            else {}
        ),
        **(
            {"target_pruning_label": str(row["target_pruning_label"])}
            if "target_pruning_label" in row
            else {}
        ),
    )


def llm_pruner_mlp_methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    pcfg = _llm_pruner_mlp_cfg(cfg)
    variants = pcfg.get("variants")
    if variants is None:
        if not pcfg:
            return []
        variants = [pcfg]
    if not isinstance(variants, list) or not variants:
        raise ValueError("llm_pruner_mlp_static_width.variants must be a non-empty list when provided")
    methods = []
    for index, row in enumerate(variants):
        if not isinstance(row, dict):
            raise ValueError(f"llm_pruner_mlp_static_width.variants[{index}] must be a mapping")
        merged = deepcopy(pcfg)
        merged.pop("variants", None)
        merged.update(deepcopy(row))
        methods.append(_llm_pruner_mlp_method_from_cfg(merged, structured_prompt(cfg)))
    names = [row["name"] for row in methods]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"LLM-Pruner MLP method names must be unique: {duplicates}")
    return methods


def _adaptive_stage_float_map_from_cfg(
    acfg: dict[str, Any],
    name: str,
    default: float = 0.0,
) -> dict[str, float]:
    return {stage: float(acfg.get(name, {}).get(stage, default)) for stage in STAGES}


def _adaptive_stage_int_map_from_cfg(
    acfg: dict[str, Any],
    name: str,
    default: int = 0,
) -> dict[str, int]:
    return {stage: int(acfg.get(name, {}).get(stage, default)) for stage in STAGES}


def _adaptive_stage_ratios(cfg: dict[str, Any]) -> dict[str, float]:
    acfg = _adaptive_griffin_cfg(cfg)
    return _adaptive_stage_float_map_from_cfg(acfg, "stage_ratios")


def _adaptive_warmup_tokens(cfg: dict[str, Any]) -> dict[str, int]:
    acfg = _adaptive_griffin_cfg(cfg)
    return _adaptive_stage_int_map_from_cfg(acfg, "warmup_tokens")


def _adaptive_stage_float_map(cfg: dict[str, Any], name: str, default: float = 0.0) -> dict[str, float]:
    acfg = _adaptive_griffin_cfg(cfg)
    return _adaptive_stage_float_map_from_cfg(acfg, name, default)


def _merge_adaptive_variant(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    merged.pop("variants", None)
    for key, value in variant.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(deepcopy(value))
            merged[key] = nested
        else:
            merged[key] = deepcopy(value)
    return merged


def _adaptive_griffin_method_from_cfg(acfg: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    return method(
        str(acfg.get("method_name", "calibrated_stage_adaptive_griffin_main")),
        str(acfg.get("policy", "calibrated_stage_adaptive_griffin")),
        _adaptive_stage_float_map_from_cfg(acfg, "stage_ratios"),
        prompt,
        bias=bool(acfg.get("bias_compensation", True)),
        alpha=float(acfg.get("alpha", 0.7)),
        runtime_weight=float(acfg.get("runtime_weight", 0.4)),
        prior_weight=float(acfg.get("prior_weight", 0.6)),
        warmup_tokens=_adaptive_stage_int_map_from_cfg(acfg, "warmup_tokens"),
        protected_core_ratios=_adaptive_stage_float_map_from_cfg(acfg, "protected_core_ratios"),
        refresh_intervals=_adaptive_stage_int_map_from_cfg(acfg, "refresh_intervals"),
        window_tokens=_adaptive_stage_int_map_from_cfg(acfg, "window_tokens", 1),
        prior_policy=str(acfg.get("prior_policy", "stage_specific")),
        base_policy=str(acfg.get("base_policy", "trajectory_global")),
        stage_prior_policy=str(acfg.get("stage_prior_policy", "stage_specific")),
        score_mode=str(acfg.get("score_mode", "activation")),
        max_mask_swap_fraction=float(acfg.get("max_mask_swap_fraction", 1.0)),
        adaptive_backend=str(acfg.get("backend", "logical_mask")),
        static_core_ratios=_adaptive_stage_float_map_from_cfg(acfg, "static_core_ratios", 1.0),
        swap_ratios=_adaptive_stage_float_map_from_cfg(acfg, "swap_ratios", 0.0),
        variant_role=str(acfg.get("variant_role", acfg.get("method_name", "adaptive"))),
        selection_note=str(acfg.get("selection_note", "")),
        fallback_behavior=str(acfg.get("fallback_behavior", "dense_after_error")),
        stage_risk_controller=deepcopy(acfg.get("stage_risk_controller", {})),
        stage_budget_controller=deepcopy(acfg.get("stage_budget_controller", {})),
        attention_head_pruning=deepcopy(acfg.get("attention_head_pruning", {})),
        multi_structure_budget_controller=deepcopy(
            acfg.get("multi_structure_budget_controller", {})
        ),
        **_pruning_target_fields(acfg),
    )


def adaptive_griffin_method(cfg: dict[str, Any]) -> dict[str, Any]:
    return adaptive_griffin_methods(cfg)[0]


def adaptive_griffin_methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    acfg = _adaptive_griffin_cfg(cfg)
    variants = acfg.get("variants")
    if variants is None:
        return [_adaptive_griffin_method_from_cfg(acfg, structured_prompt(cfg))]
    if not isinstance(variants, list) or not variants:
        raise ValueError("adaptive_griffin.variants must be a non-empty list when provided")
    methods = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"adaptive_griffin.variants[{index}] must be a mapping")
        methods.append(
            _adaptive_griffin_method_from_cfg(
                _merge_adaptive_variant(acfg, variant),
                structured_prompt(cfg),
            )
        )
    names = [row["name"] for row in methods]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Adaptive GRIFFIN method names must be unique: {duplicates}")
    return methods


def _nearest_ratio(value: float, ratios: list[float]) -> float:
    return min((float(ratio) for ratio in ratios), key=lambda ratio: (abs(ratio - value), -ratio))


def static_matched_global_method(cfg: dict[str, Any], ratios: list[float]) -> dict[str, Any]:
    acfg = _adaptive_griffin_cfg(cfg)
    configured = acfg.get("static_matched_ratio")
    if configured is None:
        configured = sum(adaptive_griffin_method(cfg)["stage_ratios"].values()) / len(STAGES)
    static_ratio = float(configured)
    matched_to = acfg.get("static_matched_to")
    if matched_to is None:
        matched_to = ",".join(row["name"] for row in adaptive_griffin_methods(cfg))
    return method(
        "static_matched_global",
        "trajectory_global",
        uniform_ratios(static_ratio),
        structured_prompt(cfg),
        bias=True,
        matched_to=str(matched_to),
        requested_matched_ratio=float(configured),
    )


def additional_static_matched_global_methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    acfg = _adaptive_griffin_cfg(cfg)
    configured = acfg.get("additional_static_matched_globals", [])
    if not configured:
        return []
    methods = []
    for index, row in enumerate(configured):
        if not isinstance(row, dict):
            raise ValueError(f"additional_static_matched_globals[{index}] must be a mapping")
        ratio = float(row["ratio"])
        name = str(row.get("name") or f"static_matched_global_{ratio:.4f}".replace(".", "p"))
        methods.append(
            method(
                name,
                str(row.get("policy", "trajectory_global")),
                uniform_ratios(ratio),
                structured_prompt(cfg),
                bias=bool(row.get("bias_compensation", True)),
                matched_to=str(row.get("matched_to", "")),
                requested_matched_ratio=ratio,
                static_matched_role=str(row.get("role", name)),
            )
        )
    return methods


def additional_fixed_stage_methods(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    acfg = _adaptive_griffin_cfg(cfg)
    configured = acfg.get("additional_fixed_stage_methods", [])
    if not configured:
        return []
    methods = []
    for index, row in enumerate(configured):
        if not isinstance(row, dict):
            raise ValueError(f"additional_fixed_stage_methods[{index}] must be a mapping")
        name = str(row["name"])
        policy = str(row["policy"])
        ratios = {
            stage: float(row.get("stage_ratios", {}).get(stage, 0.0))
            for stage in STAGES
        }
        methods.append(
            method(
                name,
                policy,
                ratios,
                structured_prompt(cfg),
                bias=bool(row.get("bias_compensation", True)),
                ablation_role=str(row.get("ablation_role", "")),
                selection_note=str(row.get("selection_note", "")),
                fallback_behavior=str(row.get("fallback_behavior", "dense_after_error")),
                **_pruning_target_fields(row),
            )
        )
    names = [row["name"] for row in methods]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Fixed stage ablation method names must be unique: {duplicates}")
    return methods


def _evaluation_threshold(cfg: dict[str, Any], name: str, default: float | int | None = None) -> Any:
    pcfg = profile(cfg)
    if name in pcfg:
        return pcfg[name]
    return cfg.get("evaluation", {}).get(name, default)


def _profile_list(cfg: dict[str, Any], name: str, default: list[Any]) -> list[Any]:
    pcfg = profile(cfg)
    if name in pcfg:
        return list(pcfg[name])
    if name in cfg.get("evaluation", {}):
        return list(cfg["evaluation"][name])
    return list(default)


def _profile_float_list(cfg: dict[str, Any], name: str, default: list[float]) -> list[float]:
    return [float(value) for value in _profile_list(cfg, name, list(default))]


def _metric_value(summary: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = summary.get(name)
    return default if value is None else float(value)


def _prompt_accuracy_gate_passed(
    cfg: dict[str, Any],
    *,
    ordinary: dict[str, Any],
    dense: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    ordinary_accuracy = _metric_value(ordinary, "accuracy")
    dense_accuracy = _metric_value(dense, "accuracy")
    max_drop = float(_evaluation_threshold(cfg, "maximum_structured_prompt_accuracy_drop", 0.02))
    percentage_passed = dense_accuracy >= ordinary_accuracy - max_drop
    extra_error_limit = _evaluation_threshold(cfg, "maximum_structured_prompt_extra_errors", None)
    extra_error_passed = False
    extra_errors = None
    if extra_error_limit is not None:
        ordinary_errors = int(ordinary["problems"]) - int(ordinary["correct"])
        dense_errors = int(dense["problems"]) - int(dense["correct"])
        extra_errors = dense_errors - ordinary_errors
        extra_error_passed = extra_errors <= int(extra_error_limit)
    return percentage_passed or extra_error_passed, {
        "maximum_structured_prompt_accuracy_drop": max_drop,
        "percentage_drop_passed": percentage_passed,
        "maximum_structured_prompt_extra_errors": extra_error_limit,
        "structured_prompt_extra_errors": extra_errors,
        "extra_error_passed": extra_error_passed,
    }


def _method_quality_passed(cfg: dict[str, Any], summary: dict[str, Any]) -> bool:
    min_protocol = float(
        _evaluation_threshold(
            cfg,
            "minimum_candidate_stage_protocol_rate",
            _evaluation_threshold(cfg, "minimum_stage_protocol_rate", 0.0),
        )
    )
    max_fallback = float(_evaluation_threshold(cfg, "maximum_candidate_fallback_rate", 1.0))
    max_truncation = float(
        _evaluation_threshold(
            cfg,
            "maximum_candidate_truncation_rate",
            _evaluation_threshold(cfg, "maximum_truncation_rate", 1.0),
        )
    )
    return (
        _metric_value(summary, "valid_stage_protocol_rate") >= min_protocol
        and _metric_value(summary, "fallback_rate") <= max_fallback
        and _metric_value(summary, "truncation_rate") <= max_truncation
    )


def _build_calibration_gate(
    cfg: dict[str, Any],
    calibration_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    default_required = bool(calibration_comparisons)
    reference_gate_required = bool(
        _evaluation_threshold(cfg, "require_calibration_reference_gate", default_required)
    )
    trajectory_promising = any(row["trajectory_strictly_best"] for row in calibration_comparisons)
    return {
        "comparisons": calibration_comparisons,
        "required": reference_gate_required,
        "skipped_reason": (
            None
            if reference_gate_required
            else "no calibration reference comparison configured; ratio selection is handled by policy_selection"
        ),
        "trajectory_calibration_promising": trajectory_promising if reference_gate_required else True,
    }


def _run_methods(cfg, p, tasks, bank, bundle, methods, output_dir, seed: int | None = None) -> list[dict[str, Any]]:
    summaries = []
    ensure_dir(output_dir)
    seed = int(cfg["seed"]) if seed is None else int(seed)
    if any(row.get("policy") == "wanda_official" for row in methods) and len(methods) != 1:
        raise RuntimeError(
            "Wanda official zeroes model weights in-place; run exactly one Wanda method per "
            "process via STAGE_FINAL_METHODS to avoid stacked pruning."
        )
    if any(row.get("policy") == "sparsegpt_official" for row in methods) and len(methods) != 1:
        raise RuntimeError(
            "SparseGPT official loads pruned weights in-place; run exactly one SparseGPT "
            "method per process via STAGE_FINAL_METHODS to avoid stacked pruning."
        )
    if any(row.get("policy") == "flap_mlp_official" for row in methods) and len(methods) != 1:
        raise RuntimeError(
            "FLAP-MLP official physically prunes MLP channels in-place; run exactly one FLAP "
            "method per process via STAGE_FINAL_METHODS to avoid stacked pruning."
        )
    if any(row.get("policy") == "llm_pruner_mlp_static_width" for row in methods) and len(methods) != 1:
        raise RuntimeError(
            "LLM-Pruner-style MLP width pruning mutates MLP channels in-place; run exactly one "
            "LLM-Pruner MLP method per process via STAGE_FINAL_METHODS to avoid stacked pruning."
        )
    for value in methods:
        suffix = f"_seed{seed}"
        rows_path = output_dir / f"{value['name']}{suffix}.jsonl"
        summary_path = output_dir / f"{value['name']}{suffix}.summary.json"
        if bool(_evaluation_threshold(cfg, "reuse_existing_method_summaries", True)) and summary_path.exists():
            summaries.append(read_json(summary_path))
            continue
        rows, summary = evaluate_method(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            tasks=tasks,
            bank=bank,
            method=value,
            generation=cfg["generation"],
            seed=seed,
        )
        write_jsonl(rows_path, rows)
        write_json(summary_path, summary)
        summaries.append(summary)
    return summaries


def command_evaluate_dev(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    tasks = read_jsonl(p["dev"])
    bank = load_mask_bank(
        p["bank"],
        expected_bank_metadata(cfg, p),
        ignored_metadata_keys=("config_hash",) if _adaptive_griffin_enabled(cfg) else (),
    )
    bundle = load_model_bundle(cfg["model"])
    ratios = [float(value) for value in cfg["masks"]["ratios"]]
    uniform_ratio_grid = _profile_float_list(cfg, "dev_uniform_ratios", ratios)
    budget_search_ratios = _profile_float_list(cfg, "budget_search_ratios", ratios)
    coordinate_rounds = int(_evaluation_threshold(cfg, "coordinate_rounds", 2))
    run_no_bias_ablation = bool(_evaluation_threshold(cfg, "run_no_bias_compensation_ablation", True))
    c4_enabled = _effective_c4_samples(cfg) > 0
    default_global_policies = ["prompt_only_global", "trajectory_global", "stage_balanced_global"]
    if c4_enabled:
        default_global_policies.insert(0, "c4_global")
    global_policies = [
        str(policy)
        for policy in _profile_list(cfg, "dev_global_policies", default_global_policies)
        if c4_enabled or str(policy) != "c4_global"
    ]
    run_stage_specific_uniform = bool(
        _evaluation_threshold(cfg, "run_stage_specific_uniform", True)
    )
    run_shuffled_uniform = bool(_evaluation_threshold(cfg, "run_shuffled_uniform", True))
    methods = [
        method("ordinary_dense", "trajectory_global", uniform_ratios(0.0), ordinary_prompt(cfg)),
        method("structured_dense", "trajectory_global", uniform_ratios(0.0), structured_prompt(cfg)),
    ]
    if _adaptive_griffin_enabled(cfg):
        methods.extend(adaptive_griffin_methods(cfg))
        methods.append(static_matched_global_method(cfg, ratios))
        methods.extend(additional_static_matched_global_methods(cfg))
        methods.extend(additional_fixed_stage_methods(cfg))
    for ratio in uniform_ratio_grid:
        if ratio <= 0:
            continue
        tag = f"{ratio:.2f}".replace(".", "p")
        for policy in global_policies:
            methods.append(method(f"{policy}_{tag}", policy, uniform_ratios(ratio), structured_prompt(cfg)))
        if run_stage_specific_uniform:
            methods.append(method(f"stage_specific_{tag}", "stage_specific", uniform_ratios(ratio), structured_prompt(cfg)))
        if run_shuffled_uniform:
            methods.append(method(f"shuffled_stage_{tag}", "shuffled_stage", uniform_ratios(ratio), structured_prompt(cfg)))
        if run_no_bias_ablation and abs(ratio - 0.10) < 1e-12:
            methods.append(
                method(
                    "stage_specific_0p10_no_bias_compensation",
                    "stage_specific",
                    uniform_ratios(ratio),
                    structured_prompt(cfg),
                    bias=False,
                )
            )
    if cfg["workflow"]["profile"] == "formal":
        methods.extend(
            [
                method(
                    "trajectory_global_al_am_0p10",
                    "trajectory_global_al_am",
                    uniform_ratios(0.10),
                    structured_prompt(cfg),
                ),
                method(
                    "stage_specific_al_am_0p10",
                    "stage_specific_al_am",
                    uniform_ratios(0.10),
                    structured_prompt(cfg),
                ),
            ]
        )
    summaries = _run_methods(cfg, p, tasks, bank, bundle, methods, p["dev_dir"] / "uniform")
    dense = next(row for row in summaries if row["method"]["name"] == "structured_dense")
    ordinary = next(row for row in summaries if row["method"]["name"] == "ordinary_dense")
    accuracy_floor = float(dense["accuracy"]) - float(cfg["evaluation"]["max_dev_accuracy_drop"])
    prompt_accuracy_passed, prompt_accuracy_details = _prompt_accuracy_gate_passed(
        cfg,
        ordinary=ordinary,
        dense=dense,
    )
    minimum_stage_protocol_rate = float(
        _evaluation_threshold(cfg, "minimum_stage_protocol_rate", 0.90)
    )
    maximum_truncation_rate = float(
        _evaluation_threshold(cfg, "maximum_truncation_rate", 0.05)
    )
    prompt_gate = {
        "structured_protocol_rate": dense["valid_stage_protocol_rate"],
        "structured_truncation_rate": dense["truncation_rate"],
        "ordinary_dense_accuracy": ordinary["accuracy"],
        "structured_dense_accuracy": dense["accuracy"],
        "minimum_stage_protocol_rate": minimum_stage_protocol_rate,
        "maximum_truncation_rate": maximum_truncation_rate,
        **prompt_accuracy_details,
        "passed": bool(
            float(dense["valid_stage_protocol_rate"])
            >= minimum_stage_protocol_rate
            and float(dense["truncation_rate"])
            <= maximum_truncation_rate
            and prompt_accuracy_passed
        ),
    }
    calibration_comparisons = []
    for ratio in uniform_ratio_grid:
        if ratio <= 0:
            continue
        matching = [
            row
            for row in summaries
            if list(row["method"]["stage_ratios"].values()) == [ratio] * len(STAGES)
        ]
        by_policy = {row["method"]["policy"]: row for row in matching}
        reference_policies = [
            policy for policy in ("c4_global", "prompt_only_global") if policy in by_policy
        ]
        if "trajectory_global" in by_policy and reference_policies:
            trajectory_quality_passed = _method_quality_passed(cfg, by_policy["trajectory_global"])
            reference_accuracies = [
                float(by_policy[policy]["accuracy"]) for policy in reference_policies
            ]
            calibration_comparisons.append(
                {
                    "ratio": ratio,
                    "c4_evaluated": c4_enabled,
                    "reference_policies": reference_policies,
                    "trajectory_accuracy": by_policy["trajectory_global"]["accuracy"],
                    "c4_accuracy": (
                        by_policy["c4_global"]["accuracy"] if "c4_global" in by_policy else None
                    ),
                    "prompt_only_accuracy": by_policy["prompt_only_global"]["accuracy"],
                    "trajectory_valid_stage_protocol_rate": by_policy["trajectory_global"][
                        "valid_stage_protocol_rate"
                    ],
                    "trajectory_fallback_rate": by_policy["trajectory_global"]["fallback_rate"],
                    "trajectory_quality_passed": trajectory_quality_passed,
                    "trajectory_strictly_best": (
                        trajectory_quality_passed
                        and
                        float(by_policy["trajectory_global"]["accuracy"]) > max(reference_accuracies)
                    ),
                }
            )
    calibration_gate = _build_calibration_gate(cfg, calibration_comparisons)
    if _adaptive_griffin_enabled(cfg):
        adaptive_method_cfgs = adaptive_griffin_methods(cfg)
        adaptive_names = [row["name"] for row in adaptive_method_cfgs]
        adaptive_summaries = [
            next(row for row in summaries if row["method"]["name"] == name)
            for name in adaptive_names
        ]
        static_summary = next(row for row in summaries if row["method"]["name"] == "static_matched_global")
        method_gates = []
        for row in adaptive_summaries:
            final_masked_tokens = int(row.get("masked_tokens_by_stage", {}).get("final", 0))
            beats_static = float(row["accuracy"]) >= float(static_summary["accuracy"])
            quality_passed = _method_quality_passed(cfg, row)
            method_gates.append(
                {
                    "method_name": row["method"]["name"],
                    "passed": bool(
                        prompt_gate["passed"]
                        and quality_passed
                        and beats_static
                        and final_masked_tokens == 0
                    ),
                    "prompt_gate_passed": bool(prompt_gate["passed"]),
                    "adaptive_quality_passed": quality_passed,
                    "adaptive_beats_static_on_dev": beats_static,
                    "adaptive_final_masked_tokens": final_masked_tokens,
                    "accuracy": row["accuracy"],
                    "static_matched_accuracy": static_summary["accuracy"],
                    "theoretical_average_mlp_pruning_ratio": row[
                        "theoretical_average_mlp_pruning_ratio"
                    ],
                    "valid_stage_protocol_rate": row["valid_stage_protocol_rate"],
                    "fallback_rate": row["fallback_rate"],
                    "truncation_rate": row["truncation_rate"],
                }
            )
        gate_by_name = {row["method_name"]: row for row in method_gates}
        primary_adaptive_summary = max(
            adaptive_summaries,
            key=lambda row: (
                int(gate_by_name[row["method"]["name"]]["passed"]),
                int(gate_by_name[row["method"]["name"]]["adaptive_beats_static_on_dev"]),
                float(row["accuracy"]),
                float(row["theoretical_average_mlp_pruning_ratio"]),
            ),
        )
        primary_adaptive_name = primary_adaptive_summary["method"]["name"]
        adaptive_griffin_gate = {
            "passed": any(row["passed"] for row in method_gates),
            "diagnostic_sweep": len(adaptive_summaries) > 1,
            "primary_method_name": primary_adaptive_name,
            "method_gates": method_gates,
            "prompt_gate_passed": bool(prompt_gate["passed"]),
            "minimum_stage_protocol_rate": float(
                _evaluation_threshold(cfg, "minimum_candidate_stage_protocol_rate", 0.90)
            ),
            "maximum_candidate_fallback_rate": float(
                _evaluation_threshold(cfg, "maximum_candidate_fallback_rate", 0.10)
            ),
            "maximum_candidate_truncation_rate": float(
                _evaluation_threshold(cfg, "maximum_candidate_truncation_rate", 0.05)
            ),
        }
        frozen = {
            "schema": (
                "stage_calibrated_adaptive_griffin_sweep_policy_v1"
                if len(adaptive_summaries) > 1
                else "stage_calibrated_adaptive_griffin_policy_v1"
            ),
            "method": primary_adaptive_summary["method"],
            "adaptive_methods": [row["method"] for row in adaptive_summaries],
            "static_matched_global": static_summary["method"],
            "structured_dense_accuracy": dense["accuracy"],
            "adaptive_accuracy": primary_adaptive_summary["accuracy"],
            "adaptive_dev_summaries": {
                row["method"]["name"]: {
                    "accuracy": row["accuracy"],
                    "theoretical_average_mlp_pruning_ratio": row[
                        "theoretical_average_mlp_pruning_ratio"
                    ],
                    "valid_stage_protocol_rate": row["valid_stage_protocol_rate"],
                    "fallback_rate": row["fallback_rate"],
                    "truncation_rate": row["truncation_rate"],
                    "masked_tokens_by_stage": row.get("masked_tokens_by_stage", {}),
                    "mask_refresh_count_by_stage": row.get("mask_refresh_count_by_stage", {}),
                }
                for row in adaptive_summaries
            },
            "static_matched_accuracy": static_summary["accuracy"],
            "adaptive_beats_static_on_dev": gate_by_name[primary_adaptive_name][
                "adaptive_beats_static_on_dev"
            ],
            "adaptive_griffin_gate": adaptive_griffin_gate,
            "test_sets_consulted": False,
            "prompt_gate_passed": bool(prompt_gate["passed"]),
            "final_evaluation_forbidden": False,
        }
        write_json(p["frozen"], frozen)
        structured_method = method(
            "structured_dense",
            "trajectory_global",
            uniform_ratios(0.0),
            structured_prompt(cfg),
        )
        selected_policies = {
            "structured_dense": {
                "role": "structured_dense",
                "method": structured_method,
                "reason": "Explicit-stage dense reference.",
            },
            "static_matched_global": {
                "role": "static_matched_global",
                "method": static_summary["method"],
                "reason": "Trajectory-global WIFV static baseline matched to the configured adaptive budget.",
                "dev_summary": {
                    "accuracy": static_summary["accuracy"],
                    "theoretical_average_mlp_pruning_ratio": static_summary[
                        "theoretical_average_mlp_pruning_ratio"
                    ],
                },
            },
        }
        for row in adaptive_summaries:
            name = row["method"]["name"]
            selected_policies[name] = {
                "role": name,
                "method": row["method"],
                "reason": "Stage-conditioned WIFV prior with per-sample dynamic channel selection.",
                "dev_summary": {
                    "accuracy": row["accuracy"],
                    "theoretical_average_mlp_pruning_ratio": row[
                        "theoretical_average_mlp_pruning_ratio"
                    ],
                    "valid_stage_protocol_rate": row["valid_stage_protocol_rate"],
                    "fallback_rate": row["fallback_rate"],
                    "truncation_rate": row["truncation_rate"],
                },
                "gate": gate_by_name[name],
            }
        downstream_methods = (
            [structured_method]
            + [row["method"] for row in adaptive_summaries]
            + [static_summary["method"]]
        )
        policy_selection = {
            "schema": "stage_policy_selection_v1",
            "selection_mode": (
                "adaptive_griffin_sweep"
                if len(adaptive_summaries) > 1
                else "adaptive_griffin_main_only"
            ),
            "input_manifest_hash": manifest_hash(tasks),
            "test_sets_consulted": False,
            "selection_policy": {
                "dense_reference": "structured_dense",
                "adaptive_policy": "adaptive_griffin_sweep",
                "static_baseline": "trajectory_global at configured static_matched_ratio",
                "downstream_results_must_not_change_selection": True,
                "primary_method_name": primary_adaptive_name,
            },
            "selected_policies": selected_policies,
            "downstream_methods": downstream_methods,
        }
        policy_selection["policy_selection_hash"] = stable_hash(policy_selection)
        policy_selection_path = p["dev_dir"] / "adaptive_griffin_policy_selection.json"
        write_json(policy_selection_path, policy_selection)
        method_table_rows = [
            f"| structured_dense | `structured_dense` | {float(dense['accuracy']):.4f} | "
            "0.0000 | 1.0000 | 0.0000 | 0.0000 | baseline |"
        ]
        for row in adaptive_summaries:
            gate = gate_by_name[row["method"]["name"]]
            method_table_rows.append(
                f"| {row['method']['name']} | `{row['method']['name']}` | "
                f"{float(row['accuracy']):.4f} | "
                f"{float(row['theoretical_average_mlp_pruning_ratio']):.4f} | "
                f"{float(row['valid_stage_protocol_rate']):.4f} | "
                f"{float(row['fallback_rate']):.4f} | "
                f"{float(row['truncation_rate']):.4f} | "
                f"{'pass' if gate['passed'] else 'diagnostic'} |"
            )
        method_table_rows.append(
            "| static_matched_global | "
            f"`{static_summary['method']['name']}` | "
            f"{float(static_summary['accuracy']):.4f} | "
            f"{float(static_summary['theoretical_average_mlp_pruning_ratio']):.4f} | "
            f"{float(static_summary['valid_stage_protocol_rate']):.4f} | "
            f"{float(static_summary['fallback_rate']):.4f} | "
            f"{float(static_summary['truncation_rate']):.4f} | baseline |"
        )
        (p["dev_dir"] / "adaptive_griffin_policy_selection.md").write_text(
            "\n".join(
                [
                    "# Adaptive GRIFFIN Policy Selection",
                    "",
                    f"- schema: `{policy_selection['schema']}`",
                    f"- selection_mode: `{policy_selection['selection_mode']}`",
                    f"- input_manifest_hash: `{policy_selection['input_manifest_hash']}`",
                    f"- test_sets_consulted: `{policy_selection['test_sets_consulted']}`",
                    f"- primary_method_name: `{primary_adaptive_name}`",
                    "",
                    "| role | method | dev accuracy | pruning | valid protocol | fallback | truncation | gate |",
                    "|---|---|---:|---:|---:|---:|---:|---|",
                    *method_table_rows,
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        write_json(
            p["dev_summary"],
            {
                "schema": "stage_calibrated_dev_summary_v1",
                **metadata(cfg, dev_manifest_hash=manifest_hash(tasks)),
                "methods": summaries,
                "c4_evaluated": c4_enabled,
                "evaluated_uniform_policies": global_policies,
                "evaluated_uniform_ratios": uniform_ratio_grid,
                "adaptive_griffin_evaluated": True,
                "adaptive_griffin_policy_frozen": True,
                "adaptive_griffin_policy_selection_path": str(policy_selection_path),
                "adaptive_griffin_method_names": adaptive_names,
                "adaptive_griffin_sweep": len(adaptive_names) > 1,
                "adaptive_griffin_gate": adaptive_griffin_gate,
                "prompt_gate": prompt_gate,
                "calibration_gate": calibration_gate,
                "stage_budget_search_performed": False,
                "frozen_policy": frozen,
            },
        )
        return
    prompt_gate_passed = bool(prompt_gate["passed"])
    calibration_gate_passed = bool(calibration_gate["trajectory_calibration_promising"])
    gate_failure_reasons = []
    if not prompt_gate_passed:
        gate_failure_reasons.append("prompt_gate_failed")
    if not calibration_gate_passed:
        gate_failure_reasons.append("calibration_gate_failed")
    diagnostic_budget_search = bool(
        _evaluation_threshold(cfg, "diagnostic_budget_search_on_gate_failure", False)
    ) and calibration_gate_passed
    if gate_failure_reasons and not diagnostic_budget_search:
        if cfg["workflow"].get("profile") == "smoke":
            global_candidates = [
                row for row in summaries if row["method"]["policy"] == "trajectory_global"
            ]
            quality_global_candidates = [
                row for row in global_candidates if _method_quality_passed(cfg, row)
            ]
            if quality_global_candidates:
                global_candidates = quality_global_candidates
            best_global = max(
                global_candidates,
                key=lambda row: (
                    float(row["accuracy"]),
                    float(row["theoretical_average_mlp_pruning_ratio"]),
                ),
            )
            frozen = {
                "schema": "stage_calibrated_frozen_policy_v1",
                "smoke_relaxed_e2e": True,
                "relaxation_reason": "development gate failed in smoke profile",
                "stage_budget_is_pruned": False,
                "reason": "dev gate failed; ratios inherited only for e2e smoke",
                "gate_failure_reasons": gate_failure_reasons,
                "best_trajectory_global": best_global["method"],
                "stage_budget": method(
                    "stage_budget",
                    "stage_specific",
                    best_global["method"]["stage_ratios"],
                    structured_prompt(cfg),
                ),
                "test_sets_consulted": False,
            }
            write_json(p["frozen"], frozen)
            write_json(
                p["dev_summary"],
                {
                    "schema": "stage_calibrated_dev_summary_v1",
                    **metadata(cfg, dev_manifest_hash=manifest_hash(tasks)),
                    "methods": summaries,
                    "c4_evaluated": c4_enabled,
                    "evaluated_uniform_policies": global_policies,
                    "evaluated_uniform_ratios": uniform_ratio_grid,
                    "budget_search_ratios": budget_search_ratios,
                    "coordinate_rounds": coordinate_rounds,
                    "prompt_gate": prompt_gate,
                    "calibration_gate": calibration_gate,
                    "stage_budget_search_performed": False,
                    "smoke_relaxed_e2e": True,
                    "frozen_policy": frozen,
                },
            )
            return
        write_json(
            p["dev_summary"],
            {
                "schema": "stage_calibrated_dev_summary_v1",
                **metadata(cfg, dev_manifest_hash=manifest_hash(tasks)),
                "methods": summaries,
                "c4_evaluated": c4_enabled,
                "evaluated_uniform_policies": global_policies,
                "evaluated_uniform_ratios": uniform_ratio_grid,
                "budget_search_ratios": budget_search_ratios,
                "coordinate_rounds": coordinate_rounds,
                "prompt_gate": prompt_gate,
                "calibration_gate": calibration_gate,
                "stage_budget_search_performed": False,
                "frozen_policy": None,
            },
        )
        raise RuntimeError(
            "Development gate failed; do not search stage budgets or use final test sets"
        )
    current = uniform_ratios(0.0)
    coordinate_summaries = []
    for round_index in range(coordinate_rounds):
        for stage in STAGES:
            candidates = []
            for ratio in budget_search_ratios:
                candidate = dict(current)
                candidate[stage] = ratio
                name = f"coordinate_r{round_index}_{stage}_{ratio:.2f}".replace(".", "p")
                result = _run_methods(
                    cfg,
                    p,
                    tasks,
                    bank,
                    bundle,
                    [method(name, "stage_specific", candidate, structured_prompt(cfg))],
                    p["dev_dir"] / "coordinate",
                )[0]
                coordinate_summaries.append(result)
                if float(result["accuracy"]) >= accuracy_floor and _method_quality_passed(cfg, result):
                    candidates.append(result)
            if candidates:
                best = max(candidates, key=lambda row: (row["theoretical_average_mlp_pruning_ratio"], row["accuracy"]))
                current = dict(best["method"]["stage_ratios"])
    all_summaries = summaries + coordinate_summaries
    feasible_trajectory = [
        row
        for row in summaries
        if row["method"]["policy"] == "trajectory_global"
        and float(row["accuracy"]) >= accuracy_floor
        and _method_quality_passed(cfg, row)
    ]
    if not feasible_trajectory:
        raise RuntimeError("No trajectory_global method passed dev accuracy and protocol/fallback gates")
    best_global = max(
        feasible_trajectory,
        key=lambda row: (row["theoretical_average_mlp_pruning_ratio"], row["accuracy"]),
    )
    frozen = {
        "schema": "stage_calibrated_frozen_policy_v1",
        "accuracy_floor": accuracy_floor,
        "structured_dense_accuracy": dense["accuracy"],
        "stage_budget_is_pruned": not bool(gate_failure_reasons),
        "diagnostic_only": bool(gate_failure_reasons),
        "final_evaluation_forbidden": bool(gate_failure_reasons),
        "gate_failure_reasons": gate_failure_reasons,
        "diagnostic_reason": (
            "budget search continued for pilot diagnostics despite failed development gate"
            if gate_failure_reasons
            else None
        ),
        "best_trajectory_global": best_global["method"],
        "stage_budget": method("stage_budget", "stage_specific", current, structured_prompt(cfg)),
        "test_sets_consulted": False,
    }
    write_json(p["frozen"], frozen)
    write_json(
        p["dev_summary"],
        {
            "schema": "stage_calibrated_dev_summary_v1",
            **metadata(cfg, dev_manifest_hash=manifest_hash(tasks)),
            "methods": all_summaries,
            "c4_evaluated": c4_enabled,
            "evaluated_uniform_policies": global_policies,
            "evaluated_uniform_ratios": uniform_ratio_grid,
            "budget_search_ratios": budget_search_ratios,
            "coordinate_rounds": coordinate_rounds,
            "prompt_gate": prompt_gate,
            "calibration_gate": calibration_gate,
            "stage_budget_search_performed": True,
            "diagnostic_budget_search": bool(gate_failure_reasons),
            "gate_failure_reasons": gate_failure_reasons,
            "frozen_policy": frozen,
        },
    )


def command_prepare_sparsegpt(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    if not _sparsegpt_official_enabled(cfg):
        raise RuntimeError("sparsegpt_official.enabled must be true to run prepare_sparsegpt")
    methods = _limit_final_methods_for_smoke(cfg, sparsegpt_official_methods(cfg))
    if not methods:
        raise RuntimeError("No SparseGPT official methods selected for prepare_sparsegpt")
    ensure_dir(p["sparsegpt_dir"])
    summaries = []
    for row in methods:
        bundle = load_model_bundle(cfg["model"])
        summary = prepare_sparsegpt_official_qwen3_artifact(
            bundle.model,
            bundle.tokenizer,
            artifact_dir=str(row["artifact_path"]),
            calibration_path=str(row["calibration_path"]),
            prompt_config=dict(row.get("prompt", {})),
            sparsity_ratio=float(row["sparsity_ratio"]),
            calibration_samples=int(row.get("calibration_samples", 128)),
            calibration_max_input_tokens=int(row.get("calibration_max_input_tokens", 2048)),
            calibration_batch_size=int(row.get("calibration_batch_size", 1)),
            target_modules=row.get("target_modules"),
            blocksize=int(row.get("blocksize", 128)),
            percdamp=float(row.get("percdamp", 0.01)),
            artifact_save_dtype=str(row.get("artifact_save_dtype", "float16")),
            matched_rasp_reference=str(row.get("matched_rasp_reference", "")),
            target_matched_to_rasp_actual_mlp_pruning=(
                float(row["target_pruning_ratio"])
                if "target_pruning_ratio" in row
                else None
            ),
            force=bool(os.environ.get("SPARSEGPT_FORCE_PREPARE") == "1"),
        )
        summaries.append({"method": row, "artifact": sparsegpt_summary_to_dict(summary)})
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(
        p["sparsegpt_summary"],
        {
            "schema": "sparsegpt_official_prepare_summary_v1",
            **metadata(cfg),
            "methods": summaries,
            "calibration_reuses_test_labels": False,
            "real_speedup_claimed": False,
        },
    )


def command_evaluate_final(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    allow_final_without_dev = bool(_evaluation_threshold(cfg, "allow_final_without_dev", False))
    final_without_dev_allowed = allow_final_without_dev and (
        _adaptive_griffin_enabled(cfg)
        or _griffin_prompt_enabled(cfg)
        or _wanda_official_enabled(cfg)
        or _sparsegpt_official_enabled(cfg)
        or _shortgpt_enabled(cfg)
        or _limits_layer_pruning_enabled(cfg)
        or _flap_mlp_official_enabled(cfg)
        or _llm_pruner_mlp_enabled(cfg)
    )
    if p["dev_summary"].exists():
        dev_summary = read_json(p["dev_summary"])
    elif final_without_dev_allowed:
        dev_summary = {
            "dev_summary_missing": True,
            "final_without_dev_allowed": True,
            "adaptive_griffin_policy_frozen": True,
            "griffin_prompt_baseline_frozen": _griffin_prompt_enabled(cfg),
            "wanda_official_baseline_frozen": _wanda_official_enabled(cfg),
            "sparsegpt_official_baseline_frozen": _sparsegpt_official_enabled(cfg),
            "shortgpt_baseline_frozen": _shortgpt_enabled(cfg),
            "limits_layer_pruning_baseline_frozen": _limits_layer_pruning_enabled(cfg),
            "flap_mlp_official_baseline_frozen": _flap_mlp_official_enabled(cfg),
            "llm_pruner_mlp_baseline_frozen": _llm_pruner_mlp_enabled(cfg),
        }
    else:
        raise RuntimeError(
            "Development summary is missing. Run evaluate_dev first, or set "
            "evaluation.allow_final_without_dev=true for an adaptive diagnostic run."
        )
    stage_risk_cfg = _adaptive_griffin_cfg(cfg)
    if str(stage_risk_cfg.get("policy")) == "stage_risk_adaptive":
        gate_path = stage_risk_cfg.get("stage_risk_controller", {}).get("dev_gate_path")
        if not gate_path:
            raise RuntimeError("Stage-risk full evaluation requires stage_risk_controller.dev_gate_path")
        gate = read_json(gate_path)
        if not bool(gate.get("full_evaluation_allowed")):
            raise RuntimeError("Stage-risk dev gate failed; full evaluation is forbidden")
    policy_selection_path = _policy_selection_path(cfg)
    policy_methods = None
    policy_selection = None
    if policy_selection_path is not None:
        policy_methods, policy_selection = load_downstream_methods_from_selection(policy_selection_path)
    if policy_methods is not None:
        methods = policy_methods
    elif (
        _adaptive_griffin_enabled(cfg)
        or _griffin_prompt_enabled(cfg)
        or _wanda_official_enabled(cfg)
        or _sparsegpt_official_enabled(cfg)
        or _shortgpt_enabled(cfg)
        or _limits_layer_pruning_enabled(cfg)
        or _flap_mlp_official_enabled(cfg)
        or _llm_pruner_mlp_enabled(cfg)
    ):
        requested_final_methods = set()
        env_final_methods = os.environ.get("STAGE_FINAL_METHODS")
        if env_final_methods:
            requested_final_methods.update(
                name.strip() for name in env_final_methods.split(",") if name.strip()
            )
        elif "final_methods" in profile(cfg):
            requested_final_methods.update(str(name) for name in profile(cfg)["final_methods"])
        dense_methods = []
        if bool(_evaluation_threshold(cfg, "include_ordinary_dense_in_final", False)) or (
            "ordinary_dense" in requested_final_methods
        ):
            dense_methods.append(
                method("ordinary_dense", "trajectory_global", uniform_ratios(0.0), ordinary_prompt(cfg))
            )
        if "structured_dense" in requested_final_methods or _adaptive_griffin_enabled(cfg):
            dense_methods.append(
                method("structured_dense", "trajectory_global", uniform_ratios(0.0), structured_prompt(cfg))
            )
        methods = dense_methods
        if _adaptive_griffin_enabled(cfg):
            methods.extend(adaptive_griffin_methods(cfg))
            methods.append(static_matched_global_method(cfg, [float(value) for value in cfg["masks"]["ratios"]]))
            methods.extend(additional_static_matched_global_methods(cfg))
            methods.extend(additional_fixed_stage_methods(cfg))
        if _griffin_prompt_enabled(cfg):
            methods.extend(griffin_prompt_methods(cfg))
        if _wanda_official_enabled(cfg):
            methods.extend(wanda_official_methods(cfg))
        if _sparsegpt_official_enabled(cfg):
            methods.extend(sparsegpt_official_methods(cfg))
        if _shortgpt_enabled(cfg):
            methods.extend(shortgpt_methods(cfg))
        if _limits_layer_pruning_enabled(cfg):
            methods.extend(limits_layer_pruning_methods(cfg))
        if _flap_mlp_official_enabled(cfg):
            methods.extend(flap_mlp_official_methods(cfg))
        if _llm_pruner_mlp_enabled(cfg):
            methods.extend(llm_pruner_mlp_methods(cfg))
    else:
        frozen = read_json(p["frozen"]) if p["frozen"].exists() else {}
        stage_budget = frozen["stage_budget"]
        shuffled_budget = method(
            "shuffled_stage_budget",
            "shuffled_stage",
            stage_budget["stage_ratios"],
            structured_prompt(cfg),
        )
        methods = [
            method("ordinary_dense", "trajectory_global", uniform_ratios(0.0), ordinary_prompt(cfg)),
            method("structured_dense", "trajectory_global", uniform_ratios(0.0), structured_prompt(cfg)),
            frozen["best_trajectory_global"],
            method(
                "stage_specific_matched_global",
                "stage_specific",
                frozen["best_trajectory_global"]["stage_ratios"],
                structured_prompt(cfg),
            ),
            stage_budget,
            shuffled_budget,
        ]
    methods = _limit_final_methods_for_smoke(cfg, methods)
    smoke_relaxed = cfg["workflow"].get("profile") == "smoke" and bool(
        dev_summary.get("smoke_relaxed_e2e")
    )
    adaptive_policy_frozen = bool(dev_summary.get("adaptive_griffin_policy_frozen"))
    if (
        policy_selection_path is None
        and not smoke_relaxed
        and not adaptive_policy_frozen
        and not final_without_dev_allowed
        and (not dev_summary.get("stage_budget_search_performed") or not dev_summary.get("frozen_policy"))
    ):
        raise RuntimeError("Development gates did not pass; final evaluation is forbidden")
    frozen = read_json(p["frozen"]) if p["frozen"].exists() else {}
    requires_bank = any(method_requires_mask_bank(row) for row in methods)
    expected_metadata = expected_bank_metadata(cfg, p) if requires_bank else {}
    bank = (
        load_mask_bank(
            p["bank"],
            expected_metadata,
            ignored_metadata_keys=("config_hash",),
        )
        if requires_bank
        else None
    )
    bank_metadata = dict(bank.get("metadata", {})) if bank is not None else {}
    metadata_extra = {
        "frozen_policy_hash": stable_hash(frozen),
        "dev_summary_missing_for_final": bool(dev_summary.get("dev_summary_missing")),
        "final_without_dev_allowed": final_without_dev_allowed,
        "mask_bank_loaded": bank is not None,
        "mask_bank_metadata": {
            "config_hash_expected": expected_metadata.get("config_hash"),
            "config_hash_actual": bank_metadata.get("config_hash"),
            "config_hash_mismatch_allowed_for_final": (
                bank_metadata.get("config_hash") != expected_metadata.get("config_hash")
            ),
            "critical_fields_checked": [
                key for key in sorted(expected_metadata) if key != "config_hash"
            ],
        },
    }
    if policy_selection is not None:
        metadata_extra.update(
            {
                "policy_selection_path": str(policy_selection_path),
                "policy_selection_hash": stable_hash(policy_selection),
                "policy_selection_input_manifest_hash": policy_selection.get("input_manifest_hash"),
            }
        )
    final_limit = _final_eval_limit(cfg)
    if final_limit == 0:
        summary_path = _effective_final_summary_path(p)
        write_json(
            summary_path,
            {
                "schema": "stage_calibrated_final_eval_v1",
                **metadata(cfg, **metadata_extra),
                "final_eval_limit": final_limit,
                "final_eval_skipped": True,
                "final_shard": _final_shard_from_env(),
                "policy_selection_used": policy_selection is not None,
                "datasets": {},
                "aggregates": {},
            },
        )
        return
    if policy_selection is None and bool(frozen.get("final_evaluation_forbidden")):
        raise RuntimeError("Frozen policy is diagnostic only; final evaluation is forbidden")
    bundle = load_model_bundle(cfg["model"])
    final_shard = _final_shard_from_env()
    output = {}
    dataset_output_dirs = {}
    env_final_seeds = os.environ.get("STAGE_FINAL_SEEDS")
    if env_final_seeds:
        seeds = [
            int(value.strip())
            for value in env_final_seeds.split(",")
            if value.strip()
        ]
    else:
        seeds = [int(value) for value in profile(cfg).get("final_seeds", [cfg["seed"]])]
    for dataset_cfg in _final_dataset_configs(cfg):
        tasks = load_tasks(dataset_cfg)
        name = _final_dataset_name(dataset_cfg)
        if final_limit is not None:
            tasks = tasks[:final_limit]
        full_task_count = len(tasks)
        tasks = annotate_final_eval_indices(tasks)
        if final_shard is not None:
            tasks = shard_tasks(
                tasks,
                shard_index=final_shard["index"],
                shard_count=final_shard["count"],
            )
            output_dir = shard_dataset_dir(
                p["final_dir"],
                name,
                shard_index=final_shard["index"],
                shard_count=final_shard["count"],
            )
        else:
            output_dir = p["final_dir"] / name
        dataset_output_dirs[name] = output_dir
        output[name] = []
        for seed in seeds:
            output[name].extend(
                _run_methods(
                    cfg,
                    p,
                    tasks,
                    bank,
                    bundle,
                    methods,
                    output_dir,
                    seed=seed,
                )
            )
        if final_shard is not None:
            for summary in output[name]:
                summary["final_shard"] = {
                    **final_shard,
                    "full_task_count": full_task_count,
                    "shard_task_count": len(tasks),
                }
    aggregates = aggregate_final_summaries(
        final_dir=p["final_dir"],
        datasets=output,
        bootstrap_seed=int(cfg["seed"]),
        dataset_row_dirs=dataset_output_dirs,
    )
    write_json(
        _effective_final_summary_path(p),
        {
            "schema": "stage_calibrated_final_eval_v1",
            **metadata(cfg, **metadata_extra),
            "final_eval_limit": final_limit,
            "final_shard": final_shard,
            "policy_selection_used": policy_selection is not None,
            "policy_selection": (
                {
                    "schema": policy_selection.get("schema"),
                    "input_manifest_hash": policy_selection.get("input_manifest_hash"),
                    "selected_policy_roles": sorted(policy_selection.get("selected_policies", {})),
                }
                if policy_selection is not None
                else None
            ),
            "evaluated_methods": [row["name"] for row in methods],
            "datasets": output,
            "aggregates": aggregates,
        },
    )


def command_merge_final_shards(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    policy_selection_path = _policy_selection_path(cfg)
    policy_selection = None
    if policy_selection_path is not None:
        _, policy_selection = load_downstream_methods_from_selection(policy_selection_path)
    frozen = read_json(p["frozen"]) if p["frozen"].exists() else {}
    metadata_extra = {"frozen_policy_hash": stable_hash(frozen)}
    if policy_selection is not None:
        metadata_extra.update(
            {
                "policy_selection_path": str(policy_selection_path),
                "policy_selection_hash": stable_hash(policy_selection),
                "policy_selection_input_manifest_hash": policy_selection.get("input_manifest_hash"),
            }
        )
    shard_count_env = os.environ.get("STAGE_FINAL_SHARD_COUNT")
    shard_count = int(shard_count_env) if shard_count_env else infer_shard_count(p["final_dir"])
    merge_final_shards(
        final_dir=p["final_dir"],
        shard_count=shard_count,
        output_summary_path=p["final_summary"],
        metadata=metadata(cfg, **metadata_extra),
        final_eval_limit=_final_eval_limit(cfg),
        bootstrap_seed=int(cfg["seed"]),
        policy_selection=policy_selection,
    )


def _final_eval_limit(cfg: dict[str, Any]) -> int | None:
    env_limit = os.environ.get("STAGE_FINAL_EVAL_LIMIT")
    if env_limit is not None:
        value = int(env_limit)
        return value if value >= 0 else None
    pcfg = profile(cfg)
    if "final_eval_limit" in pcfg:
        value = int(pcfg["final_eval_limit"])
        return value if value >= 0 else None
    return None


def _policy_selection_path(cfg: dict[str, Any]) -> Path | None:
    env_path = os.environ.get("STAGE_POLICY_SELECTION")
    if env_path:
        return Path(env_path)
    pcfg = profile(cfg)
    value = pcfg.get("policy_selection_path") or cfg.get("evaluation", {}).get("policy_selection_path")
    return Path(value) if value else None


def _final_shard_from_env() -> dict[str, int] | None:
    count = os.environ.get("STAGE_FINAL_SHARD_COUNT")
    index = os.environ.get("STAGE_FINAL_SHARD_INDEX")
    if count is None and index is None:
        return None
    if count is None or index is None:
        raise RuntimeError("Set both STAGE_FINAL_SHARD_INDEX and STAGE_FINAL_SHARD_COUNT for sharded final eval")
    shard = {"index": int(index), "count": int(count)}
    validate_shard(shard_index=shard["index"], shard_count=shard["count"])
    return shard


def _effective_final_summary_path(p: dict[str, Path]) -> Path:
    shard = _final_shard_from_env()
    if shard is None:
        return p["final_summary"]
    return shard_summary_path(
        p["final_dir"],
        shard_index=shard["index"],
        shard_count=shard["count"],
    )


def _limit_final_methods_for_smoke(cfg: dict[str, Any], methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    env_names = os.environ.get("STAGE_FINAL_METHODS")
    if env_names:
        allowed = {name.strip() for name in env_names.split(",") if name.strip()}
        return _dedupe_methods_by_name([row for row in methods if row["name"] in allowed])
    pcfg = profile(cfg)
    if "final_methods" in pcfg:
        allowed = {str(name) for name in pcfg["final_methods"]}
        return _dedupe_methods_by_name([row for row in methods if row["name"] in allowed])
    if cfg["workflow"].get("profile") == "smoke":
        allowed = {"ordinary_dense", "structured_dense", "stage_budget"}
        return _dedupe_methods_by_name([row for row in methods if row["name"] in allowed])
    return _dedupe_methods_by_name(methods)


def _dedupe_methods_by_name(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in methods:
        name = str(row["name"])
        if name in seen:
            continue
        seen.add(name)
        output.append(row)
    return output


def command_summarize(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    required = [
        p["preflight"],
        p["pool_summary"],
        p["trajectory_summary"],
        p["selection_summary"],
        p["bank_validation"],
        p["dev_summary"],
        p["final_summary"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    dev = read_json(p["dev_summary"]) if p["dev_summary"].exists() else None
    final = read_json(p["final_summary"]) if p["final_summary"].exists() else None
    gate = {
        "schema": "stage_calibrated_workflow_gate_v1",
        "completed": not missing,
        "missing_artifacts": missing,
        "logical_mask_only": True,
        "real_speedup_claimed": False,
        "final_test_questions_used_for_decontamination": True,
        "final_test_labels_or_metrics_used_only_after_freeze": bool(
            final and dev and dev["frozen_policy"].get("test_sets_consulted") is False
        ),
    }
    write_json(p["workflow_gate"], gate)
    write_json(
        p["workflow_summary"],
        {
            "schema": "stage_calibrated_workflow_summary_v1",
            **metadata(cfg),
            "gate": gate,
            "pool": read_json(p["pool_summary"]) if p["pool_summary"].exists() else None,
            "trajectories": read_json(p["trajectory_summary"]) if p["trajectory_summary"].exists() else None,
            "selection": read_json(p["selection_summary"]) if p["selection_summary"].exists() else None,
            "mask_validation": read_json(p["bank_validation"]) if p["bank_validation"].exists() else None,
            "dev": dev,
            "final": final,
        },
    )


COMMANDS = {
    "preflight": command_preflight,
    "build_pool": command_build_pool,
    "generate_trajectories": command_generate_trajectories,
    "merge_trajectory_shards": command_merge_trajectory_shards,
    "select_trajectories": command_select_trajectories,
    "calibrate_masks": command_calibrate_masks,
    "validate_masks": command_validate_masks,
    "evaluate_dev": command_evaluate_dev,
    "prepare_sparsegpt": command_prepare_sparsegpt,
    "evaluate_final": command_evaluate_final,
    "merge_final_shards": command_merge_final_shards,
    "summarize": command_summarize,
}


def completion_artifacts(p: dict[str, Path], stage: str) -> tuple[Path, ...]:
    if stage == "evaluate_final":
        return (_effective_final_summary_path(p),)
    if stage == "generate_trajectories":
        shard = _trajectory_shard_from_env()
        return (_trajectory_summary_path(p, shard), _trajectory_candidate_path(p, shard))
    return {
        "preflight": (p["preflight"],),
        "build_pool": (p["pool_summary"], p["pool"]),
        "merge_trajectory_shards": (p["trajectory_summary"], p["candidates"]),
        "select_trajectories": (p["selection_summary"], p["calibration"], p["dev"]),
        "calibrate_masks": (p["bank_summary"], p["bank"]),
        "validate_masks": (p["bank_validation"],),
        "evaluate_dev": (p["dev_summary"], p["frozen"]),
        "prepare_sparsegpt": (p["sparsegpt_summary"],),
        "merge_final_shards": (p["final_summary"],),
        "summarize": (p["workflow_summary"], p["workflow_gate"]),
    }[stage]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=tuple(COMMANDS), required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot", "formal"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    if args.profile:
        cfg["workflow"]["profile"] = args.profile
    if os.environ.get("STAGE_SEED") is not None:
        cfg["seed"] = int(os.environ["STAGE_SEED"])
    if os.environ.get("STAGE_WORKFLOW_ROOT"):
        cfg["workflow"]["root"] = os.environ["STAGE_WORKFLOW_ROOT"]
    if os.environ.get("STAGE_MODEL_NAME_OR_PATH"):
        cfg.setdefault("model", {})["name_or_path"] = os.environ["STAGE_MODEL_NAME_OR_PATH"]
    if os.environ.get("STAGE_MODEL_DTYPE"):
        cfg.setdefault("model", {})["dtype"] = os.environ["STAGE_MODEL_DTYPE"]
    _filter_final_datasets_for_env(cfg)
    set_seed(int(cfg["seed"]))
    p = paths(cfg)
    ensure_dir(p["root"] / PHASES[args.stage])
    config_manifest = p["root"] / "00_preflight" / "config_manifest.json"
    config_hash = stable_hash(cfg)
    if config_manifest.exists():
        existing = read_json(config_manifest)
        if existing.get("config_hash") != config_hash:
            if args.stage == "preflight" and args.force:
                write_json(
                    config_manifest,
                    {
                        "schema": "stage_calibrated_workflow_config_v1",
                        "config_hash": config_hash,
                        "profile": cfg["workflow"]["profile"],
                        "model_name": cfg["model"]["name_or_path"],
                        "refreshed_with_force": True,
                        "previous_config_hash": existing.get("config_hash"),
                    },
                )
            else:
                raise RuntimeError(
                    "Workflow config fingerprint changed. Run preflight with --force "
                    "only if the existing reusable artifacts are compatible, or use a "
                    "different workflow.root."
                )
    elif args.stage != "preflight":
        raise RuntimeError("Run preflight before other workflow stages")
    else:
        write_json(
            config_manifest,
            {
                "schema": "stage_calibrated_workflow_config_v1",
                "config_hash": config_hash,
                "profile": cfg["workflow"]["profile"],
                "model_name": cfg["model"]["name_or_path"],
            },
        )
    artifacts = completion_artifacts(p, args.stage)
    expansion_pending = p["expansion_request"].exists() and args.stage in {
        "build_pool",
        "generate_trajectories",
        "select_trajectories",
    }
    if (
        args.stage == "summarize"
        and p["workflow_gate"].exists()
        and not bool(read_json(p["workflow_gate"]).get("completed"))
    ):
        expansion_pending = True
    if not args.force and not expansion_pending and all(path.exists() for path in artifacts):
        print(f"SKIP completed stage={args.stage}")
        return
    COMMANDS[args.stage](cfg, p)


if __name__ == "__main__":
    main()
