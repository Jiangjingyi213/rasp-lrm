from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from collections import Counter, defaultdict
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
from src.stage_calibration.artifacts import manifest_hash, stable_hash
from src.stage_calibration.calibrate import collect_stage_statistics
from src.stage_calibration.evaluate import evaluate_method, uniform_ratios
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
    decontaminate,
    normalize_big_math_row,
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
    "select_trajectories": "03_selected",
    "calibrate_masks": "04_masks",
    "validate_masks": "04_masks",
    "evaluate_dev": "05_dev",
    "evaluate_final": "06_final",
    "merge_final_shards": "06_final",
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
    protected = [
        *load_tasks({"dataset": "gsm8k", "split": "test"}),
        *load_tasks(
            {
                "dataset": "math500",
                "name_or_path": cfg["evaluation"]["math500_name_or_path"],
                "split": "test",
            }
        ),
    ]
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
    rows = read_jsonl(p["pool"])
    pcfg = profile(cfg)
    repeats = int(pcfg["generations_per_problem"])
    bundle = load_model_bundle(cfg["model"])
    output = p["candidates"]
    existing_rows = read_jsonl(output) if output.exists() else []
    existing_keys = {
        (str(row["id"]), int(row["sample_index"])) for row in existing_rows
    }
    generation = cfg["generation"]
    prompt_cfg = cfg["prompt"]["structured"]
    eos = bundle.tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
    for problem_index, row in enumerate(tqdm(rows, desc="generate-stage-calibration-trajectories")):
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
                    "sample_index": sample_index,
                    "sample_seed": sample_seed,
                    "prompt": prompt,
                    "prompt_token_ids": prompt_ids,
                    "generated_token_ids": generated,
                    "completion": completion,
                    "prediction": extract_answer(completion),
                    "correct": answer_match(completion, row["gold"]),
                    "ended_with_eos": ended_with_eos,
                    "stopped_after_complete_stage_answer": stopping_criteria.triggered,
                    "truncated": not ended_with_eos and len(generated) >= int(generation["max_new_tokens"]),
                    "stage_protocol": stage_protocol,
                },
            )
    current_ids = {str(row["id"]) for row in rows}
    generated_rows = [row for row in read_jsonl(output) if str(row["id"]) in current_ids]
    write_json(
        p["trajectory_summary"],
        {
            "schema": "stage_calibration_candidate_trajectories_v1",
            **metadata(cfg, pool_manifest_hash=manifest_hash(rows)),
            "rows": len(generated_rows),
            "problems": len(rows),
            "correct_rate": sum(int(row["correct"]) for row in generated_rows) / len(generated_rows),
            "valid_stage_rate": sum(int(row["stage_protocol"]["valid"]) for row in generated_rows) / len(generated_rows),
            "truncation_rate": sum(int(row["truncated"]) for row in generated_rows) / len(generated_rows),
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
        row["correct"] = answer_match(completion, gold)
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


def command_calibrate_masks(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    rows = read_jsonl(p["calibration"])
    bundle = load_model_bundle(cfg["model"])
    c4_samples = _effective_c4_samples(cfg)
    metrics, means, stats_summary = collect_stage_statistics(
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


def method(name: str, policy: str, ratios: dict[str, float], prompt: dict[str, Any], bias: bool = True) -> dict[str, Any]:
    return {"name": name, "policy": policy, "stage_ratios": ratios, "prompt": prompt, "bias_compensation": bias}


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


def _run_methods(cfg, p, tasks, bank, bundle, methods, output_dir, seed: int | None = None) -> list[dict[str, Any]]:
    summaries = []
    ensure_dir(output_dir)
    seed = int(cfg["seed"]) if seed is None else int(seed)
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
    bank = load_mask_bank(p["bank"], expected_bank_metadata(cfg, p))
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
    calibration_gate = {
        "comparisons": calibration_comparisons,
        "trajectory_calibration_promising": any(
            row["trajectory_strictly_best"] for row in calibration_comparisons
        ),
    }
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


def command_evaluate_final(cfg: dict[str, Any], p: dict[str, Path]) -> None:
    expected_metadata = expected_bank_metadata(cfg, p)
    bank = load_mask_bank(
        p["bank"],
        expected_metadata,
        ignored_metadata_keys=("config_hash",),
    )
    bank_metadata = dict(bank.get("metadata", {}))
    dev_summary = read_json(p["dev_summary"])
    policy_selection_path = _policy_selection_path(cfg)
    policy_methods = None
    policy_selection = None
    if policy_selection_path is not None:
        policy_methods, policy_selection = load_downstream_methods_from_selection(policy_selection_path)
    smoke_relaxed = cfg["workflow"].get("profile") == "smoke" and bool(
        dev_summary.get("smoke_relaxed_e2e")
    )
    if (
        policy_selection_path is None
        and not smoke_relaxed
        and (not dev_summary.get("stage_budget_search_performed") or not dev_summary.get("frozen_policy"))
    ):
        raise RuntimeError("Development gates did not pass; final evaluation is forbidden")
    frozen = read_json(p["frozen"]) if p["frozen"].exists() else {}
    metadata_extra = {
        "frozen_policy_hash": stable_hash(frozen),
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
    if policy_methods is not None:
        methods = policy_methods
    else:
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
    final_shard = _final_shard_from_env()
    output = {}
    dataset_output_dirs = {}
    seeds = [int(value) for value in profile(cfg).get("final_seeds", [cfg["seed"]])]
    for dataset_cfg in (
        {"dataset": "gsm8k", "split": "test"},
        {
            "dataset": "math500",
            "name_or_path": cfg["evaluation"]["math500_name_or_path"],
            "split": "test",
        },
    ):
        tasks = load_tasks(dataset_cfg)
        name = dataset_cfg["dataset"]
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
    "select_trajectories": command_select_trajectories,
    "calibrate_masks": command_calibrate_masks,
    "validate_masks": command_validate_masks,
    "evaluate_dev": command_evaluate_dev,
    "evaluate_final": command_evaluate_final,
    "merge_final_shards": command_merge_final_shards,
    "summarize": command_summarize,
}


def completion_artifacts(p: dict[str, Path], stage: str) -> tuple[Path, ...]:
    return {
        "preflight": (p["preflight"],),
        "build_pool": (p["pool_summary"], p["pool"]),
        "generate_trajectories": (p["trajectory_summary"], p["candidates"]),
        "select_trajectories": (p["selection_summary"], p["calibration"], p["dev"]),
        "calibrate_masks": (p["bank_summary"], p["bank"]),
        "validate_masks": (p["bank_validation"],),
        "evaluate_dev": (p["dev_summary"], p["frozen"]),
        "evaluate_final": (_effective_final_summary_path(p),),
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
