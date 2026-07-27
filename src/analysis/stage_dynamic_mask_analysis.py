from __future__ import annotations

import argparse
import hashlib
import itertools
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.data.load_gsm8k import load_tasks
from src.main_stage_calibrated_pruning import adaptive_griffin_methods
from src.metrics.answer_match import answer_match, extract_answer
from src.models.load_model import load_model_bundle
from src.stage_calibration.decode import decode_with_stage_masks
from src.stage_calibration.mask_bank import load_mask_bank
from src.stage_calibration.protocol import STAGES
from src.stage_calibration.runtime import (
    SafeDynamicStageGriffinRuntime,
    _keep_mask_from_scores,
    apply_adaptive_stage_griffin_qwen3,
)
from src.utils.io import ensure_dir, read_yaml, write_json


def _mask_hash(mask: torch.Tensor) -> str:
    values = mask.detach().to(device="cpu", dtype=torch.bool).numpy().tobytes()
    return hashlib.sha1(values).hexdigest()[:16]


def _jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().to(device="cpu", dtype=torch.bool)
    right = right.detach().to(device="cpu", dtype=torch.bool)
    union = torch.logical_or(left, right).sum().item()
    if not union:
        return 1.0
    return float(torch.logical_and(left, right).sum().item()) / float(union)


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _summarize(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _default_stage_ratio(method: dict[str, Any]) -> float:
    ratios = [float(value) for value in method.get("stage_ratios", {}).values()]
    ratios = [value for value in ratios if value > 0.0]
    return float(statistics.mean(ratios)) if ratios else 0.0


def prior_overlap_analysis(
    bank: dict[str, Any],
    *,
    policy: str,
    ratio_by_stage: dict[str, float],
) -> dict[str, Any]:
    if policy not in bank["policies"]:
        raise ValueError(f"Unknown mask policy {policy!r}; available={sorted(bank['policies'])}")
    stage_masks: dict[str, dict[int, torch.Tensor]] = {}
    for stage in STAGES:
        stage_masks[stage] = {}
        ratio = float(ratio_by_stage.get(stage, _default_stage_ratio({"stage_ratios": ratio_by_stage})))
        for layer_id in bank["layers"]:
            entry = bank["policies"][policy][stage][int(layer_id)]
            stage_masks[stage][int(layer_id)] = _keep_mask_from_scores(entry["metric"], ratio)

    pair_values: dict[str, list[float]] = {}
    for left, right in itertools.combinations(STAGES, 2):
        values = [
            _jaccard(stage_masks[left][int(layer_id)], stage_masks[right][int(layer_id)])
            for layer_id in bank["layers"]
        ]
        pair_values[f"{left}_vs_{right}"] = values

    per_pair = {name: _summarize(values) for name, values in pair_values.items()}
    per_layer: list[dict[str, Any]] = []
    for layer_id in bank["layers"]:
        row = {"layer": int(layer_id)}
        for left, right in itertools.combinations(STAGES, 2):
            row[f"{left}_vs_{right}"] = _jaccard(
                stage_masks[left][int(layer_id)],
                stage_masks[right][int(layer_id)],
            )
        per_layer.append(row)
    all_values = [value for values in pair_values.values() for value in values]
    return {
        "policy": policy,
        "ratio_by_stage": {stage: float(ratio_by_stage.get(stage, 0.0)) for stage in STAGES},
        "layers": [int(layer_id) for layer_id in bank["layers"]],
        "stage_pair_jaccard": per_pair,
        "overall_stage_pair_jaccard": _summarize(all_values),
        "per_layer": per_layer,
        "interpretation": (
            "Lower Jaccard means the calibration prior keeps different FFN channels for "
            "different reasoning stages."
        ),
    }


class MaskTracingSafeDynamicRuntime(SafeDynamicStageGriffinRuntime):
    def __init__(self, *args: Any, trace_keep_indices: int = 16, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace_keep_indices = int(trace_keep_indices)
        self.trace_events: list[dict[str, Any]] = []
        self.final_masks: dict[tuple[str, int], torch.Tensor] = {}

    def reset_trace(self) -> None:
        self.trace_events.clear()
        self.final_masks.clear()

    def keep_mask(self, stage: str, layer_id: int) -> torch.Tensor:
        was_cached = (stage, layer_id) in self._mask_cache
        mask = super().keep_mask(stage, layer_id)
        if not was_cached:
            mask_cpu = mask.detach().to(device="cpu", dtype=torch.bool)
            keep_indices = torch.nonzero(mask_cpu, as_tuple=False).flatten()
            self.trace_events.append(
                {
                    "stage": stage,
                    "layer": int(layer_id),
                    "mask_hash": _mask_hash(mask_cpu),
                    "keep_count": int(mask_cpu.sum().item()),
                    "channel_count": int(mask_cpu.numel()),
                    "pruning_ratio": 1.0 - float(mask_cpu.float().mean().item()),
                    "keep_indices_head": [
                        int(value) for value in keep_indices[: self.trace_keep_indices].tolist()
                    ],
                }
            )
        self.final_masks[(stage, int(layer_id))] = mask.detach().to(device="cpu", dtype=torch.bool)
        return mask


def _method_by_name(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    methods = adaptive_griffin_methods(cfg)
    for method in methods:
        if method["name"] == name:
            return method
    names = ", ".join(method["name"] for method in methods)
    raise ValueError(f"Could not find adaptive method {name!r}; available: {names}")


def _limited_tasks(dataset: str, limit: int, offset: int) -> list[dict[str, Any]]:
    if dataset == "gsm8k":
        cfg = {"dataset": "gsm8k", "split": "test", "limit": limit, "offset": offset}
    elif dataset == "math500":
        cfg = {
            "dataset": "math500",
            "name_or_path": "HuggingFaceH4/MATH-500",
            "split": "test",
            "limit": limit,
            "offset": offset,
        }
    elif dataset == "arc_challenge":
        cfg = {
            "dataset": "arc_challenge",
            "name_or_path": "allenai/ai2_arc",
            "dataset_config": "ARC-Challenge",
            "split": "test",
            "limit": limit,
            "offset": offset,
        }
    else:
        raise ValueError(f"Unsupported probe dataset: {dataset}")
    return load_tasks(cfg)


@torch.no_grad()
def runtime_diversity_probe(
    *,
    cfg: dict[str, Any],
    bank: dict[str, Any],
    method: dict[str, Any],
    dataset: str,
    limit: int,
    offset: int,
    max_new_tokens: int,
    trace_keep_indices: int,
) -> dict[str, Any]:
    tasks = _limited_tasks(dataset, limit, offset)
    bundle = load_model_bundle(cfg["model"])
    runtime = MaskTracingSafeDynamicRuntime(
        bank,
        stage_ratios={stage: float(method["stage_ratios"].get(stage, 0.0)) for stage in STAGES},
        runtime_weight=float(method.get("runtime_weight", 0.25)),
        prior_weight=float(method.get("prior_weight", 0.75)),
        warmup_tokens={stage: int(method.get("warmup_tokens", {}).get(stage, 0)) for stage in STAGES},
        protected_core_ratios={
            stage: float(method.get("protected_core_ratios", {}).get(stage, 0.0)) for stage in STAGES
        },
        refresh_intervals={
            stage: int(method.get("refresh_intervals", {}).get(stage, 0)) for stage in STAGES
        },
        window_tokens={stage: int(method.get("window_tokens", {}).get(stage, 1)) for stage in STAGES},
        bias_compensation=bool(method.get("bias_compensation", True)),
        prior_policy=str(method.get("prior_policy", "stage_specific")),
        trace_keep_indices=trace_keep_indices,
    )
    apply_adaptive_stage_griffin_qwen3(bundle.model, runtime)

    generation = dict(cfg.get("generation", {}))
    prompt_cfg = dict(method.get("prompt", cfg.get("prompt", {}).get("structured", {})))
    rows: list[dict[str, Any]] = []
    masks_by_stage_layer: dict[str, dict[int, list[torch.Tensor]]] = {
        stage: defaultdict(list) for stage in STAGES
    }
    hashes_by_stage_layer: dict[str, dict[int, set[str]]] = {
        stage: defaultdict(set) for stage in STAGES
    }

    from src.data.format_prompt import build_prompt, forced_assistant_prefix

    for index, task in enumerate(tasks):
        runtime.reset()
        runtime.reset_trace()
        prompt = build_prompt(task["question"], bundle.tokenizer, prompt_cfg)
        result = decode_with_stage_masks(
            bundle.model,
            bundle.tokenizer,
            prompt,
            runtime,
            prefill=forced_assistant_prefix(prompt_cfg),
            max_new_tokens=max_new_tokens,
            max_input_tokens=int(generation.get("max_input_tokens", 4096)),
            temperature=float(generation.get("temperature", 0.6)),
            top_p=float(generation.get("top_p", 0.95)),
            top_k=int(generation.get("top_k", 20)),
        )
        for (stage, layer_id), mask in runtime.final_masks.items():
            masks_by_stage_layer[stage][layer_id].append(mask)
            hashes_by_stage_layer[stage][layer_id].add(_mask_hash(mask))
        rows.append(
            {
                "index": index,
                "id": task.get("id"),
                "dataset": task.get("dataset", dataset),
                "correct": answer_match(
                    result["completion"],
                    str(task.get("gold", "")),
                    answer_type=task.get("answer_type"),
                ),
                "prediction": extract_answer(result["completion"]),
                "gold": task.get("gold"),
                "generated_tokens": result["generated_tokens"],
                "truncated": result["truncated"],
                "fallback_reason": result["runtime_stage_mask"].get("fallback_reason"),
                "tokens_by_stage": result["runtime_stage_mask"].get("tokens_by_stage", {}),
                "masked_tokens_by_stage": result["runtime_stage_mask"].get("masked_tokens_by_stage", {}),
                "mask_refresh_count_by_stage": result["runtime_stage_mask"].get(
                    "mask_refresh_count_by_stage", {}
                ),
                "trace_events": runtime.trace_events,
            }
        )

    per_stage_layer: list[dict[str, Any]] = []
    stage_level: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        stage_values: list[float] = []
        unique_rates: list[float] = []
        for layer_id in sorted(masks_by_stage_layer[stage]):
            masks = masks_by_stage_layer[stage][layer_id]
            pairwise = [
                _jaccard(left, right)
                for left, right in itertools.combinations(masks, 2)
            ]
            unique_count = len(hashes_by_stage_layer[stage][layer_id])
            unique_rate = unique_count / len(masks) if masks else 0.0
            unique_rates.append(unique_rate)
            stage_values.extend(pairwise)
            per_stage_layer.append(
                {
                    "stage": stage,
                    "layer": int(layer_id),
                    "samples_with_mask": len(masks),
                    "unique_mask_count": unique_count,
                    "unique_mask_rate": unique_rate,
                    "pairwise_jaccard": _summarize(pairwise),
                }
            )
        stage_level[stage] = {
            "pairwise_jaccard": _summarize(stage_values),
            "unique_mask_rate": _summarize(unique_rates),
        }

    return {
        "schema": "stage_dynamic_runtime_mask_diversity_v1",
        "method": method["name"],
        "dataset": dataset,
        "limit": limit,
        "offset": offset,
        "max_new_tokens": max_new_tokens,
        "problems": len(rows),
        "correct": sum(int(row["correct"]) for row in rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows) if rows else 0.0,
        "stage_level": stage_level,
        "per_stage_layer": per_stage_layer,
        "rows": rows,
        "interpretation": (
            "A unique_mask_rate above zero and pairwise Jaccard below one indicate that "
            "different problems select different channel masks within the same stage."
        ),
    }


def write_markdown(path: Path, prior: dict[str, Any], runtime: dict[str, Any] | None) -> None:
    lines = [
        "# Stage-Dynamic Mask Analysis",
        "",
        "## Stage-Conditioned Calibration Prior",
        "",
        "This section compares masks built from calibration WIFV metrics for different reasoning stages.",
        "",
        "| stage pair | mean Jaccard | median | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in prior["stage_pair_jaccard"].items():
        lines.append(
            f"| `{name}` | {stats['mean']:.4f} | {stats['median']:.4f} | "
            f"{stats['min']:.4f} | {stats['max']:.4f} |"
        )
    overall = prior["overall_stage_pair_jaccard"]
    lines += [
        "",
        f"Overall mean stage-pair Jaccard: `{overall['mean']:.4f}`. Lower values mean stronger stage-conditioned mask diversity.",
    ]
    if runtime is not None:
        lines += [
            "",
            "## Runtime Mask Diversity",
            "",
            f"Probe method: `{runtime['method']}`; dataset: `{runtime['dataset']}`; problems: `{runtime['problems']}`.",
            "",
            "| stage | mean pairwise Jaccard | median | mean unique mask rate |",
            "|---|---:|---:|---:|",
        ]
        for stage in STAGES:
            pairwise = runtime["stage_level"][stage]["pairwise_jaccard"]
            unique = runtime["stage_level"][stage]["unique_mask_rate"]
            mean_pair = "n/a" if pairwise["mean"] is None else f"{pairwise['mean']:.4f}"
            med_pair = "n/a" if pairwise["median"] is None else f"{pairwise['median']:.4f}"
            mean_unique = "n/a" if unique["mean"] is None else f"{unique['mean']:.4f}"
            lines.append(f"| `{stage}` | {mean_pair} | {med_pair} | {mean_unique} |")
        lines += [
            "",
            "Interpretation: if same-stage masks were identical across problems, Jaccard would be 1 and unique mask rate would be near 0. Values below 1 show sample-adaptive channel selection.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage_calibrated_pruning/mixed_reasoning_seed3_t30_math_safe_full.yaml")
    parser.add_argument("--source-root", default="runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3")
    parser.add_argument("--method", default="t30_math_safe")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "arc_challenge"])
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--prior-policy", default="stage_specific")
    parser.add_argument("--prior-only", action="store_true")
    parser.add_argument("--trace-keep-indices", type=int, default=16)
    parser.add_argument("--output-dir", default="runs/08_stage_calibrated_pruning/analysis_stage_dynamic_masks")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    method = _method_by_name(cfg, args.method)
    bank = load_mask_bank(Path(args.source_root) / "04_masks" / "mask_bank.pt")
    prior = prior_overlap_analysis(
        bank,
        policy=args.prior_policy,
        ratio_by_stage={stage: float(method["stage_ratios"].get(stage, 0.0)) for stage in STAGES},
    )
    runtime = None
    if not args.prior_only:
        runtime = runtime_diversity_probe(
            cfg=cfg,
            bank=bank,
            method=method,
            dataset=args.dataset,
            limit=args.limit,
            offset=args.offset,
            max_new_tokens=args.max_new_tokens,
            trace_keep_indices=args.trace_keep_indices,
        )
    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "stage_prior_overlap.json", prior)
    if runtime is not None:
        write_json(out_dir / "runtime_mask_diversity.json", runtime)
    write_markdown(out_dir / "stage_dynamic_mask_analysis.md", prior, runtime)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
