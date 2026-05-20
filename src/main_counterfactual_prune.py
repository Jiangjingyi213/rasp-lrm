from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.data.format_prompt import build_prompt
from src.main_generate import generate_text
from src.metrics.answer_match import extract_answer
from src.metrics.flip_rate import answer_flipped
from src.models.hooks import get_decoder_layers, next_token_entropy, token_hidden_states
from src.models.load_model import load_model_bundle
from src.pruning.attention_pruner import attention_zero
from src.pruning.layer_skipper import layer_skip
from src.pruning.mlp_pruner import mlp_zero
from src.utils.io import ensure_dir, read_jsonl, read_yaml, write_json, write_jsonl
from src.utils.seed import set_seed


def prune_context(model, unit: str, layer_id: int):
    if unit == "layer":
        return layer_skip(model, [layer_id])
    if unit == "mlp":
        return mlp_zero(model, [layer_id])
    if unit == "attention":
        return attention_zero(model, [layer_id])
    if unit == "none":
        return nullcontext()
    raise ValueError(f"Unsupported prune unit: {unit}")


def segment_prefix(completion: str, segment: dict[str, Any], boundary: str) -> str:
    key = "start_char" if boundary == "start" else "end_char"
    return completion[: int(segment[key])].strip()


def summarize_oracles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    layer_scores: dict[int, list[int]] = {}
    prompt_layer_scores: dict[tuple[str, int], list[int]] = {}
    step_layer_scores: dict[tuple[str, int], list[int]] = {}
    for row in rows:
        flip = int(row["flipped"])
        layer_id = int(row["layer_id"])
        segment_id = int(row["segment_id"])
        layer_scores.setdefault(layer_id, []).append(flip)
        prompt_layer_scores.setdefault((row["id"], layer_id), []).append(flip)
        step_layer_scores.setdefault((row["id"], segment_id), []).append(flip)

    static_best = max((sum(v) / len(v), k) for k, v in layer_scores.items())
    prompt_best_by_id: dict[str, float] = {}
    for (item_id, _layer_id), values in prompt_layer_scores.items():
        prompt_best_by_id[item_id] = max(prompt_best_by_id.get(item_id, 0.0), sum(values) / len(values))
    step_best = [max(values) for values in step_layer_scores.values()]
    prompt_oracle = sum(prompt_best_by_id.values()) / len(prompt_best_by_id)
    step_oracle = sum(step_best) / len(step_best)
    return {
        "static_oracle_best_layer": static_best[1],
        "static_oracle_flip_rate": static_best[0],
        "prompt_oracle_flip_rate": prompt_oracle,
        "step_oracle_flip_rate": step_oracle,
        "n_counterfactuals": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    set_seed(cfg.get("seed", 1))
    cf_cfg = cfg.get("counterfactual", {})
    in_path = args.input or cfg["paths"]["segments"]
    out_path = args.output or cfg["paths"]["counterfactuals"]
    probe_path = cfg["paths"].get("probe_dataset")
    hidden_path = cfg["paths"].get("probe_hidden_states", str(Path(out_path).with_suffix(".hidden.pt")))

    bundle = load_model_bundle(cfg["model"])
    n_layers = len(get_decoder_layers(bundle.model))
    layers = cf_cfg.get("layers") or list(range(n_layers))
    max_segments = cf_cfg.get("max_segments_per_example")
    unit = cf_cfg.get("unit", "layer")
    boundary = cf_cfg.get("prefix_boundary", "end")
    generation_cfg = {**cfg.get("generation", {}), **cf_cfg.get("generation", {})}
    hidden_layer = cf_cfg.get("hidden_layer", -1)
    collect_hidden = cf_cfg.get("collect_hidden", True)

    rows = []
    probe_rows = []
    hidden_tensors = []
    for item in tqdm(read_jsonl(in_path), desc="counterfactual"):
        baseline = item["completion"]
        baseline_answer = extract_answer(baseline)
        segments = item["segments"][:max_segments] if max_segments else item["segments"]
        for segment in segments:
            prefix = segment_prefix(baseline, segment, boundary)
            conditioned_prompt = build_prompt(item["question"], bundle.tokenizer, cfg.get("prompt", {}), prefix=prefix)
            entropy = next_token_entropy(
                bundle.model,
                bundle.tokenizer,
                conditioned_prompt,
                max_length=cfg.get("generation", {}).get("max_input_tokens", 2048),
            )
            pooled_hidden = None
            if collect_hidden:
                hidden = token_hidden_states(
                    bundle.model,
                    bundle.tokenizer,
                    conditioned_prompt,
                    layer=hidden_layer,
                    max_length=cfg.get("generation", {}).get("max_input_tokens", 2048),
                )
                pooled_hidden = hidden[-1].clone()
            for layer_id in layers:
                with prune_context(bundle.model, unit, int(layer_id)):
                    cf_completion = generate_text(bundle, conditioned_prompt, generation_cfg)
                flipped = answer_flipped(baseline, cf_completion)
                row = {
                    "id": item["id"],
                    "dataset": item.get("dataset"),
                    "segment_id": segment["segment_id"],
                    "layer_id": int(layer_id),
                    "unit": unit,
                    "prefix_boundary": boundary,
                    "segment_text": segment["text"],
                    "baseline_answer": baseline_answer,
                    "counterfactual_answer": extract_answer(cf_completion),
                    "counterfactual_completion": cf_completion,
                    "flipped": flipped,
                    "entropy": entropy,
                }
                rows.append(row)
                probe_row = {k: row[k] for k in row if k != "counterfactual_completion"}
                if collect_hidden:
                    probe_row["hidden_index"] = len(hidden_tensors)
                    hidden_tensors.append(pooled_hidden)
                probe_rows.append(probe_row)

    write_jsonl(out_path, rows)
    if probe_path:
        write_jsonl(probe_path, probe_rows)
    if collect_hidden and hidden_tensors:
        ensure_dir(Path(hidden_path).parent)
        torch.save(torch.stack(hidden_tensors), hidden_path)
    write_json(str(Path(out_path).with_suffix(".oracles.json")), summarize_oracles(rows))


if __name__ == "__main__":
    main()
