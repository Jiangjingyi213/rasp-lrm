from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.data.format_prompt import build_prompt
from src.main_generate import generate_text
from src.metrics.answer_match import extract_answer
from src.metrics.flip_rate import answer_flipped
from src.metrics.oracles import summarize_oracles
from src.models.hooks import activation_summary, get_decoder_layers, next_token_stats, token_hidden_states
from src.models.load_model import load_model_bundle
from src.pruning.contexts import pruning_context, select_layers
from src.utils.io import ensure_dir, read_jsonl, read_yaml, write_json, write_jsonl
from src.utils.seed import set_seed


def segment_prefix(completion: str, segment: dict[str, Any], boundary: str) -> str:
    key = "start_char" if boundary == "start" else "end_char"
    return completion[: int(segment[key])].strip()


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
    activation_path = cfg["paths"].get("probe_activation_features", str(Path(out_path).with_suffix(".activation.pt")))

    bundle = load_model_bundle(cfg["model"])
    n_layers = len(get_decoder_layers(bundle.model))
    layers = cf_cfg.get("layers") or list(range(n_layers))
    modules = cf_cfg.get("modules") or [cf_cfg.get("unit", "layer")]
    ratios = cf_cfg.get("ratios") or [cf_cfg.get("ratio", 1.0)]
    max_segments = cf_cfg.get("max_segments_per_example")
    boundary = cf_cfg.get("prefix_boundary", "end")
    generation_cfg = {**cfg.get("generation", {}), **cf_cfg.get("generation", {})}
    hidden_layer = cf_cfg.get("hidden_layer", -1)
    collect_hidden = cf_cfg.get("collect_hidden", True)
    collect_activation = cf_cfg.get("collect_activation", True)

    rows = []
    probe_rows = []
    hidden_tensors = []
    activation_tensors = []
    for item in tqdm(read_jsonl(in_path), desc="counterfactual"):
        baseline = item["completion"]
        baseline_answer = extract_answer(baseline)
        segments = item["segments"][:max_segments] if max_segments else item["segments"]
        for segment in segments:
            prefix = segment_prefix(baseline, segment, boundary)
            conditioned_prompt = build_prompt(item["question"], bundle.tokenizer, cfg.get("prompt", {}), prefix=prefix)
            stats = next_token_stats(
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
            act_features = None
            if collect_activation:
                act_features = torch.tensor(
                    activation_summary(
                        bundle.model,
                        bundle.tokenizer,
                        conditioned_prompt,
                        [int(layer) for layer in layers],
                        max_length=cfg.get("generation", {}).get("max_input_tokens", 2048),
                    ),
                    dtype=torch.float32,
                )
            for module in modules:
                for ratio in ratios:
                    if str(module) == "mlp_intermediate_channels":
                        # Here ratio controls FFN neurons, not the number of
                        # layers. Apply the deployment action to every
                        # configured runtime layer.
                        pruned_layers = [int(layer) for layer in layers]
                    else:
                        pruned_layers = select_layers([int(layer) for layer in layers], float(ratio))
                    with pruning_context(bundle.model, str(module), pruned_layers, float(ratio)):
                        cf_completion = generate_text(bundle, conditioned_prompt, generation_cfg)
                    flipped = answer_flipped(baseline, cf_completion)
                    row = {
                        "id": item["id"],
                        "dataset": item.get("dataset"),
                        "segment_id": segment["segment_id"],
                        "segment_index": segment["segment_id"],
                        "num_segments": len(segments),
                        "segment_type": segment.get("segment_type", "derivation"),
                        "module": str(module),
                        "unit": str(module),
                        "ratio": float(ratio),
                        "pruned_layers": pruned_layers,
                        "layer_id": pruned_layers[0] if len(pruned_layers) == 1 else None,
                        "prefix_boundary": boundary,
                        "segment_text": segment["text"],
                        "baseline_answer": baseline_answer,
                        "counterfactual_answer": extract_answer(cf_completion),
                        "counterfactual_completion": cf_completion,
                        "flipped": flipped,
                        "entropy": stats["entropy"],
                        "confidence": stats["confidence"],
                    }
                    rows.append(row)
                    probe_row = {k: row[k] for k in row if k != "counterfactual_completion"}
                    if collect_hidden:
                        probe_row["hidden_index"] = len(hidden_tensors)
                        hidden_tensors.append(pooled_hidden)
                    if collect_activation:
                        probe_row["activation_index"] = len(activation_tensors)
                        activation_tensors.append(act_features)
                    probe_rows.append(probe_row)

    write_jsonl(out_path, rows)
    if probe_path:
        write_jsonl(probe_path, probe_rows)
    if collect_hidden and hidden_tensors:
        ensure_dir(Path(hidden_path).parent)
        torch.save(torch.stack(hidden_tensors), hidden_path)
    if collect_activation and activation_tensors:
        ensure_dir(Path(activation_path).parent)
        torch.save(torch.stack(activation_tensors), activation_path)
    write_json(str(Path(out_path).with_suffix(".oracles.json")), summarize_oracles(rows))


if __name__ == "__main__":
    main()
