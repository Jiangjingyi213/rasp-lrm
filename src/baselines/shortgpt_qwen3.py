from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from src.data.format_prompt import build_prompt
from src.models.hooks import get_decoder_layers, model_device
from src.utils.io import read_jsonl


@dataclass
class ShortGPTSummary:
    method: str
    source: str
    baseline_type: str
    pruning_granularity: str
    prune_ratio: float
    density: float
    calibration_path: str
    calibration_samples: int
    calibration_max_input_tokens: int
    total_layers: int
    candidate_layers: list[int]
    pruned_layers: list[int]
    kept_layers: list[int]
    block_influence_by_layer: dict[str, float]
    selection_method: str
    pruning_method: str
    real_speedup_claimed: bool
    matched_rasp_reference: str
    target_matched_to_rasp_actual_mlp_pruning: float | None


def _decoder_layer_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _identity_layer_output(inputs: tuple[Any, ...], output: Any) -> Any:
    hidden_states = inputs[0]
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    return hidden_states


def _validate_layer_ids(layer_ids: Iterable[int], total_layers: int, *, name: str) -> list[int]:
    validated = sorted({int(layer_id) for layer_id in layer_ids})
    for layer_id in validated:
        if layer_id < 0 or layer_id >= total_layers:
            raise ValueError(f"{name} contains layer id {layer_id}, outside [0, {total_layers})")
    return validated


def _candidate_layers(
    total_layers: int,
    *,
    candidate_layers: Iterable[int] | None = None,
    protected_first_layers: int = 0,
    protected_last_layers: int = 0,
) -> list[int]:
    if candidate_layers is not None:
        return _validate_layer_ids(candidate_layers, total_layers, name="candidate_layers")
    first = max(0, int(protected_first_layers))
    last = max(0, int(protected_last_layers))
    end = max(first, total_layers - last)
    return list(range(first, end))


def _pruned_layer_count(total_candidates: int, prune_ratio: float) -> int:
    if not 0.0 <= float(prune_ratio) < 1.0:
        raise ValueError(f"ShortGPT prune_ratio must be in [0, 1), got {prune_ratio}")
    if prune_ratio <= 0.0 or total_candidates <= 0:
        return 0
    return min(total_candidates, max(1, int(round(total_candidates * float(prune_ratio)))))


def select_reverse_layers(
    *,
    prune_ratio: float,
    total_layers: int,
    candidate_layers: Iterable[int] | None = None,
    protected_first_layers: int = 0,
    protected_last_layers: int = 0,
) -> list[int]:
    candidates = _candidate_layers(
        total_layers,
        candidate_layers=candidate_layers,
        protected_first_layers=protected_first_layers,
        protected_last_layers=protected_last_layers,
    )
    prune = _pruned_layer_count(len(candidates), prune_ratio)
    return sorted(sorted(candidates, reverse=True)[:prune])


@torch.no_grad()
def collect_shortgpt_block_influence_qwen3(
    model: nn.Module,
    tokenizer,
    calibration_rows: list[dict[str, Any]],
    *,
    prompt_config: dict[str, Any],
    calibration_samples: int,
    max_input_tokens: int,
) -> dict[int, float]:
    """Collect ShortGPT Block Influence scores for Qwen3 decoder layers.

    BI follows ShortGPT's layer-removal criterion: 1 - cosine similarity between
    a block's input hidden states and output hidden states, averaged over tokens
    and calibration prompts. Lower BI means the block is treated as more
    removable.
    """

    layers = get_decoder_layers(model)
    total_layers = len(layers)
    similarity_sum = torch.zeros(total_layers, dtype=torch.float64)
    token_count = torch.zeros(total_layers, dtype=torch.float64)
    handles = []

    def make_hook(layer_id: int):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            layer_input = inputs[0].detach()
            layer_output = _decoder_layer_output(output).detach()
            if layer_input.ndim != 3 or layer_output.ndim != 3:
                raise ValueError(
                    "ShortGPT BI expects decoder hidden states with shape "
                    f"[batch, tokens, hidden], got {tuple(layer_input.shape)} and "
                    f"{tuple(layer_output.shape)}"
                )
            cosine = F.cosine_similarity(layer_input.float(), layer_output.float(), dim=-1)
            similarity_sum[layer_id] += cosine.double().sum().cpu()
            token_count[layer_id] += float(cosine.numel())

        return hook

    for layer_id, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(layer_id)))

    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False
    try:
        rows = calibration_rows[: int(calibration_samples)]
        for row in tqdm(rows, desc="shortgpt-bi-calibration"):
            prompt = build_prompt(str(row["question"]), tokenizer, prompt_config)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
            inputs = inputs.to(model_device(model))
            model(**inputs, use_cache=False, return_dict=True)
    finally:
        if use_cache is not None:
            model.config.use_cache = use_cache
        for handle in handles:
            handle.remove()

    if torch.any(token_count <= 0):
        missing = torch.nonzero(token_count <= 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"ShortGPT BI did not observe decoder layers: {missing}")
    mean_similarity = similarity_sum / token_count.clamp_min(1.0)
    block_influence = 1.0 - mean_similarity
    return {layer_id: float(block_influence[layer_id].item()) for layer_id in range(total_layers)}


def select_shortgpt_layers(
    block_influence: dict[int, float],
    *,
    prune_ratio: float,
    total_layers: int,
    candidate_layers: Iterable[int] | None = None,
    protected_first_layers: int = 0,
    protected_last_layers: int = 0,
) -> list[int]:
    candidates = _candidate_layers(
        total_layers,
        candidate_layers=candidate_layers,
        protected_first_layers=protected_first_layers,
        protected_last_layers=protected_last_layers,
    )
    prune = _pruned_layer_count(len(candidates), prune_ratio)
    ranked = sorted(candidates, key=lambda layer_id: (block_influence[layer_id], layer_id))
    return sorted(ranked[:prune])


def apply_shortgpt_layer_skip_hooks(model: nn.Module, pruned_layers: Iterable[int]) -> list[Any]:
    layers = get_decoder_layers(model)
    selected = _validate_layer_ids(pruned_layers, len(layers), name="pruned_layers")
    handles = []
    for layer_id in selected:
        handles.append(layers[layer_id].register_forward_hook(lambda _m, inp, out: _identity_layer_output(inp, out)))
    return handles


@torch.no_grad()
def prepare_shortgpt_qwen3(
    model: nn.Module,
    tokenizer,
    *,
    calibration_path: str,
    prompt_config: dict[str, Any],
    prune_ratio: float,
    selection_method: str = "block_influence",
    calibration_samples: int = 128,
    max_input_tokens: int = 2048,
    candidate_layers: Iterable[int] | None = None,
    protected_first_layers: int = 0,
    protected_last_layers: int = 0,
    pruned_layers: Iterable[int] | None = None,
    matched_rasp_reference: str = "",
    target_matched_to_rasp_actual_mlp_pruning: float | None = None,
) -> tuple[ShortGPTSummary, list[Any]]:
    layers = get_decoder_layers(model)
    total_layers = len(layers)
    candidates = _candidate_layers(
        total_layers,
        candidate_layers=candidate_layers,
        protected_first_layers=protected_first_layers,
        protected_last_layers=protected_last_layers,
    )
    if pruned_layers is None:
        normalized_selection = str(selection_method).lower()
        if normalized_selection in {"reverse", "reverse_order", "tail", "last_layers"}:
            sample_count = 0
            block_influence = {}
            selected = select_reverse_layers(
                prune_ratio=float(prune_ratio),
                total_layers=total_layers,
                candidate_layers=candidates,
            )
            selection_method = "reverse_order_tail_layers"
        elif normalized_selection in {"block_influence", "bi", "shortgpt_bi"}:
            calibration_rows = read_jsonl(calibration_path)
            sample_count = min(int(calibration_samples), len(calibration_rows))
            if sample_count <= 0:
                raise ValueError(f"No ShortGPT calibration rows found at {calibration_path}")
            block_influence = collect_shortgpt_block_influence_qwen3(
                model,
                tokenizer,
                calibration_rows,
                prompt_config=prompt_config,
                calibration_samples=sample_count,
                max_input_tokens=int(max_input_tokens),
            )
            selected = select_shortgpt_layers(
                block_influence,
                prune_ratio=float(prune_ratio),
                total_layers=total_layers,
                candidate_layers=candidates,
            )
            selection_method = "block_influence_lowest"
        else:
            raise ValueError(f"Unsupported layer pruning selection_method: {selection_method}")
    else:
        selected = _validate_layer_ids(pruned_layers, total_layers, name="pruned_layers")
        sample_count = 0
        block_influence = {}
        selection_method = "manual_layer_ids"

    handles = apply_shortgpt_layer_skip_hooks(model, selected)
    kept = [layer_id for layer_id in range(total_layers) if layer_id not in set(selected)]
    actual_ratio = len(selected) / total_layers if total_layers else 0.0
    summary = ShortGPTSummary(
        method="shortgpt_qwen3",
        source="ShortGPT-style depth pruning for Qwen3 using Block Influence, arXiv:2403.03853.",
        baseline_type="shortgpt_depth_pruning",
        pruning_granularity="decoder_layer_logical_skip",
        prune_ratio=float(prune_ratio),
        density=1.0 - actual_ratio,
        calibration_path=str(calibration_path),
        calibration_samples=sample_count,
        calibration_max_input_tokens=int(max_input_tokens),
        total_layers=total_layers,
        candidate_layers=candidates,
        pruned_layers=selected,
        kept_layers=kept,
        block_influence_by_layer={str(k): float(v) for k, v in sorted(block_influence.items())},
        selection_method=selection_method,
        pruning_method="decoder_layer_forward_hook_identity_output",
        real_speedup_claimed=False,
        matched_rasp_reference=str(matched_rasp_reference),
        target_matched_to_rasp_actual_mlp_pruning=target_matched_to_rasp_actual_mlp_pruning,
    )
    return summary, handles


def summary_to_dict(summary: ShortGPTSummary) -> dict[str, Any]:
    return asdict(summary)
