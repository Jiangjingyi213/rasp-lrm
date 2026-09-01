from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import torch
from torch import nn
from tqdm import tqdm

from src.data.format_prompt import build_prompt
from src.models.hooks import get_decoder_layers, model_device
from src.utils.io import read_json, read_jsonl


@dataclass
class GispMlpSummary:
    method: str
    source: str
    baseline_type: str
    pruning_granularity: str
    ratio: float
    iterations: int
    calibration_path: str
    calibration_samples: int
    calibration_max_input_tokens: int
    calibration_prompt_mode: str
    calibration_text_field: str
    score_normalization: str
    target: str
    physical_pruning: bool
    total_layers: int
    pruned_layers: list[int]
    original_intermediate_size: int
    kept_channels_per_layer: dict[str, int]
    pruned_channels_per_layer: dict[str, int]
    actual_mlp_channel_pruning_ratio: float
    matched_rasp_reference: str
    target_matched_to_rasp_actual_mlp_pruning: float | None
    prune_skip: bool = True
    protected_layers: list[int] = field(default_factory=list)
    upstream_reference: str = (
        "uncc-efficient-ai/GISP external_code/GISP/pruners/grad_sp_global.py; "
        "adapted as a Qwen3 MLP-channel-only logical-mask port."
    )


def _qwen3_mlp(layer: nn.Module) -> nn.Module:
    mlp = getattr(layer, "mlp", None)
    if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
        raise ValueError("Expected Qwen3 MLP with gate_proj/up_proj/down_proj")
    return mlp


def _layer_ids(model: nn.Module, requested: Iterable[int] | None = None) -> list[int]:
    layers = get_decoder_layers(model)
    if requested is None:
        return list(range(len(layers)))
    ids = [int(i) for i in requested]
    for layer_id in ids:
        if layer_id < 0 or layer_id >= len(layers):
            raise ValueError(f"Layer id {layer_id} is out of range for {len(layers)} decoder layers")
    return ids


def _intermediate_size(mlp: nn.Module) -> int:
    gate = int(mlp.gate_proj.weight.shape[0])
    up = int(mlp.up_proj.weight.shape[0])
    down = int(mlp.down_proj.weight.shape[1])
    if gate != up or gate != down:
        raise ValueError("Qwen3 MLP projections do not share a common intermediate width")
    return gate


def _protected_layer_ids(
    selected_layers: Iterable[int],
    total_layers: int,
    *,
    prune_skip: bool,
) -> set[int]:
    if not prune_skip:
        return set()
    prunable = set(range(int(total_layers * 0.1), int(total_layers) - 1))
    return {int(layer_id) for layer_id in selected_layers if int(layer_id) not in prunable}


def _resolve_prompt(
    row: dict[str, Any],
    tokenizer,
    prompt_config: dict[str, Any],
    calibration_prompt_mode: str,
    calibration_text_field: str,
) -> str:
    if calibration_prompt_mode == "raw_text":
        if calibration_text_field not in row:
            raise KeyError(
                f"GISP raw_text calibration expected field {calibration_text_field!r}; "
                f"available fields={sorted(row)}"
            )
        return str(row[calibration_text_field])
    if calibration_prompt_mode in {"structured_prompt", "prompt"}:
        return build_prompt(str(row["question"]), tokenizer, prompt_config)
    raise ValueError(
        "Unsupported GISP calibration_prompt_mode="
        f"{calibration_prompt_mode!r}; expected raw_text or structured_prompt"
    )


def _install_logical_mlp_masks(
    model: nn.Module,
    masks: dict[int, torch.Tensor],
) -> list[Any]:
    handles = []
    layers = get_decoder_layers(model)

    def make_hook(layer_id: int):
        def hook(_module: nn.Module, inputs):
            intermediate = inputs[0]
            mask = masks[layer_id].to(device=intermediate.device, dtype=intermediate.dtype)
            return (intermediate * mask, *inputs[1:])

        return hook

    for layer_id in sorted(masks):
        mlp = _qwen3_mlp(layers[layer_id])
        handles.append(mlp.down_proj.register_forward_pre_hook(make_hook(layer_id)))
    return handles


def _set_only_target_weights_trainable(
    model: nn.Module,
    selected_layers: list[int],
) -> list[tuple[nn.Parameter, bool]]:
    previous = [(param, bool(param.requires_grad)) for param in model.parameters()]
    for param, _old in previous:
        param.requires_grad_(False)
    layers = get_decoder_layers(model)
    for layer_id in selected_layers:
        mlp = _qwen3_mlp(layers[layer_id])
        for module in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            module.weight.requires_grad_(True)
    return previous


def _restore_requires_grad(previous: list[tuple[nn.Parameter, bool]]) -> None:
    for param, requires_grad in previous:
        param.requires_grad_(requires_grad)


def _zero_target_grads(model: nn.Module, selected_layers: list[int]) -> None:
    layers = get_decoder_layers(model)
    for layer_id in selected_layers:
        mlp = _qwen3_mlp(layers[layer_id])
        for module in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if module.weight.grad is not None:
                module.weight.grad = None


def _channel_saliency_from_grad(mlp: nn.Module) -> torch.Tensor:
    parts = []
    for module, dim in (
        (mlp.gate_proj, 1),
        (mlp.up_proj, 1),
        (mlp.down_proj, 0),
    ):
        if module.weight.grad is None:
            raise RuntimeError("GISP calibration did not produce gradients for a target MLP projection")
        score = (module.weight.detach().float() * module.weight.grad.detach().float()).abs()
        parts.append(score.sum(dim=dim).cpu())
    return parts[0] + parts[1] + parts[2]


def _normalise_scores(scores: dict[int, torch.Tensor], masks: dict[int, torch.Tensor], mode: str) -> dict[int, torch.Tensor]:
    mode = str(mode)
    output: dict[int, torch.Tensor] = {}
    for layer_id, values in scores.items():
        values = values.float().clone()
        active = masks[layer_id].bool()
        active_values = values[active]
        if mode == "none":
            normed = values
        elif mode == "layer_mean":
            denom = active_values.mean().abs().clamp_min(1e-12) if active_values.numel() else torch.tensor(1.0)
            normed = values / denom
        elif mode == "layer_zscore":
            if active_values.numel() <= 1:
                normed = values
            else:
                normed = (values - active_values.mean()) / active_values.std(unbiased=False).clamp_min(1e-12)
        else:
            raise ValueError("GISP score_normalization must be one of {'none', 'layer_mean', 'layer_zscore'}")
        output[layer_id] = normed
    return output


def _update_global_masks(
    masks: dict[int, torch.Tensor],
    scores: dict[int, torch.Tensor],
    *,
    target_ratio: float,
    protected_layers: set[int] | None = None,
) -> None:
    if not 0.0 <= float(target_ratio) < 1.0:
        raise ValueError(f"GISP target_ratio must be in [0, 1), got {target_ratio}")
    total = sum(int(mask.numel()) for mask in masks.values())
    target_pruned = min(total - 1, max(0, int(round(total * float(target_ratio)))))
    current_pruned = sum(int((~mask).sum().item()) for mask in masks.values())
    additional = target_pruned - current_pruned
    if additional <= 0:
        return
    candidates = []
    protected_layers = set(protected_layers or set())
    for layer_id, mask in masks.items():
        if int(layer_id) in protected_layers:
            continue
        active_idx = torch.nonzero(mask.bool(), as_tuple=False).flatten()
        for channel_id in active_idx.tolist():
            candidates.append((float(scores[layer_id][channel_id].item()), int(layer_id), int(channel_id)))
    if not candidates:
        return
    additional = min(additional, len(candidates))
    for _score, layer_id, channel_id in sorted(candidates, key=lambda item: item[0])[:additional]:
        masks[layer_id][channel_id] = False


def _summarize_masks(
    *,
    masks: dict[int, torch.Tensor],
    total_layers: int,
    original_intermediate_size: int,
    ratio: float,
    iterations: int,
    calibration_path: str,
    calibration_samples: int,
    calibration_max_input_tokens: int,
    calibration_prompt_mode: str,
    calibration_text_field: str,
    score_normalization: str,
    matched_rasp_reference: str,
    target_matched_to_rasp_actual_mlp_pruning: float | None,
    prune_skip: bool,
    protected_layers: Iterable[int],
    source: str,
) -> GispMlpSummary:
    kept: dict[str, int] = {}
    pruned: dict[str, int] = {}
    total_pruned = 0
    total_channels = 0
    for layer_id, keep_mask in masks.items():
        kept[str(layer_id)] = int(keep_mask.sum().item())
        pruned[str(layer_id)] = int((~keep_mask).sum().item())
        total_pruned += pruned[str(layer_id)]
        total_channels += int(keep_mask.numel())
    return GispMlpSummary(
        method="gisp_mlp_qwen3",
        source=source,
        baseline_type="gisp_global_iterative_structured_mlp_qwen3_port",
        pruning_granularity="mlp_channel_structured",
        ratio=float(ratio),
        iterations=int(iterations),
        calibration_path=str(calibration_path),
        calibration_samples=int(calibration_samples),
        calibration_max_input_tokens=int(calibration_max_input_tokens),
        calibration_prompt_mode=str(calibration_prompt_mode),
        calibration_text_field=str(calibration_text_field),
        score_normalization=str(score_normalization),
        target="mlp.intermediate_channels",
        physical_pruning=False,
        total_layers=int(total_layers),
        pruned_layers=sorted(int(layer_id) for layer_id in masks),
        original_intermediate_size=int(original_intermediate_size),
        kept_channels_per_layer=kept,
        pruned_channels_per_layer=pruned,
        actual_mlp_channel_pruning_ratio=total_pruned / total_channels if total_channels else 0.0,
        matched_rasp_reference=str(matched_rasp_reference),
        target_matched_to_rasp_actual_mlp_pruning=target_matched_to_rasp_actual_mlp_pruning,
        prune_skip=bool(prune_skip),
        protected_layers=sorted(int(layer_id) for layer_id in protected_layers),
    )


def build_gisp_mlp_mask_artifact_from_scores(
    scores_by_iteration: list[dict[int, torch.Tensor]],
    *,
    ratio: float,
    initial_masks: dict[int, torch.Tensor],
    score_normalization: str = "layer_mean",
    protected_layers: Iterable[int] | None = None,
) -> dict[int, torch.Tensor]:
    masks = {layer_id: mask.detach().cpu().bool().clone() for layer_id, mask in initial_masks.items()}
    iterations = max(1, len(scores_by_iteration))
    protected = {int(layer_id) for layer_id in protected_layers or []}
    for step, scores in enumerate(scores_by_iteration):
        normed = _normalise_scores(scores, masks, score_normalization)
        _update_global_masks(
            masks,
            normed,
            target_ratio=float(ratio) * float(step + 1) / float(iterations),
            protected_layers=protected,
        )
    return masks


def _collect_gradient_scores(
    model: nn.Module,
    tokenizer,
    calibration_rows: list[dict[str, Any]],
    *,
    selected_layers: list[int],
    masks: dict[int, torch.Tensor],
    prompt_config: dict[str, Any],
    calibration_samples: int,
    max_input_tokens: int,
    calibration_prompt_mode: str,
    calibration_text_field: str,
) -> dict[int, torch.Tensor]:
    layers = get_decoder_layers(model)
    scores = {
        layer_id: torch.zeros(_intermediate_size(_qwen3_mlp(layers[layer_id])), dtype=torch.float64)
        for layer_id in selected_layers
    }
    handles = _install_logical_mlp_masks(model, masks)
    use_cache = getattr(model.config, "use_cache", None)
    previous_requires_grad = _set_only_target_weights_trainable(model, selected_layers)
    was_training = bool(model.training)
    model.eval()
    if use_cache is not None:
        model.config.use_cache = False
    try:
        for row in tqdm(calibration_rows[:calibration_samples], desc="gisp-mlp-gradient-calibration"):
            _zero_target_grads(model, selected_layers)
            prompt = _resolve_prompt(
                row,
                tokenizer,
                prompt_config,
                calibration_prompt_mode,
                calibration_text_field,
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
            inputs = inputs.to(model_device(model))
            labels = inputs["input_ids"].clone()
            with torch.enable_grad():
                outputs = model(**inputs, labels=labels, use_cache=False, return_dict=True)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("GISP calibration expected model outputs to include a loss")
                loss.backward()
            for layer_id in selected_layers:
                scores[layer_id] += _channel_saliency_from_grad(_qwen3_mlp(layers[layer_id])).double()
        _zero_target_grads(model, selected_layers)
    finally:
        if use_cache is not None:
            model.config.use_cache = use_cache
        if was_training:
            model.train()
        _restore_requires_grad(previous_requires_grad)
        for handle in handles:
            handle.remove()
    return {layer_id: value.float() for layer_id, value in scores.items()}


def build_gisp_mlp_mask_artifact_qwen3(
    model: nn.Module,
    tokenizer,
    *,
    calibration_path: str,
    prompt_config: dict[str, Any],
    ratio: float,
    iterations: int = 4,
    calibration_samples: int = 64,
    max_input_tokens: int = 1024,
    calibration_prompt_mode: str = "structured_prompt",
    calibration_text_field: str = "text",
    score_normalization: str = "layer_mean",
    layers: Iterable[int] | None = None,
    matched_rasp_reference: str = "",
    target_matched_to_rasp_actual_mlp_pruning: float | None = None,
    prune_skip: bool = True,
) -> dict[str, Any]:
    calibration_rows = read_jsonl(calibration_path)
    sample_count = min(int(calibration_samples), len(calibration_rows))
    if sample_count <= 0:
        raise ValueError(f"No GISP calibration rows found at {calibration_path}")
    resolved_prompt_mode = str(calibration_prompt_mode)
    if (
        resolved_prompt_mode in {"structured_prompt", "prompt"}
        and calibration_rows
        and "question" not in calibration_rows[0]
        and calibration_text_field in calibration_rows[0]
    ):
        resolved_prompt_mode = "raw_text"
    selected_layers = _layer_ids(model, layers)
    if not selected_layers:
        raise ValueError("GISP MLP pruning requires at least one selected decoder layer")
    decoder_layers = get_decoder_layers(model)
    protected_layers = _protected_layer_ids(
        selected_layers,
        len(decoder_layers),
        prune_skip=bool(prune_skip),
    )
    first_mlp = _qwen3_mlp(decoder_layers[selected_layers[0]])
    original_intermediate_size = _intermediate_size(first_mlp)
    masks = {
        layer_id: torch.ones(_intermediate_size(_qwen3_mlp(decoder_layers[layer_id])), dtype=torch.bool)
        for layer_id in selected_layers
    }
    iterations = max(1, int(iterations))
    for step in range(iterations):
        scores = _collect_gradient_scores(
            model,
            tokenizer,
            calibration_rows,
            selected_layers=selected_layers,
            masks=masks,
            prompt_config=prompt_config,
            calibration_samples=sample_count,
            max_input_tokens=int(max_input_tokens),
            calibration_prompt_mode=resolved_prompt_mode,
            calibration_text_field=str(calibration_text_field),
        )
        normed = _normalise_scores(scores, masks, str(score_normalization))
        _update_global_masks(
            masks,
            normed,
            target_ratio=float(ratio) * float(step + 1) / float(iterations),
            protected_layers=protected_layers,
        )

    summary = _summarize_masks(
        masks=masks,
        total_layers=len(decoder_layers),
        original_intermediate_size=original_intermediate_size,
        ratio=float(ratio),
        iterations=iterations,
        calibration_path=str(calibration_path),
        calibration_samples=sample_count,
        calibration_max_input_tokens=int(max_input_tokens),
        calibration_prompt_mode=resolved_prompt_mode,
        calibration_text_field=str(calibration_text_field),
        score_normalization=str(score_normalization),
        matched_rasp_reference=str(matched_rasp_reference),
        target_matched_to_rasp_actual_mlp_pruning=target_matched_to_rasp_actual_mlp_pruning,
        prune_skip=bool(prune_skip),
        protected_layers=protected_layers,
        source=(
            "GISP-style global iterative structured MLP-channel pruning for Qwen3; "
            "first-order |W * grad(W)| saliency aggregated over gate/up/down projections."
        ),
    )
    return {
        "schema": "gisp_mlp_qwen3_mask_artifact_v1",
        "summary": summary_to_dict(summary),
        "keep_indices_by_layer": {
            str(layer_id): torch.where(mask)[0].cpu().tolist()
            for layer_id, mask in masks.items()
        },
        "pruned_indices_by_layer": {
            str(layer_id): torch.where(~mask)[0].cpu().tolist()
            for layer_id, mask in masks.items()
        },
    }


def apply_gisp_mlp_artifact_qwen3(
    model: nn.Module,
    artifact: dict | str,
) -> tuple[GispMlpSummary, list[Any]]:
    if isinstance(artifact, str):
        artifact = read_json(artifact)
    if artifact.get("schema") != "gisp_mlp_qwen3_mask_artifact_v1":
        raise ValueError(f"Unsupported GISP artifact schema: {artifact.get('schema')}")
    summary = GispMlpSummary(**dict(artifact["summary"]))
    masks: dict[int, torch.Tensor] = {}
    layers = get_decoder_layers(model)
    for layer_key, keep_indices in artifact["keep_indices_by_layer"].items():
        layer_id = int(layer_key)
        if layer_id < 0 or layer_id >= len(layers):
            raise ValueError(f"GISP artifact layer {layer_id} is out of range for {len(layers)} decoder layers")
        actual_intermediate_size = _intermediate_size(_qwen3_mlp(layers[layer_id]))
        if actual_intermediate_size != int(summary.original_intermediate_size):
            raise ValueError(
                "GISP artifact intermediate size mismatch: "
                f"artifact={summary.original_intermediate_size}, model layer {layer_id}={actual_intermediate_size}"
            )
        keep_mask = torch.zeros(summary.original_intermediate_size, dtype=torch.bool)
        keep_tensor = torch.tensor([int(i) for i in keep_indices], dtype=torch.long)
        if keep_tensor.numel() and (
            int(keep_tensor.min().item()) < 0
            or int(keep_tensor.max().item()) >= int(summary.original_intermediate_size)
        ):
            raise ValueError(f"GISP artifact keep indices out of bounds for layer {layer_id}")
        keep_mask[keep_tensor] = True
        masks[layer_id] = keep_mask
    handles = _install_logical_mlp_masks(model, masks)
    return summary, handles


def prepare_gisp_mlp_qwen3(
    model: nn.Module,
    tokenizer,
    *,
    calibration_path: str,
    prompt_config: dict[str, Any],
    ratio: float,
    iterations: int = 4,
    calibration_samples: int = 64,
    max_input_tokens: int = 1024,
    calibration_prompt_mode: str = "structured_prompt",
    calibration_text_field: str = "text",
    score_normalization: str = "layer_mean",
    layers: Iterable[int] | None = None,
    precomputed_masks_path: str | None = None,
    matched_rasp_reference: str = "",
    target_matched_to_rasp_actual_mlp_pruning: float | None = None,
    prune_skip: bool = True,
) -> tuple[GispMlpSummary, list[Any]]:
    if precomputed_masks_path:
        return apply_gisp_mlp_artifact_qwen3(model, precomputed_masks_path)
    artifact = build_gisp_mlp_mask_artifact_qwen3(
        model,
        tokenizer,
        calibration_path=calibration_path,
        prompt_config=prompt_config,
        ratio=float(ratio),
        iterations=int(iterations),
        calibration_samples=int(calibration_samples),
        max_input_tokens=int(max_input_tokens),
        calibration_prompt_mode=str(calibration_prompt_mode),
        calibration_text_field=str(calibration_text_field),
        score_normalization=str(score_normalization),
        layers=layers,
        matched_rasp_reference=str(matched_rasp_reference),
        target_matched_to_rasp_actual_mlp_pruning=target_matched_to_rasp_actual_mlp_pruning,
        prune_skip=bool(prune_skip),
    )
    return apply_gisp_mlp_artifact_qwen3(model, artifact)


def summary_to_dict(summary: GispMlpSummary) -> dict[str, Any]:
    return asdict(summary)
