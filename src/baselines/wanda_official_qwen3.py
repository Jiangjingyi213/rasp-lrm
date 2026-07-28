from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import nn
from tqdm import tqdm

from src.data.format_prompt import build_prompt
from src.models.hooks import get_decoder_layers, model_device
from src.utils.io import read_jsonl


DEFAULT_QWEN3_WANDA_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass
class WandaOfficialSummary:
    method: str
    source: str
    baseline_type: str
    pruning_granularity: str
    sparsity_ratio: float
    calibration_path: str
    calibration_samples: int
    calibration_max_input_tokens: int
    calibration_prompt_mode: str
    calibration_text_field: str
    calibration_seed: int | None
    target_modules: list[str]
    total_pruned_weights: int
    total_target_weights: int
    weight_sparsity_overall: float
    weight_sparsity_by_module: dict[str, float]
    pruning_method: str
    real_speedup_claimed: bool
    matched_rasp_reference: str
    target_matched_to_rasp_actual_mlp_pruning: float | None


class WandaInputStats:
    def __init__(self, module: nn.Linear) -> None:
        self.in_features = int(module.weight.shape[1])
        self.scaler_row = torch.zeros(self.in_features, dtype=torch.float64, device="cpu")
        self.nsamples = 0

    def add_batch(self, inputs: torch.Tensor) -> None:
        values = inputs.detach()
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3:
            raise ValueError(f"Expected Linear input with shape [batch, tokens, hidden], got {tuple(values.shape)}")
        flat = values.reshape(-1, values.shape[-1]).to(device="cpu", dtype=torch.float32)
        self.scaler_row += flat.pow(2).sum(dim=0).double()
        self.nsamples += int(flat.shape[0])

    def activation_scale(self) -> torch.Tensor:
        if self.nsamples <= 0:
            return torch.ones(self.in_features, dtype=torch.float32)
        return (self.scaler_row / float(self.nsamples)).float().sqrt()


def _resolve_module(root: nn.Module, dotted: str) -> nn.Module | None:
    current: nn.Module | None = root
    for part in dotted.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def qwen3_wanda_linear_modules(
    model: nn.Module,
    target_modules: Iterable[str] = DEFAULT_QWEN3_WANDA_TARGETS,
) -> dict[str, nn.Linear]:
    modules: dict[str, nn.Linear] = {}
    layers = get_decoder_layers(model)
    for layer_id, layer in enumerate(layers):
        for target in target_modules:
            module = _resolve_module(layer, target)
            if module is None:
                continue
            if not isinstance(module, nn.Linear):
                raise ValueError(f"Qwen3 Wanda target layer {layer_id}.{target} is not nn.Linear")
            modules[f"layers.{layer_id}.{target}"] = module
    if not modules:
        raise ValueError("No Qwen3 Wanda target Linear modules found")
    return modules


def assert_wanda_weights_materialized(modules: dict[str, nn.Linear]) -> None:
    meta_modules = [name for name, module in modules.items() if module.weight.is_meta]
    if meta_modules:
        preview = ", ".join(meta_modules[:5])
        raise RuntimeError(
            "Wanda requires materialized Linear weights, but some Qwen3 target weights "
            f"are still on the meta device: {preview}. Set `model.device_map: null` "
            "for Wanda configs so the model is loaded onto the current CUDA device "
            "before Wanda applies weight masks."
        )


def _wanda_calibration_forward(model: nn.Module, inputs: dict[str, torch.Tensor]) -> None:
    """Run only the transformer backbone; Wanda hooks do not need lm_head logits."""
    backbone = getattr(model, "model", None)
    if backbone is not None and backbone is not model:
        kwargs = {
            key: value
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        backbone(**kwargs, use_cache=False, return_dict=True)
        return
    try:
        model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
    except TypeError:
        model(**inputs, use_cache=False, return_dict=True)


@torch.no_grad()
def collect_wanda_input_stats_qwen3(
    model: nn.Module,
    tokenizer,
    calibration_rows: list[dict[str, Any]],
    *,
    prompt_config: dict[str, Any],
    calibration_samples: int,
    max_input_tokens: int,
    calibration_prompt_mode: str = "structured_prompt",
    calibration_text_field: str = "text",
    target_modules: Iterable[str] = DEFAULT_QWEN3_WANDA_TARGETS,
) -> dict[str, WandaInputStats]:
    modules = qwen3_wanda_linear_modules(model, target_modules)
    assert_wanda_weights_materialized(modules)
    stats = {name: WandaInputStats(module) for name, module in modules.items()}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs, _output) -> None:
            stats[name].add_batch(inputs[0])

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False
    try:
        for row in tqdm(calibration_rows[:calibration_samples], desc="wanda-official-calibration"):
            if calibration_prompt_mode == "raw_text":
                if calibration_text_field not in row:
                    raise KeyError(
                        f"Wanda raw_text calibration expected field "
                        f"{calibration_text_field!r}; available fields={sorted(row)}"
                    )
                prompt = str(row[calibration_text_field])
            elif calibration_prompt_mode in ("structured_prompt", "prompt"):
                prompt = build_prompt(str(row["question"]), tokenizer, prompt_config)
            else:
                raise ValueError(
                    "Unsupported Wanda calibration_prompt_mode="
                    f"{calibration_prompt_mode!r}; expected raw_text or structured_prompt"
                )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
            inputs = inputs.to(model_device(model))
            _wanda_calibration_forward(model, inputs)
    finally:
        if use_cache is not None:
            model.config.use_cache = use_cache
        for handle in handles:
            handle.remove()
    return stats


@torch.no_grad()
def apply_wanda_unstructured_masks_qwen3(
    model: nn.Module,
    stats: dict[str, WandaInputStats],
    *,
    sparsity_ratio: float,
    target_modules: Iterable[str] = DEFAULT_QWEN3_WANDA_TARGETS,
) -> tuple[dict[str, float], int, int]:
    if not 0.0 <= float(sparsity_ratio) < 1.0:
        raise ValueError(f"Wanda sparsity_ratio must be in [0, 1), got {sparsity_ratio}")
    modules = qwen3_wanda_linear_modules(model, target_modules)
    assert_wanda_weights_materialized(modules)
    missing = sorted(set(modules) - set(stats))
    if missing:
        raise ValueError(f"Missing Wanda activation stats for modules: {missing[:5]}")

    sparsity_by_module: dict[str, float] = {}
    total_pruned = 0
    total_weights = 0
    for name, module in modules.items():
        weight = module.weight.data
        scale = stats[name].activation_scale().to(device=weight.device, dtype=torch.float32)
        metric = weight.detach().float().abs() * scale.reshape(1, -1)
        prune_per_row = int(round(metric.shape[1] * float(sparsity_ratio)))
        prune_per_row = min(max(0, prune_per_row), metric.shape[1] - 1)
        if prune_per_row > 0:
            indices = torch.topk(metric, k=prune_per_row, dim=1, largest=False).indices
            mask = torch.zeros_like(metric, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            weight[mask] = 0
        pruned = int((weight == 0).sum().item())
        total = int(weight.numel())
        sparsity_by_module[name] = pruned / total if total else 0.0
        total_pruned += pruned
        total_weights += total
    overall = total_pruned / total_weights if total_weights else 0.0
    return sparsity_by_module, total_pruned, total_weights


@torch.no_grad()
def apply_wanda_official_qwen3(
    model: nn.Module,
    tokenizer,
    *,
    calibration_path: str,
    prompt_config: dict[str, Any],
    sparsity_ratio: float,
    calibration_samples: int = 128,
    max_input_tokens: int = 2048,
    calibration_prompt_mode: str = "structured_prompt",
    calibration_text_field: str = "text",
    calibration_seed: int | None = None,
    target_modules: Iterable[str] | None = DEFAULT_QWEN3_WANDA_TARGETS,
    matched_rasp_reference: str = "",
    target_matched_to_rasp_actual_mlp_pruning: float | None = None,
) -> WandaOfficialSummary:
    calibration_rows = read_jsonl(calibration_path)
    target_modules = list(target_modules or DEFAULT_QWEN3_WANDA_TARGETS)
    sample_count = min(int(calibration_samples), len(calibration_rows))
    if sample_count <= 0:
        raise ValueError(f"No Wanda calibration rows found at {calibration_path}")
    stats = collect_wanda_input_stats_qwen3(
        model,
        tokenizer,
        calibration_rows,
        prompt_config=prompt_config,
        calibration_samples=sample_count,
        max_input_tokens=int(max_input_tokens),
        calibration_prompt_mode=str(calibration_prompt_mode),
        calibration_text_field=str(calibration_text_field),
        target_modules=target_modules,
    )
    sparsity_by_module, total_pruned, total_weights = apply_wanda_unstructured_masks_qwen3(
        model,
        stats,
        sparsity_ratio=float(sparsity_ratio),
        target_modules=target_modules,
    )
    return WandaOfficialSummary(
        method="wanda_official_qwen3",
        source="Official-style Wanda adaptation for Qwen3; saliency abs(W) * sqrt(input activation scaler), unstructured per-output-row weight pruning.",
        baseline_type="official_style_wanda",
        pruning_granularity="weight_unstructured",
        sparsity_ratio=float(sparsity_ratio),
        calibration_path=str(calibration_path),
        calibration_samples=sample_count,
        calibration_max_input_tokens=int(max_input_tokens),
        calibration_prompt_mode=str(calibration_prompt_mode),
        calibration_text_field=str(calibration_text_field),
        calibration_seed=calibration_seed,
        target_modules=target_modules,
        total_pruned_weights=total_pruned,
        total_target_weights=total_weights,
        weight_sparsity_overall=total_pruned / total_weights if total_weights else 0.0,
        weight_sparsity_by_module=sparsity_by_module,
        pruning_method="unstructured_per_output_row_zero_out",
        real_speedup_claimed=False,
        matched_rasp_reference=str(matched_rasp_reference),
        target_matched_to_rasp_actual_mlp_pruning=target_matched_to_rasp_actual_mlp_pruning,
    )


def summary_to_dict(summary: WandaOfficialSummary) -> dict[str, Any]:
    return asdict(summary)
