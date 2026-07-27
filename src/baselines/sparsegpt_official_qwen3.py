from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from tqdm import tqdm

from src.data.format_prompt import build_prompt
from src.models.hooks import get_decoder_layers, model_device
from src.utils.io import ensure_dir, read_json, read_jsonl, write_json


DEFAULT_QWEN3_SPARSEGPT_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass
class SparseGPTArtifactSummary:
    method: str
    source: str
    baseline_type: str
    pruning_granularity: str
    sparsity_ratio: float
    calibration_path: str
    calibration_samples: int
    calibration_max_input_tokens: int
    calibration_batch_size: int
    target_modules: list[str]
    blocksize: int
    percdamp: float
    total_pruned_weights: int
    total_target_weights: int
    weight_sparsity_overall: float
    weight_sparsity_by_module: dict[str, float]
    pruning_method: str
    artifact_path: str
    artifact_hash: str
    artifact_save_dtype: str
    real_speedup_claimed: bool
    matched_rasp_reference: str
    target_matched_to_rasp_actual_mlp_pruning: float | None


class SparseGPTInputStats:
    def __init__(self, module: nn.Linear) -> None:
        self.columns = int(module.weight.shape[1])
        self.hessian = torch.zeros((self.columns, self.columns), dtype=torch.float32, device="cpu")
        self.nsamples = 0

    def add_batch(self, inputs: torch.Tensor) -> None:
        values = inputs.detach()
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3:
            raise ValueError(f"Expected Linear input with shape [batch, tokens, hidden], got {tuple(values.shape)}")
        flat = values.reshape(-1, values.shape[-1]).float()
        self.hessian += flat.t().matmul(flat).cpu()
        self.nsamples += int(flat.shape[0])

    def normalized_hessian(self) -> torch.Tensor:
        if self.nsamples <= 0:
            return torch.eye(self.columns, dtype=torch.float32)
        return self.hessian / float(self.nsamples)


def _resolve_module(root: nn.Module, dotted: str) -> nn.Module | None:
    current: nn.Module | None = root
    for part in dotted.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def qwen3_sparsegpt_linear_modules_for_layer(
    layer: nn.Module,
    layer_id: int,
    target_modules: Iterable[str] = DEFAULT_QWEN3_SPARSEGPT_TARGETS,
) -> dict[str, nn.Linear]:
    modules: dict[str, nn.Linear] = {}
    for target in target_modules:
        module = _resolve_module(layer, target)
        if module is None:
            continue
        if not isinstance(module, nn.Linear):
            raise ValueError(f"Qwen3 SparseGPT target layer {layer_id}.{target} is not nn.Linear")
        modules[f"layers.{layer_id}.{target}"] = module
    return modules


def qwen3_sparsegpt_linear_modules(
    model: nn.Module,
    target_modules: Iterable[str] = DEFAULT_QWEN3_SPARSEGPT_TARGETS,
) -> dict[str, nn.Linear]:
    modules: dict[str, nn.Linear] = {}
    for layer_id, layer in enumerate(get_decoder_layers(model)):
        modules.update(qwen3_sparsegpt_linear_modules_for_layer(layer, layer_id, target_modules))
    if not modules:
        raise ValueError("No Qwen3 SparseGPT target Linear modules found")
    return modules


def _batched(rows: list[dict[str, Any]], batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


@torch.no_grad()
def collect_sparsegpt_input_stats_for_layer_qwen3(
    model: nn.Module,
    tokenizer,
    calibration_rows: list[dict[str, Any]],
    *,
    layer_id: int,
    prompt_config: dict[str, Any],
    calibration_samples: int,
    max_input_tokens: int,
    calibration_batch_size: int,
    target_modules: Iterable[str] = DEFAULT_QWEN3_SPARSEGPT_TARGETS,
) -> dict[str, SparseGPTInputStats]:
    layers = get_decoder_layers(model)
    modules = qwen3_sparsegpt_linear_modules_for_layer(layers[layer_id], layer_id, target_modules)
    stats = {name: SparseGPTInputStats(module) for name, module in modules.items()}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs, _output) -> None:
            stats[name].add_batch(inputs[0])

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    rows = calibration_rows[: int(calibration_samples)]
    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False
    try:
        for batch in tqdm(rows and list(_batched(rows, calibration_batch_size)) or [], desc=f"sparsegpt-calibration-layer-{layer_id}"):
            prompts = [build_prompt(str(row["question"]), tokenizer, prompt_config) for row in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_input_tokens),
            )
            inputs = inputs.to(model_device(model))
            model(**inputs, use_cache=False, return_dict=True)
    finally:
        if use_cache is not None:
            model.config.use_cache = use_cache
        for handle in handles:
            handle.remove()
    return stats


def _stable_cholesky_inverse_factor(hessian: torch.Tensor, *, percdamp: float) -> torch.Tensor:
    hessian = hessian.float()
    diagonal = torch.diag(hessian)
    mean_diag = torch.mean(diagonal[diagonal > 0]) if torch.any(diagonal > 0) else torch.tensor(1.0)
    damp = float(percdamp) * float(mean_diag.item())
    diag_idx = torch.arange(hessian.shape[0], device=hessian.device)
    hessian = hessian.clone()
    hessian[diag_idx, diag_idx] += max(damp, 1e-6)
    for attempt in range(6):
        try:
            chol = torch.linalg.cholesky(hessian)
            inv = torch.cholesky_inverse(chol)
            return torch.linalg.cholesky(inv, upper=True)
        except RuntimeError:
            hessian[diag_idx, diag_idx] += (10**attempt) * max(damp, 1e-5)
    chol = torch.linalg.cholesky(hessian)
    inv = torch.cholesky_inverse(chol)
    return torch.linalg.cholesky(inv, upper=True)


@torch.no_grad()
def sparsegpt_prune_linear_unstructured(
    module: nn.Linear,
    stats: SparseGPTInputStats,
    *,
    sparsity_ratio: float,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> float:
    if not 0.0 <= float(sparsity_ratio) < 1.0:
        raise ValueError(f"SparseGPT sparsity_ratio must be in [0, 1), got {sparsity_ratio}")
    weight = module.weight.data
    original_dtype = weight.dtype
    weight_float = weight.detach().float().clone()
    rows, columns = weight_float.shape
    hessian = stats.normalized_hessian()
    if hessian.shape != (columns, columns):
        raise ValueError(f"Hessian shape {tuple(hessian.shape)} does not match weight columns {columns}")

    dead = torch.diag(hessian) == 0
    if torch.any(dead):
        hessian[dead, dead] = 1.0
        weight_float[:, dead] = 0.0

    h_inv = _stable_cholesky_inverse_factor(hessian, percdamp=float(percdamp)).to(weight_float.device)
    blocksize = max(1, int(blocksize))

    for start in range(0, columns, blocksize):
        end = min(start + blocksize, columns)
        count = end - start
        block = weight_float[:, start:end].clone()
        quantized = torch.zeros_like(block)
        errors = torch.zeros_like(block)
        h_inv_block = h_inv[start:end, start:end]
        saliency = block.square() / (torch.diag(h_inv_block).reshape(1, -1).square() + 1e-12)
        prune_count = int(round(float(sparsity_ratio) * saliency.numel()))
        prune_count = min(max(prune_count, 0), saliency.numel() - 1)
        if prune_count > 0:
            threshold = torch.topk(saliency.flatten(), k=prune_count, largest=False).values.max()
            mask = saliency <= threshold
        else:
            mask = torch.zeros_like(saliency, dtype=torch.bool)

        for offset in range(count):
            w = block[:, offset]
            d = h_inv_block[offset, offset].clamp_min(1e-12)
            q = w.clone()
            q[mask[:, offset]] = 0.0
            quantized[:, offset] = q
            err = (w - q) / d
            if offset + 1 < count:
                block[:, offset + 1 :] -= err.unsqueeze(1).matmul(
                    h_inv_block[offset, offset + 1 :].unsqueeze(0)
                )
            errors[:, offset] = err

        weight_float[:, start:end] = quantized
        if end < columns:
            weight_float[:, end:] -= errors.matmul(h_inv[start:end, end:])

    weight.copy_(weight_float.to(device=weight.device, dtype=original_dtype))
    return float((weight == 0).sum().item() / weight.numel())


def _artifact_hash(artifact_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(artifact_dir.glob("weights/*.pt")):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _save_dtype(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"float16", "fp16", "half"}:
        return torch.float16
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if lowered in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported SparseGPT artifact_save_dtype: {name}")


@torch.no_grad()
def prepare_sparsegpt_official_qwen3_artifact(
    model: nn.Module,
    tokenizer,
    *,
    artifact_dir: str | Path,
    calibration_path: str,
    prompt_config: dict[str, Any],
    sparsity_ratio: float,
    calibration_samples: int = 128,
    calibration_max_input_tokens: int = 2048,
    calibration_batch_size: int = 1,
    target_modules: Iterable[str] | None = DEFAULT_QWEN3_SPARSEGPT_TARGETS,
    blocksize: int = 128,
    percdamp: float = 0.01,
    artifact_save_dtype: str = "float16",
    matched_rasp_reference: str = "",
    target_matched_to_rasp_actual_mlp_pruning: float | None = None,
    force: bool = False,
) -> SparseGPTArtifactSummary:
    artifact_dir = Path(artifact_dir)
    summary_path = artifact_dir / "summary.json"
    weights_dir = artifact_dir / "weights"
    if summary_path.exists() and not force:
        return SparseGPTArtifactSummary(**read_json(summary_path))

    ensure_dir(weights_dir)
    calibration_rows = read_jsonl(calibration_path)
    sample_count = min(int(calibration_samples), len(calibration_rows))
    if sample_count <= 0:
        raise ValueError(f"No SparseGPT calibration rows found at {calibration_path}")
    target_modules = list(target_modules or DEFAULT_QWEN3_SPARSEGPT_TARGETS)
    save_dtype = _save_dtype(artifact_save_dtype)
    layers = get_decoder_layers(model)
    weight_sparsity_by_module: dict[str, float] = {}
    total_pruned = 0
    total_weights = 0

    for layer_id, layer in enumerate(tqdm(layers, desc="sparsegpt-prepare-layers")):
        stats = collect_sparsegpt_input_stats_for_layer_qwen3(
            model,
            tokenizer,
            calibration_rows,
            layer_id=layer_id,
            prompt_config=prompt_config,
            calibration_samples=sample_count,
            max_input_tokens=int(calibration_max_input_tokens),
            calibration_batch_size=int(calibration_batch_size),
            target_modules=target_modules,
        )
        modules = qwen3_sparsegpt_linear_modules_for_layer(layer, layer_id, target_modules)
        layer_weights = {}
        for name, module in modules.items():
            sparsity = sparsegpt_prune_linear_unstructured(
                module,
                stats[name],
                sparsity_ratio=float(sparsity_ratio),
                blocksize=int(blocksize),
                percdamp=float(percdamp),
            )
            weight_sparsity_by_module[name] = sparsity
            total_pruned += int((module.weight.data == 0).sum().item())
            total_weights += int(module.weight.data.numel())
            layer_weights[name] = module.weight.detach().cpu().to(save_dtype)
        torch.save(layer_weights, weights_dir / f"layer_{layer_id:05d}.pt")
        del stats

    artifact_hash = _artifact_hash(artifact_dir)
    summary = SparseGPTArtifactSummary(
        method="sparsegpt_official_qwen3",
        source="Official-style SparseGPT adaptation for Qwen3; Hessian/OBS one-shot unstructured pruning with error compensation.",
        baseline_type="official_style_sparsegpt",
        pruning_granularity="weight_unstructured",
        sparsity_ratio=float(sparsity_ratio),
        calibration_path=str(calibration_path),
        calibration_samples=sample_count,
        calibration_max_input_tokens=int(calibration_max_input_tokens),
        calibration_batch_size=int(calibration_batch_size),
        target_modules=target_modules,
        blocksize=int(blocksize),
        percdamp=float(percdamp),
        total_pruned_weights=total_pruned,
        total_target_weights=total_weights,
        weight_sparsity_overall=total_pruned / total_weights if total_weights else 0.0,
        weight_sparsity_by_module=weight_sparsity_by_module,
        pruning_method="sparsegpt_unstructured_blockwise_obs_zero_out",
        artifact_path=str(artifact_dir),
        artifact_hash=artifact_hash,
        artifact_save_dtype=str(artifact_save_dtype),
        real_speedup_claimed=False,
        matched_rasp_reference=str(matched_rasp_reference),
        target_matched_to_rasp_actual_mlp_pruning=target_matched_to_rasp_actual_mlp_pruning,
    )
    write_json(summary_path, asdict(summary))
    return summary


@torch.no_grad()
def apply_sparsegpt_official_qwen3_artifact(
    model: nn.Module,
    *,
    artifact_dir: str | Path,
) -> SparseGPTArtifactSummary:
    artifact_dir = Path(artifact_dir)
    summary = SparseGPTArtifactSummary(**read_json(artifact_dir / "summary.json"))
    modules = qwen3_sparsegpt_linear_modules(model, summary.target_modules)
    for path in sorted((artifact_dir / "weights").glob("layer_*.pt")):
        tensors = torch.load(path, map_location="cpu")
        for name, tensor in tensors.items():
            if name not in modules:
                raise ValueError(f"SparseGPT artifact module {name} is not present in model")
            modules[name].weight.data.copy_(tensor.to(device=modules[name].weight.device, dtype=modules[name].weight.dtype))
    return summary


def summary_to_dict(summary: SparseGPTArtifactSummary) -> dict[str, Any]:
    return asdict(summary)
