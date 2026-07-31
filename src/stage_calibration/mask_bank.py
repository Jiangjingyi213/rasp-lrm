from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .artifacts import assert_metadata_matches
from .protocol import STAGES
from .statistics import keep_mask, stage_balanced_metric


def ratio_key(ratio: float) -> str:
    return f"{float(ratio):.4f}"


def _tensor_hash(value: torch.Tensor) -> str:
    """Return a deterministic fingerprint without serialising an entire bank."""

    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _policy_hash(policy: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for stage in STAGES:
        for layer_id in sorted(policy[stage]):
            digest.update(stage.encode("ascii"))
            digest.update(str(layer_id).encode("ascii"))
            digest.update(_tensor_hash(policy[stage][layer_id]["metric"]).encode("ascii"))
            digest.update(_tensor_hash(policy[stage][layer_id]["mean"]).encode("ascii"))
    return digest.hexdigest()


def _al_am_masks(layer_metrics: dict[int, torch.Tensor], ratios: list[float]) -> dict[int, dict[str, torch.Tensor]]:
    layer_ids = sorted(layer_metrics)
    stacked = torch.stack([layer_metrics[layer_id].float().cpu() for layer_id in layer_ids])
    standardized = (stacked - stacked.mean(dim=1, keepdim=True)) / stacked.std(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    output = {layer_id: {} for layer_id in layer_ids}
    for ratio in ratios:
        prune = min(
            standardized.numel() - 1,
            max(0, int(round(standardized.numel() * float(ratio)))),
        )
        flat = torch.ones(standardized.numel(), dtype=torch.bool)
        if prune:
            indices = torch.topk(standardized.reshape(-1), k=prune, largest=False).indices
            flat[indices] = False
        matrix = flat.reshape_as(standardized)
        for index, layer_id in enumerate(layer_ids):
            output[layer_id][ratio_key(ratio)] = matrix[index].clone()
    return output


def build_mask_bank(
    *,
    metadata: dict[str, Any],
    metrics: dict[str, dict[int, torch.Tensor]],
    means: dict[str, dict[int, torch.Tensor]],
    ratios: list[float],
    output_norms: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    required = {"c4", "prompt_only", "trajectory", *STAGES}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Missing calibration metrics: {sorted(missing)}")
    layers = sorted(metrics["trajectory"])
    if output_norms is not None and set(output_norms) != set(layers):
        raise ValueError("Output norms must cover exactly the calibrated layers")
    policies: dict[str, Any] = {}

    def global_policy(source: str) -> dict[str, Any]:
        return {
            stage: {
                layer_id: {
                    "metric": metrics[source][layer_id].float().cpu(),
                    "mean": means[source][layer_id].float().cpu(),
                    **(
                        {"output_norm": output_norms[layer_id].float().cpu()}
                        if output_norms is not None
                        else {}
                    ),
                    "masks": {
                        ratio_key(ratio): keep_mask(metrics[source][layer_id], ratio)
                        for ratio in ratios
                    },
                }
                for layer_id in layers
            }
            for stage in STAGES
        }

    policies["c4_global"] = global_policy("c4")
    policies["prompt_only_global"] = global_policy("prompt_only")
    policies["trajectory_global"] = global_policy("trajectory")
    balanced_metrics = {
        layer_id: stage_balanced_metric(
            {stage: metrics[stage][layer_id] for stage in STAGES}
        )
        for layer_id in layers
    }
    balanced_means = {
        layer_id: torch.stack([means[stage][layer_id].float() for stage in STAGES]).mean(dim=0)
        for layer_id in layers
    }
    policies["stage_balanced_global"] = {
        stage: {
            layer_id: {
                "metric": balanced_metrics[layer_id].cpu(),
                "mean": balanced_means[layer_id].cpu(),
                **(
                    {"output_norm": output_norms[layer_id].float().cpu()}
                    if output_norms is not None
                    else {}
                ),
                "masks": {
                    ratio_key(ratio): keep_mask(balanced_metrics[layer_id], ratio)
                    for ratio in ratios
                },
            }
            for layer_id in layers
        }
        for stage in STAGES
    }
    policies["stage_specific"] = {
        stage: {
            layer_id: {
                "metric": metrics[stage][layer_id].float().cpu(),
                "mean": means[stage][layer_id].float().cpu(),
                **(
                    {"output_norm": output_norms[layer_id].float().cpu()}
                    if output_norms is not None
                    else {}
                ),
                "masks": {
                    ratio_key(ratio): keep_mask(metrics[stage][layer_id], ratio)
                    for ratio in ratios
                },
            }
            for layer_id in layers
        }
        for stage in STAGES
    }
    shuffled = dict(zip(STAGES, (*STAGES[1:], STAGES[0])))
    policies["shuffled_stage"] = {
        stage: policies["stage_specific"][shuffled[stage]] for stage in STAGES
    }
    trajectory_al_am = _al_am_masks(metrics["trajectory"], ratios)
    policies["trajectory_global_al_am"] = {
        stage: {
            layer_id: {
                "metric": metrics["trajectory"][layer_id].float().cpu(),
                "mean": means["trajectory"][layer_id].float().cpu(),
                **(
                    {"output_norm": output_norms[layer_id].float().cpu()}
                    if output_norms is not None
                    else {}
                ),
                "masks": trajectory_al_am[layer_id],
            }
            for layer_id in layers
        }
        for stage in STAGES
    }
    policies["stage_specific_al_am"] = {}
    for stage in STAGES:
        stage_masks = _al_am_masks(metrics[stage], ratios)
        policies["stage_specific_al_am"][stage] = {
            layer_id: {
                "metric": metrics[stage][layer_id].float().cpu(),
                "mean": means[stage][layer_id].float().cpu(),
                **(
                    {"output_norm": output_norms[layer_id].float().cpu()}
                    if output_norms is not None
                    else {}
                ),
                "masks": stage_masks[layer_id],
            }
            for layer_id in layers
        }
    bank = {
        "schema": "stage_calibrated_mask_bank_v1",
        "metadata": metadata,
        "ratios": [float(value) for value in ratios],
        "layers": layers,
        "policies": policies,
        "policy_structures": {
            name: ("AL-AM" if name.endswith("_al_am") else "UL-UM")
            for name in policies
        },
        "shuffled_stage_mapping": shuffled,
    }
    add_stage_residual_policies(bank)
    return bank


def add_stage_residual_policies(
    bank: dict[str, Any],
    strengths: tuple[float, ...] = (0.25, 0.50),
) -> dict[str, Any]:
    """Derive bounded stage corrections from an existing compatible mask bank."""

    policies = bank["policies"]
    required = {"trajectory_global", "stage_specific"}
    missing = required - set(policies)
    if missing:
        raise ValueError(f"Cannot build stage residual policies; missing {sorted(missing)}")
    ratios = [float(value) for value in bank["ratios"]]
    layers = [int(layer_id) for layer_id in bank["layers"]]
    global_hash = _policy_hash(policies["trajectory_global"])
    stage_hash = _policy_hash(policies["stage_specific"])
    source_bank_hash = hashlib.sha256(
        f"{global_hash}:{stage_hash}:{','.join(ratio_key(value) for value in ratios)}".encode("ascii")
    ).hexdigest()
    policy_structures = bank.setdefault("policy_structures", {})
    for strength in strengths:
        strength = float(strength)
        if not 0.0 < strength < 1.0:
            raise ValueError("Stage residual strengths must be strictly between 0 and 1")
        name = f"stage_residual_{int(round(strength * 100)):03d}"
        policy: dict[str, Any] = {}
        for stage in STAGES:
            policy[stage] = {}
            for layer_id in layers:
                global_entry = policies["trajectory_global"][stage][layer_id]
                stage_entry = policies["stage_specific"][stage][layer_id]
                metric = global_entry["metric"].float() + strength * (
                    stage_entry["metric"].float() - global_entry["metric"].float()
                )
                mean = global_entry["mean"].float() + strength * (
                    stage_entry["mean"].float() - global_entry["mean"].float()
                )
                entry = {
                    "metric": metric.cpu(),
                    "mean": mean.cpu(),
                    "masks": {ratio_key(ratio): keep_mask(metric, ratio) for ratio in ratios},
                    "residual_strength": strength,
                    "parent_policy_hash": {
                        "trajectory_global": global_hash,
                        "stage_specific": stage_hash,
                    },
                    "bank_hash": source_bank_hash,
                }
                if "output_norm" in global_entry:
                    entry["output_norm"] = global_entry["output_norm"].float().cpu()
                policy[stage][layer_id] = entry
        policies[name] = policy
        policy_structures[name] = "UL-UM"
    return bank


def save_mask_bank(path: str | Path, bank: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)


def load_mask_bank(
    path: str | Path,
    expected_metadata: dict[str, Any] | None = None,
    ignored_metadata_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    try:
        bank = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        bank = torch.load(Path(path), map_location="cpu")
    if bank.get("schema") != "stage_calibrated_mask_bank_v1":
        raise ValueError("Unsupported stage-calibrated mask bank schema")
    if expected_metadata:
        assert_metadata_matches(
            bank.get("metadata", {}),
            expected_metadata,
            ignored_keys=ignored_metadata_keys,
        )
    validate_mask_bank(bank)
    return bank


def validate_mask_bank(bank: dict[str, Any]) -> None:
    ratios = [float(value) for value in bank["ratios"]]
    for policy_name, policy in bank["policies"].items():
        structure = bank.get("policy_structures", {}).get(policy_name, "UL-UM")
        if set(policy) != set(STAGES):
            raise ValueError(f"Policy {policy_name} does not contain exactly the four stages")
        for stage in STAGES:
            previous_by_layer = {}
            for layer_id in bank["layers"]:
                entry = policy[stage][layer_id]
                masks = entry["masks"]
                if "output_norm" in entry and int(entry["output_norm"].numel()) != int(entry["metric"].numel()):
                    raise ValueError(f"Output-norm width mismatch in {policy_name}/{stage}/{layer_id}")
                if set(masks) != {ratio_key(value) for value in ratios}:
                    raise ValueError(f"Mask ratio grid mismatch in {policy_name}/{stage}/{layer_id}")
                channels = int(entry["metric"].numel())
                previous = torch.ones(channels, dtype=torch.bool)
                for ratio in sorted(ratios):
                    mask = masks[ratio_key(ratio)].bool()
                    expected_keep = max(1, channels - int(round(channels * ratio)))
                    if structure == "UL-UM" and int(mask.sum()) != expected_keep:
                        raise ValueError(f"Incorrect keep count in {policy_name}/{stage}/{layer_id}/{ratio}")
                    if bool((mask & ~previous).any()):
                        raise ValueError(f"Masks are not nested in {policy_name}/{stage}/{layer_id}")
                    previous = mask
                previous_by_layer[layer_id] = previous
            if structure == "AL-AM":
                total_channels = sum(
                    int(policy[stage][layer_id]["metric"].numel()) for layer_id in bank["layers"]
                )
                for ratio in ratios:
                    actual_keep = sum(
                        int(policy[stage][layer_id]["masks"][ratio_key(ratio)].sum())
                        for layer_id in bank["layers"]
                    )
                    expected_keep = max(1, total_channels - int(round(total_channels * ratio)))
                    if actual_keep != expected_keep:
                        raise ValueError(f"Incorrect AL-AM keep count in {policy_name}/{stage}/{ratio}")
