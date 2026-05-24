from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _select_neurons(scores: torch.Tensor, method: str, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.empty(scores.shape[0], 0, dtype=torch.long, device=scores.device)
    k = min(k, scores.shape[-1])
    if method == "topk":
        return torch.topk(scores, k=k, dim=-1).indices
    if method == "magnitude":
        return torch.topk(scores, k=k, dim=-1).indices
    raise ValueError(f"Unsupported GRIFFIN selection method: {method}")


class Qwen3GriffinMLP(nn.Module):
    def __init__(self, original_mlp: nn.Module, config: Any, density: float, selection_method: str, mode: str) -> None:
        super().__init__()
        if not 0.0 <= density <= 1.0:
            raise ValueError(f"GRIFFIN density must be in [0, 1], got {density}")
        if mode not in {"gen", "class"}:
            raise ValueError(f"Unsupported GRIFFIN mode: {mode}")
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.intermediate_size)
        self.density = float(density)
        self.selection_method = selection_method
        self.mode = mode

        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn

        self.gate_proj_reduced: nn.Linear | None = None
        self.up_proj_reduced: nn.Linear | None = None
        self.down_proj_reduced: nn.Linear | None = None

    def prepare_reduced_weights(self, topk_indices: torch.Tensor) -> None:
        if topk_indices.ndim != 2 or topk_indices.shape[0] != 1:
            raise ValueError("GRIFFIN Qwen3 adapter currently supports batch size 1")
        topk_indices = topk_indices[0]
        device = self.gate_proj.weight.device
        dtype = self.gate_proj.weight.dtype
        kept = int(topk_indices.numel())

        gate = nn.Linear(self.gate_proj.weight.shape[1], kept, bias=False, device=device, dtype=dtype)
        up = nn.Linear(self.up_proj.weight.shape[1], kept, bias=False, device=device, dtype=dtype)
        down = nn.Linear(kept, self.down_proj.weight.shape[0], bias=False, device=device, dtype=dtype)

        with torch.no_grad():
            gate.weight.copy_(self.gate_proj.weight.index_select(0, topk_indices))
            up.weight.copy_(self.up_proj.weight.index_select(0, topk_indices))
            down.weight.copy_(self.down_proj.weight.index_select(1, topk_indices))

        self.gate_proj_reduced = gate
        self.up_proj_reduced = up
        self.down_proj_reduced = down

    def _full_mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def _reduced_mlp(self, x: torch.Tensor) -> torch.Tensor:
        if self.density <= 0.0:
            return torch.zeros_like(x)
        if self.gate_proj_reduced is None or self.up_proj_reduced is None or self.down_proj_reduced is None:
            return self._full_mlp(x)
        return self.down_proj_reduced(self.act_fn(self.gate_proj_reduced(x)) * self.up_proj_reduced(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.density >= 1.0:
            return self._full_mlp(x)

        if self.mode == "gen":
            if x.shape[1] > 1:
                int_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                if self.density > 0.0:
                    k = max(1, int(int_states.shape[-1] * self.density))
                    normalized = int_states / int_states.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    neuron_stat = normalized.norm(dim=1)
                    topk_indices = _select_neurons(neuron_stat, self.selection_method, k)
                    self.prepare_reduced_weights(topk_indices)
                return self.down_proj(int_states)
            return self._reduced_mlp(x)

        if self.mode == "class":
            int_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            if self.density <= 0.0:
                int_states[:, -1] = 0
            else:
                k = max(1, int(int_states.shape[-1] * self.density))
                normalized = int_states / int_states.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                neuron_stat = normalized[:, :-1].norm(dim=1)
                topk_indices = _select_neurons(neuron_stat, self.selection_method, k)
                mask = torch.zeros_like(int_states[:, -1], dtype=torch.bool)
                mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(mask))
                int_states[:, -1] = mask * int_states[:, -1]
            return self.down_proj(int_states)

        raise NotImplementedError(f"Unsupported GRIFFIN mode: {self.mode}")


def apply_griffin_qwen3(
    model: nn.Module,
    density: float = 0.5,
    selection_method: str = "topk",
    mode: str = "gen",
) -> nn.Module:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected a Qwen3-style causal LM with model.layers")
    model.config.griffin_density = float(density)
    model.config.griffin_selection_method = selection_method
    model.config.griffin_mode = mode
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            raise ValueError("Expected Qwen3 MLP with gate_proj/up_proj/down_proj/act_fn")
        layer.mlp = Qwen3GriffinMLP(mlp, model.config, density, selection_method, mode)
    return model
