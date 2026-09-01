from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn

    from src.baselines.gisp_mlp_qwen3 import (
        apply_gisp_mlp_artifact_qwen3,
        build_gisp_mlp_mask_artifact_from_scores,
    )
    from src.stage_calibration.evaluate import method_requires_mask_bank

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TinyMlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(4, 4, bias=False)
            self.up_proj = nn.Linear(4, 4, bias=False)
            self.down_proj = nn.Linear(4, 2, bias=False)

    class TinyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = TinyMlp()

    class TinyInner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer()])

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyInner()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for GISP tests")
class GispMlpQwen3Test(unittest.TestCase):
    def test_gisp_policy_does_not_require_mask_bank(self) -> None:
        method = {
            "name": "gisp",
            "policy": "gisp_mlp",
            "stage_ratios": {"setup": 0.30, "reasoning": 0.30, "verify": 0.30, "final": 0.30},
        }
        self.assertFalse(method_requires_mask_bank(method))

    def test_global_iterative_masks_lowest_score_channels(self) -> None:
        masks = {
            0: torch.ones(4, dtype=torch.bool),
            1: torch.ones(4, dtype=torch.bool),
        }
        scores = [
            {
                0: torch.tensor([10.0, 1.0, 9.0, 8.0]),
                1: torch.tensor([7.0, 6.0, 0.5, 5.0]),
            },
            {
                0: torch.tensor([10.0, 1.0, 9.0, 8.0]),
                1: torch.tensor([7.0, 6.0, 0.5, 5.0]),
            },
        ]
        output = build_gisp_mlp_mask_artifact_from_scores(
            scores,
            ratio=0.25,
            initial_masks=masks,
            score_normalization="none",
        )
        self.assertEqual(sum(int((~mask).sum().item()) for mask in output.values()), 2)
        self.assertFalse(bool(output[0][1]))
        self.assertFalse(bool(output[1][2]))

    def test_global_iterative_respects_protected_layers(self) -> None:
        masks = {
            0: torch.ones(4, dtype=torch.bool),
            1: torch.ones(4, dtype=torch.bool),
        }
        scores = [
            {
                0: torch.tensor([0.1, 0.2, 0.3, 0.4]),
                1: torch.tensor([5.0, 6.0, 7.0, 8.0]),
            }
        ]
        output = build_gisp_mlp_mask_artifact_from_scores(
            scores,
            ratio=0.25,
            initial_masks=masks,
            score_normalization="none",
            protected_layers=[0],
        )
        self.assertTrue(bool(output[0].all()))
        self.assertEqual(int((~output[1]).sum().item()), 2)

    def test_logical_artifact_masks_down_projection_input_and_closes(self) -> None:
        model = TinyModel()
        down = model.model.layers[0].mlp.down_proj
        down.weight.data.fill_(1.0)
        artifact = {
            "schema": "gisp_mlp_qwen3_mask_artifact_v1",
            "summary": {
                "method": "gisp_mlp_qwen3",
                "source": "test",
                "baseline_type": "gisp_global_iterative_structured_mlp_qwen3_port",
                "pruning_granularity": "mlp_channel_structured",
                "ratio": 0.5,
                "iterations": 1,
                "calibration_path": "calibration.jsonl",
                "calibration_samples": 1,
                "calibration_max_input_tokens": 8,
                "calibration_prompt_mode": "structured_prompt",
                "calibration_text_field": "text",
                "score_normalization": "none",
                "target": "mlp.intermediate_channels",
                "physical_pruning": False,
                "total_layers": 1,
                "pruned_layers": [0],
                "original_intermediate_size": 4,
                "kept_channels_per_layer": {"0": 2},
                "pruned_channels_per_layer": {"0": 2},
                "actual_mlp_channel_pruning_ratio": 0.5,
                "matched_rasp_reference": "t30_math_safe",
                "target_matched_to_rasp_actual_mlp_pruning": 0.5,
            },
            "keep_indices_by_layer": {"0": [0, 2]},
            "pruned_indices_by_layer": {"0": [1, 3]},
        }
        _summary, handles = apply_gisp_mlp_artifact_qwen3(model, artifact)
        try:
            output = down(torch.ones(1, 1, 4))
            self.assertTrue(torch.equal(output, torch.full((1, 1, 2), 2.0)))
        finally:
            for handle in handles:
                handle.remove()
        output = down(torch.ones(1, 1, 4))
        self.assertTrue(torch.equal(output, torch.full((1, 1, 2), 4.0)))


if __name__ == "__main__":
    unittest.main()
