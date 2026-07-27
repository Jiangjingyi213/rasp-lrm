from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn

    from src.baselines.sparsegpt_official_qwen3 import (
        SparseGPTArtifactSummary,
        SparseGPTInputStats,
        apply_sparsegpt_official_qwen3_artifact,
        qwen3_sparsegpt_linear_modules,
        sparsegpt_prune_linear_unstructured,
    )
    from src.stage_calibration.evaluate import method_requires_mask_bank
    from src.utils.io import write_json

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TinyAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.k_proj = nn.Linear(4, 4, bias=False)
            self.v_proj = nn.Linear(4, 4, bias=False)
            self.o_proj = nn.Linear(4, 4, bias=False)

    class TinyMlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(4, 4, bias=False)
            self.up_proj = nn.Linear(4, 4, bias=False)
            self.down_proj = nn.Linear(4, 4, bias=False)

    class TinyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = TinyAttn()
            self.mlp = TinyMlp()

    class TinyInner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer()])

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyInner()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for SparseGPT tests")
class SparseGPTOfficialQwen3Test(unittest.TestCase):
    def test_sparsegpt_policy_does_not_require_mask_bank(self) -> None:
        method = {
            "name": "sparsegpt",
            "policy": "sparsegpt_official",
            "stage_ratios": {"setup": 0.35, "reasoning": 0.35, "verify": 0.35, "final": 0.35},
        }
        self.assertFalse(method_requires_mask_bank(method))

    def test_target_module_discovery_covers_attention_and_mlp_linears(self) -> None:
        modules = qwen3_sparsegpt_linear_modules(TinyModel())
        self.assertEqual(len(modules), 7)
        self.assertIn("layers.0.self_attn.q_proj", modules)
        self.assertIn("layers.0.mlp.down_proj", modules)

    def test_unstructured_sparsegpt_reaches_target_sparsity(self) -> None:
        module = nn.Linear(4, 4, bias=False)
        module.weight.data = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
        stats = SparseGPTInputStats(module)
        stats.hessian = torch.eye(4)
        stats.nsamples = 4
        sparsity = sparsegpt_prune_linear_unstructured(module, stats, sparsity_ratio=0.25, blocksize=2)
        self.assertAlmostEqual(sparsity, 0.25)
        self.assertEqual(int((module.weight.data == 0).sum().item()), 4)

    def test_apply_artifact_loads_pruned_weights(self) -> None:
        model = TinyModel()
        module_name = "layers.0.mlp.down_proj"
        target = model.model.layers[0].mlp.down_proj
        new_weight = torch.zeros_like(target.weight.data)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            (artifact / "weights").mkdir()
            torch.save({module_name: new_weight.half()}, artifact / "weights" / "layer_00000.pt")
            summary = SparseGPTArtifactSummary(
                method="sparsegpt_official_qwen3",
                source="test",
                baseline_type="official_style_sparsegpt",
                pruning_granularity="weight_unstructured",
                sparsity_ratio=1.0,
                calibration_path="calibration.jsonl",
                calibration_samples=1,
                calibration_max_input_tokens=8,
                calibration_batch_size=1,
                target_modules=["mlp.down_proj"],
                blocksize=2,
                percdamp=0.01,
                total_pruned_weights=16,
                total_target_weights=16,
                weight_sparsity_overall=1.0,
                weight_sparsity_by_module={module_name: 1.0},
                pruning_method="test",
                artifact_path=str(artifact),
                artifact_hash="testhash",
                artifact_save_dtype="float16",
                real_speedup_claimed=False,
                matched_rasp_reference="t30_math_safe",
                target_matched_to_rasp_actual_mlp_pruning=0.35,
            )
            write_json(artifact / "summary.json", summary.__dict__)
            apply_sparsegpt_official_qwen3_artifact(model, artifact_dir=artifact)
        self.assertEqual(float(target.weight.data.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
