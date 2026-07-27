from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn

    from src.baselines.wanda_official_qwen3 import (
        WandaInputStats,
        apply_wanda_unstructured_masks_qwen3,
        qwen3_wanda_linear_modules,
    )
    from src.stage_calibration.evaluate import method_requires_mask_bank

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


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for Wanda tests")
class WandaOfficialQwen3Test(unittest.TestCase):
    def test_wanda_policy_does_not_require_mask_bank(self) -> None:
        method = {
            "name": "wanda",
            "policy": "wanda_official",
            "stage_ratios": {"setup": 0.35, "reasoning": 0.35, "verify": 0.35, "final": 0.35},
        }
        self.assertFalse(method_requires_mask_bank(method))

    def test_target_module_discovery_covers_attention_and_mlp_linears(self) -> None:
        modules = qwen3_wanda_linear_modules(TinyModel())
        self.assertEqual(len(modules), 7)
        self.assertIn("layers.0.self_attn.q_proj", modules)
        self.assertIn("layers.0.mlp.down_proj", modules)

    def test_unstructured_wanda_zeroes_low_saliency_weights_per_row(self) -> None:
        model = TinyModel()
        module = model.model.layers[0].mlp.down_proj
        module.weight.data = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        )
        stats = {"layers.0.mlp.down_proj": WandaInputStats(module)}
        stats["layers.0.mlp.down_proj"].scaler_row = torch.tensor([1.0, 4.0, 9.0, 16.0])
        stats["layers.0.mlp.down_proj"].nsamples = 1
        by_module, pruned, total = apply_wanda_unstructured_masks_qwen3(
            model,
            stats,
            sparsity_ratio=0.25,
            target_modules=["mlp.down_proj"],
        )
        self.assertEqual(total, 16)
        self.assertEqual(pruned, 4)
        self.assertAlmostEqual(by_module["layers.0.mlp.down_proj"], 0.25)
        self.assertEqual(int((module.weight.data == 0).sum().item()), 4)


if __name__ == "__main__":
    unittest.main()
