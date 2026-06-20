from __future__ import annotations

import unittest

from src.stage_calibration.protocol import STAGES, illegal_stage_tag_reason

try:
    import torch
    from torch import nn

    from src.stage_calibration.mask_bank import build_mask_bank
    from src.stage_calibration.runtime import FixedStageMaskedQwen3MLP, StageMaskRuntime

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TinyMlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(2, 4, bias=False)
            self.up_proj = nn.Linear(2, 4, bias=False)
            self.down_proj = nn.Linear(4, 2, bias=False)
            self.act_fn = nn.Identity()


def tiny_bank():
    if not TORCH_AVAILABLE:
        return {}
    sources = ("c4", "prompt_only", "trajectory", *STAGES)
    metrics = {source: {0: torch.arange(4, dtype=torch.float32)} for source in sources}
    means = {source: {0: torch.zeros(4)} for source in sources}
    return build_mask_bank(metadata={}, metrics=metrics, means=means, ratios=[0.0, 0.5])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for stage calibration runtime tests")
class StageCalibrationRuntimeTest(unittest.TestCase):
    def test_ratio_zero_is_dense_equivalent(self) -> None:
        original = TinyMlp()
        runtime = StageMaskRuntime(tiny_bank(), "stage_specific", {stage: 0.0 for stage in STAGES})
        wrapped = FixedStageMaskedQwen3MLP(original, 0, runtime)
        value = torch.randn(1, 1, 2)
        runtime.set_stage("setup")
        expected = original.down_proj(original.act_fn(original.gate_proj(value)) * original.up_proj(value))
        self.assertTrue(torch.allclose(wrapped(value), expected))

    def test_fallback_disables_mask(self) -> None:
        runtime = StageMaskRuntime(tiny_bank(), "stage_specific", {stage: 0.5 for stage in STAGES})
        runtime.set_stage("reasoning")
        self.assertEqual(runtime.active_ratio(), 0.5)
        runtime.fallback_dense("invalid")
        self.assertEqual(runtime.active_ratio(), 0.0)

    def test_runtime_keeps_protocol_illegal_tag_reason(self) -> None:
        runtime = StageMaskRuntime(tiny_bank(), "stage_specific", {stage: 0.5 for stage in STAGES})
        reason = illegal_stage_tag_reason("<STAGE_SETUP> done </STAGE_SETUP>")
        self.assertEqual(reason, "closing_stage_marker:</STAGE_SETUP>")
        runtime.set_stage("setup")
        runtime.fallback_dense(reason)
        self.assertEqual(runtime.summary()["fallback_reason"], reason)
        self.assertEqual(runtime.active_ratio(), 0.0)


if __name__ == "__main__":
    unittest.main()
