from __future__ import annotations

import unittest

from src.stage_calibration.protocol import STAGES, illegal_stage_tag_reason

try:
    import torch
    from torch import nn

    from src.stage_calibration.mask_bank import build_mask_bank
    from src.rasp.activation_ranker import rank_intermediate_neurons
    from src.stage_calibration.runtime import (
        AdaptiveStageGriffinQwen3MLP,
        AdaptiveStageGriffinRuntime,
        FixedStageMaskedQwen3MLP,
        GriffinPromptQwen3MLP,
        GriffinPromptRuntime,
        SafeDynamicStageGriffinRuntime,
        StageMaskRuntime,
        StaticCoreResidualStageRuntime,
        griffin_activation_score,
    )

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

    def test_fixed_stage_runtime_accepts_ratio_outside_bank_grid(self) -> None:
        original = TinyMlp()
        runtime = StageMaskRuntime(tiny_bank(), "stage_specific", {stage: 0.25 for stage in STAGES})
        wrapped = FixedStageMaskedQwen3MLP(original, 0, runtime)
        runtime.set_stage("setup")
        value = torch.randn(1, 1, 2)
        output = wrapped(value)
        self.assertEqual(tuple(output.shape), (1, 1, 2))
        self.assertEqual(runtime.active_ratio(), 0.25)

    def test_runtime_keeps_protocol_illegal_tag_reason(self) -> None:
        runtime = StageMaskRuntime(tiny_bank(), "stage_specific", {stage: 0.5 for stage in STAGES})
        reason = illegal_stage_tag_reason("[[STAGE_SETUP]] done </STAGE_SETUP>")
        self.assertEqual(reason, "closing_stage_marker:</STAGE_SETUP>")
        self.assertEqual(
            illegal_stage_tag_reason("<STAGE_SETUP> done"),
            "legacy_stage_marker:<STAGE_SETUP>",
        )
        runtime.set_stage("setup")
        runtime.fallback_dense(reason)
        self.assertEqual(runtime.summary()["fallback_reason"], reason)
        self.assertEqual(runtime.active_ratio(), 0.0)

    def test_griffin_score_matches_existing_activation_ranker(self) -> None:
        values = torch.tensor(
            [[[1.0, 0.0, 2.0, 0.5], [0.0, 3.0, 1.0, 0.5]]],
            dtype=torch.float32,
        )
        expected = rank_intermediate_neurons(values)
        actual = torch.argsort(griffin_activation_score(values), descending=True)
        self.assertTrue(torch.equal(actual, expected))

    def test_prompt_griffin_prefill_dense_then_decode_masks(self) -> None:
        original = TinyMlp()
        runtime = GriffinPromptRuntime(prune_ratio=0.5)
        wrapped = GriffinPromptQwen3MLP(original, 0, runtime)
        prompt = torch.randn(1, 3, 2)
        expected_prompt = original.down_proj(original.act_fn(original.gate_proj(prompt)) * original.up_proj(prompt))
        self.assertTrue(torch.allclose(wrapped(prompt), expected_prompt))
        runtime.set_stage("reasoning")
        token = torch.randn(1, 1, 2)
        output = wrapped(token)
        self.assertEqual(tuple(output.shape), (1, 1, 2))
        summary = runtime.summary()
        self.assertEqual(summary["backend"], "griffin_prompt_logical_v1")
        self.assertEqual(summary["prompt_dense_tokens"], 3)
        self.assertEqual(summary["decode_masked_tokens"], 1)
        self.assertAlmostEqual(summary["actual_average_mlp_pruning_ratio"], 0.125)

    def test_prompt_griffin_ratio_zero_is_dense_equivalent(self) -> None:
        original = TinyMlp()
        runtime = GriffinPromptRuntime(prune_ratio=0.0)
        wrapped = GriffinPromptQwen3MLP(original, 0, runtime)
        wrapped(torch.randn(1, 3, 2))
        value = torch.randn(1, 1, 2)
        expected = original.down_proj(original.act_fn(original.gate_proj(value)) * original.up_proj(value))
        self.assertTrue(torch.allclose(wrapped(value), expected))
        self.assertEqual(runtime.summary()["actual_average_mlp_pruning_ratio"], 0.0)

    def test_prompt_griffin_fallback_is_record_only(self) -> None:
        runtime = GriffinPromptRuntime(prune_ratio=0.5)
        runtime.set_stage("setup")
        runtime.fallback_dense("invalid")
        runtime.record_token()
        summary = runtime.summary()
        self.assertEqual(summary["fallback_reason"], "invalid")
        self.assertEqual(summary["fallback_behavior"], "record_only_keep_prompt_masking")
        self.assertEqual(summary["tokens_by_stage"]["setup"], 1)

    def test_adaptive_runtime_warmup_then_masks(self) -> None:
        original = TinyMlp()
        runtime = AdaptiveStageGriffinRuntime(
            tiny_bank(),
            stage_ratios={"setup": 0.5, "reasoning": 0.5, "verify": 0.5, "final": 0.0},
            warmup_tokens={"setup": 0, "reasoning": 2, "verify": 0, "final": 0},
            alpha=0.7,
        )
        wrapped = AdaptiveStageGriffinQwen3MLP(original, 0, runtime)
        prompt = torch.randn(1, 3, 2)
        wrapped(prompt)
        runtime.set_stage("reasoning")
        token = torch.randn(1, 1, 2)
        wrapped(token)
        wrapped(token)
        self.assertEqual(runtime.summary()["dense_observation_tokens_by_stage"]["reasoning"], 2)
        wrapped(token)
        summary = runtime.summary()
        self.assertEqual(summary["masked_tokens_by_stage"]["reasoning"], 1)
        self.assertEqual(summary["mask_refresh_count_by_stage"]["reasoning"], 1)

    def test_adaptive_final_ratio_zero_is_dense_equivalent(self) -> None:
        original = TinyMlp()
        runtime = AdaptiveStageGriffinRuntime(
            tiny_bank(),
            stage_ratios={"setup": 0.5, "reasoning": 0.5, "verify": 0.5, "final": 0.0},
            warmup_tokens={"setup": 0, "reasoning": 0, "verify": 0, "final": 0},
        )
        wrapped = AdaptiveStageGriffinQwen3MLP(original, 0, runtime)
        runtime.set_stage("final")
        value = torch.randn(1, 1, 2)
        expected = original.down_proj(original.act_fn(original.gate_proj(value)) * original.up_proj(value))
        self.assertTrue(torch.allclose(wrapped(value), expected))
        summary = runtime.summary()
        self.assertNotIn("final", summary.get("masked_tokens_by_stage", {}))

    def test_adaptive_fallback_keeps_following_tokens_dense(self) -> None:
        original = TinyMlp()
        runtime = AdaptiveStageGriffinRuntime(
            tiny_bank(),
            stage_ratios={stage: 0.5 for stage in STAGES},
            warmup_tokens={stage: 0 for stage in STAGES},
        )
        wrapped = AdaptiveStageGriffinQwen3MLP(original, 0, runtime)
        wrapped(torch.randn(1, 3, 2))
        runtime.set_stage("setup")
        runtime.fallback_dense("invalid")
        wrapped(torch.randn(1, 1, 2))
        summary = runtime.summary()
        self.assertEqual(summary["fallback_reason"], "invalid")
        self.assertEqual(summary["dense_observation_tokens_by_stage"]["dense"], 1)
        self.assertNotIn("setup", summary.get("masked_tokens_by_stage", {}))

    def test_safe_dynamic_protected_core_keeps_high_wifv_channels(self) -> None:
        runtime = SafeDynamicStageGriffinRuntime(
            tiny_bank(),
            stage_ratios={"setup": 0.5, "reasoning": 0.5, "verify": 0.5, "final": 0.0},
            protected_core_ratios={"setup": 0.5, "reasoning": 0.0, "verify": 0.0, "final": 1.0},
            refresh_intervals={"setup": 2, "reasoning": 0, "verify": 0, "final": 0},
            window_tokens={"setup": 2, "reasoning": 1, "verify": 1, "final": 1},
            runtime_weight=1.0,
            prior_weight=0.0,
        )
        runtime.set_stage("setup")
        wrapped = AdaptiveStageGriffinQwen3MLP(TinyMlp(), 0, runtime)
        wrapped(torch.randn(1, 1, 2))
        mask = runtime.keep_mask("setup", 0)
        self.assertTrue(bool(mask[2]))
        self.assertTrue(bool(mask[3]))

    def test_safe_dynamic_accepts_ratio_outside_bank_grid(self) -> None:
        runtime = SafeDynamicStageGriffinRuntime(
            tiny_bank(),
            stage_ratios={stage: 0.25 for stage in STAGES},
            protected_core_ratios={stage: 0.0 for stage in STAGES},
            runtime_weight=0.4,
            prior_weight=0.6,
        )
        runtime.set_stage("setup")
        wrapped = AdaptiveStageGriffinQwen3MLP(TinyMlp(), 0, runtime)
        wrapped(torch.randn(1, 1, 2))
        mask = runtime.keep_mask("setup", 0)
        self.assertEqual(int(mask.sum().item()), 3)

    def test_safe_dynamic_refreshes_masks_inside_stage(self) -> None:
        runtime = SafeDynamicStageGriffinRuntime(
            tiny_bank(),
            stage_ratios={"setup": 0.5, "reasoning": 0.5, "verify": 0.5, "final": 0.0},
            protected_core_ratios={stage: 0.0 for stage in STAGES},
            refresh_intervals={"setup": 2, "reasoning": 0, "verify": 0, "final": 0},
            window_tokens={"setup": 2, "reasoning": 1, "verify": 1, "final": 1},
            runtime_weight=0.4,
            prior_weight=0.6,
        )
        runtime.set_stage("setup")
        wrapped = AdaptiveStageGriffinQwen3MLP(TinyMlp(), 0, runtime)
        token = torch.randn(1, 1, 2)
        wrapped(token)
        wrapped(token)
        wrapped(token)
        summary = runtime.summary()
        self.assertEqual(summary["masked_tokens_by_stage"]["setup"], 3)
        self.assertGreaterEqual(summary["mask_refresh_count_by_stage"]["setup"], 2)
        self.assertEqual(summary["refresh_intervals"]["setup"], 2)

    def test_safe_dynamic_final_ratio_zero_is_dense_equivalent(self) -> None:
        original = TinyMlp()
        runtime = SafeDynamicStageGriffinRuntime(
            tiny_bank(),
            stage_ratios={"setup": 0.5, "reasoning": 0.5, "verify": 0.5, "final": 0.0},
            protected_core_ratios={stage: 0.0 for stage in STAGES},
        )
        wrapped = AdaptiveStageGriffinQwen3MLP(original, 0, runtime)
        runtime.set_stage("final")
        value = torch.randn(1, 1, 2)
        expected = original.down_proj(original.act_fn(original.gate_proj(value)) * original.up_proj(value))
        self.assertTrue(torch.allclose(wrapped(value), expected))
        self.assertNotIn("final", runtime.summary().get("masked_tokens_by_stage", {}))

    def test_static_core_residual_swaps_without_changing_keep_count(self) -> None:
        runtime = StaticCoreResidualStageRuntime(
            tiny_bank(),
            stage_ratios={"setup": 0.5, "reasoning": 0.5, "verify": 0.5, "final": 0.0},
            static_core_ratios={"setup": 0.5, "reasoning": 1.0, "verify": 1.0, "final": 1.0},
            swap_ratios={"setup": 0.25, "reasoning": 0.0, "verify": 0.0, "final": 0.0},
            runtime_weight=0.0,
            prior_weight=1.0,
        )
        runtime.set_stage("setup")
        wrapped = AdaptiveStageGriffinQwen3MLP(TinyMlp(), 0, runtime)
        wrapped(torch.randn(1, 1, 2))
        mask = runtime.keep_mask("setup", 0)
        self.assertEqual(int(mask.sum().item()), 2)
        self.assertTrue(bool(mask[3]))
        self.assertTrue(bool(mask[1]))
        self.assertFalse(bool(mask[2]))
        summary = runtime.summary()
        self.assertEqual(
            summary["actual_swapped_channels_by_stage_layer"]["setup"]["0"],
            1,
        )

    def test_static_core_residual_full_core_prevents_swaps(self) -> None:
        runtime = StaticCoreResidualStageRuntime(
            tiny_bank(),
            stage_ratios={"setup": 0.5, "reasoning": 0.5, "verify": 0.5, "final": 0.0},
            static_core_ratios={stage: 1.0 for stage in STAGES},
            swap_ratios={"setup": 0.25, "reasoning": 0.0, "verify": 0.0, "final": 0.0},
            runtime_weight=0.0,
            prior_weight=1.0,
        )
        runtime.set_stage("setup")
        wrapped = AdaptiveStageGriffinQwen3MLP(TinyMlp(), 0, runtime)
        wrapped(torch.randn(1, 1, 2))
        mask = runtime.keep_mask("setup", 0)
        expected_base = tiny_bank()["policies"]["trajectory_global"]["setup"][0]["masks"]["0.5000"]
        self.assertTrue(torch.equal(mask.cpu(), expected_base.cpu()))
        self.assertEqual(
            runtime.summary()["actual_swapped_channels_by_stage_layer"]["setup"]["0"],
            0,
        )

    def test_static_core_residual_final_ratio_zero_is_dense_equivalent(self) -> None:
        original = TinyMlp()
        runtime = StaticCoreResidualStageRuntime(
            tiny_bank(),
            stage_ratios={"setup": 0.5, "reasoning": 0.5, "verify": 0.5, "final": 0.0},
            static_core_ratios={stage: 0.0 for stage in STAGES},
            swap_ratios={stage: 0.25 for stage in STAGES},
        )
        wrapped = AdaptiveStageGriffinQwen3MLP(original, 0, runtime)
        runtime.set_stage("final")
        value = torch.randn(1, 1, 2)
        expected = original.down_proj(original.act_fn(original.gate_proj(value)) * original.up_proj(value))
        self.assertTrue(torch.allclose(wrapped(value), expected))
        self.assertNotIn("final", runtime.summary().get("masked_tokens_by_stage", {}))

    def test_static_core_residual_fallback_keeps_following_tokens_dense(self) -> None:
        runtime = StaticCoreResidualStageRuntime(
            tiny_bank(),
            stage_ratios={stage: 0.5 for stage in STAGES},
            static_core_ratios={stage: 0.0 for stage in STAGES},
            swap_ratios={stage: 0.25 for stage in STAGES},
        )
        wrapped = AdaptiveStageGriffinQwen3MLP(TinyMlp(), 0, runtime)
        runtime.set_stage("setup")
        runtime.fallback_dense("invalid")
        wrapped(torch.randn(1, 1, 2))
        summary = runtime.summary()
        self.assertEqual(summary["fallback_reason"], "invalid")
        self.assertEqual(summary["dense_observation_tokens_by_stage"]["dense"], 1)
        self.assertNotIn("setup", summary.get("masked_tokens_by_stage", {}))


if __name__ == "__main__":
    unittest.main()
