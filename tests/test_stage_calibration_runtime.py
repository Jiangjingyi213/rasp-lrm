from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

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
        StageRiskAdaptiveRuntime,
        StageMaskRuntime,
        StaticLayerPruningRuntime,
        StaticCoreResidualStageRuntime,
        griffin_activation_score,
    )
    from src.stage_calibration.stage_risk import STAGE_RISK_CHECKPOINT_SCHEMA, STAGE_RISK_FEATURE_NAMES
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from src.baselines.shortgpt_qwen3 import (
        apply_shortgpt_layer_skip_hooks,
        select_reverse_layers,
        select_shortgpt_layers,
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


    class TinyAddLayer(nn.Module):
        def __init__(self, delta: float) -> None:
            super().__init__()
            self.delta = float(delta)

        def forward(self, hidden_states: torch.Tensor):
            return (hidden_states + self.delta,)


    class TinyDecoderModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(
                [TinyAddLayer(1.0), TinyAddLayer(2.0), TinyAddLayer(3.0)]
            )

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            for layer in self.model.layers:
                output = layer(hidden_states)
                hidden_states = output[0] if isinstance(output, tuple) else output
            return hidden_states


def tiny_bank():
    if not TORCH_AVAILABLE:
        return {}
    sources = ("c4", "prompt_only", "trajectory", *STAGES)
    metrics = {source: {0: torch.arange(4, dtype=torch.float32)} for source in sources}
    means = {source: {0: torch.zeros(4)} for source in sources}
    return build_mask_bank(metadata={}, metrics=metrics, means=means, ratios=[0.0, 0.5])


def tiny_stage_risk_checkpoint(path: Path) -> None:
    """Write the smallest valid, already-gated controller fixture."""

    rng = np.random.default_rng(1)
    design = rng.normal(size=(16, len(STAGE_RISK_FEATURE_NAMES) * 5 + 4))
    labels = np.asarray([0, 1] * 8)
    scaler = StandardScaler().fit(design)
    model = LogisticRegression(max_iter=200).fit(scaler.transform(design), labels)
    raw = model.decision_function(scaler.transform(design))
    platt = LogisticRegression(C=1e6, max_iter=200).fit(raw.reshape(-1, 1), labels)
    joblib.dump(
        {
            "schema": STAGE_RISK_CHECKPOINT_SCHEMA,
            "feature_names": STAGE_RISK_FEATURE_NAMES,
            "stages": STAGES,
            "action_ratios": (0.2, 0.3, 0.4),
            "calibration": "grouped_training_fold_platt",
            "mechanism_gate_passed": True,
            "scaler": scaler,
            "model": model,
            "platt_model": platt,
        },
        path,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for stage calibration runtime tests")
class StageCalibrationRuntimeTest(unittest.TestCase):
    def test_stage_risk_runtime_uses_gated_controller_and_reports_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "controller.joblib"
            tiny_stage_risk_checkpoint(checkpoint)
            runtime = StageRiskAdaptiveRuntime(
                tiny_bank(),
                stage_ratios={stage: 0.4 for stage in STAGES},
                controller_checkpoint_path=str(checkpoint),
                action_ratios=[0.0, 0.2, 0.3, 0.4],
                risk_thresholds={stage: 1.0 for stage in STAGES},
                stage_ratio_caps={stage: 0.4 for stage in STAGES},
                min_warmup_tokens={stage: 0 for stage in STAGES},
                decision_window_tokens=64,
                target_actual_pruning=0.34,
                protected_core_ratios={stage: 0.0 for stage in STAGES},
            )
            runtime.set_stage("setup")
            runtime.set_decode_observation(entropy=1.0, confidence=0.5)
            mask = runtime.observe_or_mask(0, torch.ones(1, 1, 4))
            self.assertIsNotNone(mask)
            summary = runtime.summary()
            self.assertEqual(summary["policy"], "stage_risk_adaptive")
            self.assertEqual(len(summary["controller_decisions"]), 1)
            self.assertIn("setup:0.40", summary["selected_ratio_tokens"])

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

    def test_safe_dynamic_output_aware_uses_down_proj_channel_norms(self) -> None:
        runtime = SafeDynamicStageGriffinRuntime(
            tiny_bank(),
            stage_ratios={stage: 0.5 for stage in STAGES},
            protected_core_ratios={stage: 0.0 for stage in STAGES},
            runtime_weight=1.0,
            prior_weight=0.0,
            score_mode="output_aware",
        )
        runtime.set_output_norms({0: torch.tensor([1.0, 100.0, 1.0, 1.0])})
        runtime.set_stage("setup")
        runtime.observe_or_mask(0, torch.tensor([[[2.0, 0.1, 0.1, 0.1]]]))
        mask = runtime.keep_mask("setup", 0)
        self.assertTrue(bool(mask[0]))
        self.assertTrue(bool(mask[1]))
        summary = runtime.summary()
        self.assertEqual(summary["score_mode"], "output_aware")
        self.assertEqual(summary["output_norm_source"], "frozen_model_down_proj")
        self.assertEqual(summary["output_norm_layers"], [0])

    def test_safe_dynamic_continuity_limits_optional_swaps(self) -> None:
        runtime = SafeDynamicStageGriffinRuntime(
            tiny_bank(),
            stage_ratios={stage: 0.5 for stage in STAGES},
            protected_core_ratios={stage: 0.0 for stage in STAGES},
            runtime_weight=1.0,
            prior_weight=0.0,
            window_tokens={stage: 1 for stage in STAGES},
            max_mask_swap_fraction=0.25,
        )
        runtime.set_stage("setup")
        runtime.observe_or_mask(0, torch.tensor([[[3.0, 2.0, 0.0, 0.0]]]))
        first = runtime.keep_mask("setup", 0).clone()
        runtime._clear_stage_cache("setup")
        runtime.observe_or_mask(0, torch.tensor([[[0.0, 0.0, 3.0, 2.0]]]))
        second = runtime.keep_mask("setup", 0)
        self.assertEqual(int(first.sum().item()), int(second.sum().item()))
        self.assertLessEqual(int((first != second).sum().item()), 2)
        summary = runtime.summary()
        self.assertLessEqual(summary["mask_swap_pairs_by_stage_layer"]["setup:0"], 1)
        self.assertIn("setup:0", summary["mean_mask_jaccard_by_stage_layer"])

    def test_safe_dynamic_default_continuity_preserves_legacy_mask(self) -> None:
        kwargs = {
            "stage_ratios": {stage: 0.5 for stage in STAGES},
            "protected_core_ratios": {stage: 0.0 for stage in STAGES},
            "runtime_weight": 1.0,
            "prior_weight": 0.0,
        }
        legacy = SafeDynamicStageGriffinRuntime(tiny_bank(), **kwargs)
        explicit = SafeDynamicStageGriffinRuntime(
            tiny_bank(), max_mask_swap_fraction=1.0, **kwargs
        )
        values = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
        for runtime in (legacy, explicit):
            runtime.set_stage("setup")
            runtime.observe_or_mask(0, values)
        self.assertTrue(torch.equal(legacy.keep_mask("setup", 0), explicit.keep_mask("setup", 0)))

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

    def test_shortgpt_selects_lowest_bi_layers(self) -> None:
        selected = select_shortgpt_layers(
            {0: 0.9, 1: 0.1, 2: 0.4, 3: 0.2},
            prune_ratio=0.5,
            total_layers=4,
        )
        self.assertEqual(selected, [1, 3])

    def test_reverse_layer_pruning_selects_last_layers(self) -> None:
        selected = select_reverse_layers(prune_ratio=10 / 28, total_layers=28)
        self.assertEqual(selected, list(range(18, 28)))

    def test_shortgpt_layer_skip_hooks_are_removable(self) -> None:
        model = TinyDecoderModel()
        value = torch.zeros(1, 1, 1)
        self.assertTrue(torch.allclose(model(value), torch.full_like(value, 6.0)))
        handles = apply_shortgpt_layer_skip_hooks(model, [1])
        try:
            self.assertTrue(torch.allclose(model(value), torch.full_like(value, 4.0)))
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(torch.allclose(model(value), torch.full_like(value, 6.0)))

    def test_static_layer_runtime_closes_hooks(self) -> None:
        model = TinyDecoderModel()
        handles = apply_shortgpt_layer_skip_hooks(model, [1])
        runtime = StaticLayerPruningRuntime(
            policy="shortgpt",
            backend="shortgpt_layer_skip_logical_v1",
            baseline_type="shortgpt_depth_pruning",
            pruning_granularity="decoder_layer_logical_skip",
            total_layers=3,
            pruned_layers=[1],
            handles=handles,
        )
        self.assertAlmostEqual(runtime.summary()["layer_pruning_ratio"], 1.0 / 3.0)
        runtime.close()
        value = torch.zeros(1, 1, 1)
        self.assertTrue(torch.allclose(model(value), torch.full_like(value, 6.0)))


if __name__ == "__main__":
    unittest.main()
