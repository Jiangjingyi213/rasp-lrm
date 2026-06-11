from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.probes.rasp_train_dataset import (
    DEFAULT_RATIOS,
    RaspTrainPolicyDataset,
    build_policy_features,
    ratio_index,
)
from src.main_train_rasp_train_router import problem_level_three_way_split, validate_equivalent_action_labels
from src.rasp.budget_controller import RuntimeObservation
from src.rasp.fair_benchmark import (
    LinearActionRiskNet,
    create_split_manifest,
    indices_for_split,
    monotonic_risk_envelope,
)
from src.rasp.train_controller import RaspTrainPolicyController
from src.rasp.phase_b2 import (
    PhaseB2LinearFlipNet,
    PhaseB2MultiTaskNet,
    build_phase_b2_state_features,
    multitask_loss,
    stratified_problem_split,
    validate_phase_b2_manifest,
)
from src.rasp.phase_b25 import (
    PhaseB25ActionNet,
    boundary_any_flip_metrics,
    fit_phase_b25_transform,
    transform_phase_b25_features,
)
from src.rasp.phase_b25b import HiddenActionResidual, combined_risks
from src.rasp.phase_b2_controller import PhaseB2UncertaintyController
from src.rasp.stage_probe import (
    StageProbeNet,
    classify_operational_stage,
    fit_stage_transform,
    indices_for_stage_split,
    problem_stage_split,
    stage_metrics,
    transform_stage_features,
    validate_stage_manifest,
)
from src.rasp.train_policy import (
    POLICY_FEATURE_SCHEMA,
    ActionRiskPolicyNet,
    action_risk_loss,
    threshold_for_budget,
)
from src.utils.io import write_jsonl


class RaspTrainPolicyTest(unittest.TestCase):
    def test_ratio_index_requires_supported_ratio(self) -> None:
        self.assertEqual(ratio_index(0.2, DEFAULT_RATIOS), DEFAULT_RATIOS.index(0.2))
        with self.assertRaises(ValueError):
            ratio_index(0.15, DEFAULT_RATIOS)

    def test_policy_dataset_builds_one_step_budget_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "dataset": "gsm8k",
                    "id": "x",
                    "segment_id": 0,
                    "entropy": 0.1,
                    "confidence": 0.9,
                    "position": 0.0,
                    "target_budget": 0.2,
                    "available_budget_before_selection": 0.2,
                    "oracle_ratio": 0.1,
                    "candidate_flipped": [False, False, False, False, True, True, True],
                    "candidate_unsafe": [False, False, False, False, True, True, True],
                }
            ]
            write_jsonl(root / "rows.jsonl", rows)
            torch.save(torch.zeros(1, 4), root / "hidden.pt")
            dataset = RaspTrainPolicyDataset(root / "rows.jsonl", root / "hidden.pt")
            features, unsafe_mask, flipped_mask, ratios, target_budget, index = dataset[0]
            self.assertEqual(index, 0)
            self.assertAlmostEqual(float(target_budget), 0.2)
            self.assertEqual(tuple(unsafe_mask.shape), (len(DEFAULT_RATIOS),))
            self.assertEqual(tuple(flipped_mask.shape), (len(DEFAULT_RATIOS),))
            self.assertEqual(tuple(ratios.shape), (len(DEFAULT_RATIOS),))
            self.assertGreater(features.numel(), 4)

    def test_risk_features_do_not_depend_on_budget(self) -> None:
        common = {
            "hidden_state": torch.zeros(4),
            "entropy": 0.1,
            "confidence": 0.9,
            "position": 0.5,
            "dataset": "gsm8k",
        }
        features = build_policy_features(**common)
        self.assertEqual(features.numel(), 11)

    def test_three_way_split_keeps_problems_disjoint(self) -> None:
        rows = [
            {"dataset": "gsm8k", "id": str(problem), "segment_id": segment}
            for problem in range(10)
            for segment in range(2)
        ]
        train, calibration, test, _keys = problem_level_three_way_split(rows, 0.4, 1)
        problem_sets = [
            {(rows[index]["dataset"], rows[index]["id"]) for index in indices}
            for indices in (train, calibration, test)
        ]
        self.assertTrue(problem_sets[0].isdisjoint(problem_sets[1]))
        self.assertTrue(problem_sets[0].isdisjoint(problem_sets[2]))
        self.assertTrue(problem_sets[1].isdisjoint(problem_sets[2]))
        repeated = problem_level_three_way_split(rows, 0.4, 1)
        self.assertEqual((train, calibration, test), repeated[:3])

    def test_fair_manifest_is_reusable_and_problem_disjoint(self) -> None:
        rows = [
            {"dataset": "gsm8k", "id": str(problem), "segment_id": segment}
            for problem in range(20)
            for segment in range(2)
        ]
        manifest = create_split_manifest(rows, seed=2)
        split_indices = {
            name: indices_for_split(rows, manifest, name)
            for name in ("train", "calibration", "test")
        }
        split_problems = {
            name: {(rows[index]["dataset"], rows[index]["id"]) for index in indices}
            for name, indices in split_indices.items()
        }
        self.assertTrue(split_problems["train"].isdisjoint(split_problems["calibration"]))
        self.assertTrue(split_problems["train"].isdisjoint(split_problems["test"]))
        self.assertTrue(split_problems["calibration"].isdisjoint(split_problems["test"]))
        self.assertEqual(manifest, create_split_manifest(rows, seed=2))

    def test_fair_linear_ratio_only_and_monotonic_envelope(self) -> None:
        model = LinearActionRiskNet(0)
        logits = model(torch.empty(2, 0), torch.tensor(DEFAULT_RATIOS))
        self.assertEqual(tuple(logits.shape), (2, len(DEFAULT_RATIOS)))
        risks = monotonic_risk_envelope([[0.1, 0.4, 0.2, 0.8]])
        self.assertEqual(risks, [[0.1, 0.4, 0.4, 0.8]])

    def test_phase_b2_split_is_problem_disjoint_and_stratified(self) -> None:
        rows = []
        for dataset in ("gsm8k", "math_train"):
            for problem in range(40):
                rows.append(
                    {
                        "dataset": dataset,
                        "id": str(problem),
                        "candidate_flipped": [False, problem % 2 == 0],
                    }
                )
        manifest = stratified_problem_split(rows, seed=1)
        split_sets = {
            name: {tuple(value) for value in manifest["split_problem_keys"][name]}
            for name in ("train", "validation", "calibration", "test")
        }
        for name, values in split_sets.items():
            others = set().union(*(other for other_name, other in split_sets.items() if other_name != name))
            self.assertTrue(values.isdisjoint(others))
        for split in split_sets.values():
            self.assertEqual({key[0] for key in split}, {"gsm8k", "math_train"})
        validate_phase_b2_manifest(rows, manifest, seed=1)
        self.assertEqual(
            manifest["split_strategy"],
            "problem_level_dataset_and_positive_burden_stratified_60_10_15_15",
        )
        broken = {**manifest, "split_problem_keys": {**manifest["split_problem_keys"]}}
        broken["split_problem_keys"]["validation"] = broken["split_problem_keys"]["train"]
        with self.assertRaises(ValueError):
            validate_phase_b2_manifest(rows, broken, seed=1)

    def test_phase_b2_simple_feature_baselines(self) -> None:
        row = {"position": 0.25, "entropy": 0.5, "confidence": 0.75}
        hidden = torch.zeros(4)
        self.assertEqual(build_phase_b2_state_features(hidden, row, "ratio_only").tolist(), [0.0])
        self.assertEqual(build_phase_b2_state_features(hidden, row, "position").tolist(), [0.25])
        self.assertEqual(
            build_phase_b2_state_features(hidden, row, "uncertainty").tolist(),
            [0.5, 0.75, 0.25],
        )

    def test_phase_b2_multitask_loss_is_finite(self) -> None:
        model = PhaseB2MultiTaskNet(dim=4, hidden_dim=8)
        outputs = model(torch.zeros(2, 4), torch.tensor(DEFAULT_RATIOS))
        flipped = torch.zeros(2, len(DEFAULT_RATIOS))
        flipped[:, -1] = 1.0
        divergence = torch.tensor(DEFAULT_RATIOS).unsqueeze(0).expand(2, -1)
        drift = divergence.clone()
        loss, parts = multitask_loss(
            outputs,
            flipped,
            divergence,
            drift,
            torch.tensor(DEFAULT_RATIOS).unsqueeze(0).expand(2, -1),
            positive_weight=5.0,
            divergence_weight=0.5,
            hidden_drift_weight=0.5,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts["flip_loss"], 0.0)

    def test_phase_b2_linear_flip_model_shape(self) -> None:
        model = PhaseB2LinearFlipNet(dim=4)
        outputs = model(torch.zeros(2, 4), torch.tensor(DEFAULT_RATIOS))
        self.assertEqual(tuple(outputs["flip_logits"].shape), (2, len(DEFAULT_RATIOS)))

    def test_phase_b2_one_dimensional_state_is_not_erased(self) -> None:
        model = PhaseB2MultiTaskNet(dim=1, hidden_dim=8)
        model.eval()
        ratios = torch.tensor(DEFAULT_RATIOS)
        with torch.no_grad():
            first = model(torch.tensor([[0.0]]), ratios)["flip_logits"]
            second = model(torch.tensor([[1.0]]), ratios)["flip_logits"]
        self.assertFalse(torch.equal(first, second))

    def test_phase_b25_transform_is_fit_on_train_and_reusable(self) -> None:
        rows = [
            {"entropy": float(index), "confidence": 0.5, "position": index / 10}
            for index in range(6)
        ]
        hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        transform = fit_phase_b25_transform(rows, hidden, [0, 1, 2, 3], pca_dim=2)
        uncertainty, hidden_pca = transform_phase_b25_features(rows, hidden, transform)
        self.assertEqual(transform["fit_split"], "train")
        self.assertEqual(transform["fit_row_count"], 4)
        self.assertEqual(tuple(uncertainty.shape), (6, 3))
        self.assertEqual(tuple(hidden_pca.shape), (6, 2))
        self.assertTrue(torch.allclose(uncertainty[:4].mean(dim=0), torch.zeros(3), atol=1e-5))

    def test_phase_b25_variants_produce_action_logits(self) -> None:
        ratios = torch.tensor(DEFAULT_RATIOS)
        uncertainty = torch.zeros(2, 3)
        hidden_pca = torch.zeros(2, 4)
        for variant in (
            "uncertainty_nonlinear",
            "hidden_pca_linear",
            "hidden_pca_nonlinear",
            "uncertainty_hidden_residual",
        ):
            model = PhaseB25ActionNet(variant, hidden_pca_dim=4, model_dim=8)
            logits = model(uncertainty, hidden_pca, ratios)
            self.assertEqual(tuple(logits.shape), (2, len(DEFAULT_RATIOS)))

    def test_phase_b25_boundary_any_flip_metrics(self) -> None:
        rows = [
            {"candidate_flipped": [False, False, True]},
            {"candidate_flipped": [False, False, False]},
        ]
        labels, scores = boundary_any_flip_metrics(rows, [[0.0, 0.2, 0.8], [0.0, 0.1, 0.3]])
        self.assertEqual(labels, [1, 0])
        self.assertEqual(scores, [0.8, 0.3])

    def test_phase_b25b_residual_is_zero_initialized_and_alpha_zero_reproduces_base(self) -> None:
        model = HiddenActionResidual(hidden_pca_dim=4, model_dim=8)
        residual = model(torch.randn(2, 4), torch.tensor(DEFAULT_RATIOS))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        base_logits = torch.randn(2, len(DEFAULT_RATIOS))
        base = combined_risks(base_logits, residual, alpha=0.0)
        combined = combined_risks(base_logits, torch.randn_like(residual), alpha=0.0)
        self.assertEqual(base, combined)

    def test_shared_training_requires_equivalent_action_labels(self) -> None:
        row = {
            "dataset": "gsm8k",
            "id": "x",
            "segment_id": 0,
            "candidate_flipped": [False, True],
            "candidate_unsafe": [False, True],
        }
        validate_equivalent_action_labels([row], [{**row, "target_budget": 0.2}])
        with self.assertRaises(ValueError):
            validate_equivalent_action_labels(
                [row],
                [{**row, "candidate_unsafe": [False, False]}],
            )

    def test_threshold_is_selected_by_controller_budget(self) -> None:
        metadata = {"calibrated_thresholds": {"0.15": 0.3, "0.20": 0.4}}
        self.assertAlmostEqual(threshold_for_budget(metadata, 0.15), 0.3)
        with self.assertRaises(ValueError):
            threshold_for_budget(metadata, 0.10)

    def test_action_risk_loss_is_finite_and_rewards_monotonic_risk(self) -> None:
        logits = torch.zeros(2, len(DEFAULT_RATIOS))
        unsafe_mask = torch.zeros(2, len(DEFAULT_RATIOS))
        unsafe_mask[:, 4:] = 1.0
        loss = action_risk_loss(
            logits,
            unsafe_mask,
            positive_weight=2.0,
            monotonic_weight=1.0,
            ranking_weight=1.0,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_runtime_controller_respects_budget_and_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dim = 11
            model = ActionRiskPolicyNet(dim, hidden_dim=8)
            checkpoint = {
                "model": model.state_dict(),
                "dim": dim,
                "hidden_dim": 8,
                "ratios": DEFAULT_RATIOS,
                "metadata": {
                    "feature_schema": POLICY_FEATURE_SCHEMA,
                    "calibrated_thresholds": {"0.10": 0.5},
                },
            }
            torch.save(checkpoint, root / "policy.pt")
            controller = RaspTrainPolicyController(
                checkpoint_path=str(root / "policy.pt"),
                dataset="gsm8k",
                target_average_ratio=0.1,
                max_new_tokens=128,
                default_max_ratio=0.4,
                early_tokens=64,
                early_max_ratio=0.05,
            )
            ratio = controller.choose_ratio(
                RuntimeObservation(
                    generated_tokens=0,
                    entropy=0.1,
                    confidence=0.9,
                    hidden_state=torch.zeros(1, 4),
                )
            )
            self.assertLessEqual(ratio, 0.05)
            self.assertLessEqual(ratio, 0.1)

    def test_phase_b2_uncertainty_runtime_controller_uses_calibration_and_no_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = PhaseB2MultiTaskNet(dim=3, hidden_dim=8)
            checkpoint = {
                "model": model.state_dict(),
                "metadata": {
                    "schema": "rasp_phase_b2_multitask_v3",
                    "variant": "uncertainty_flip_only",
                    "feature_set": "uncertainty",
                    "model_type": "nonlinear",
                    "dim": 3,
                    "hidden_dim": 8,
                    "seed": 1,
                    "ratios": DEFAULT_RATIOS,
                    "calibration": {"0.10": {"threshold": 1.0}},
                },
            }
            torch.save(checkpoint, root / "policy.pt")
            controller = PhaseB2UncertaintyController(
                checkpoint_path=str(root / "policy.pt"),
                target_average_ratio=0.10,
                max_new_tokens=128,
                policy_horizon_tokens=64,
                early_tokens=64,
                early_max_ratio=0.05,
            )
            ratio = controller.choose_ratio(
                RuntimeObservation(
                    generated_tokens=0,
                    entropy=0.1,
                    confidence=0.9,
                    hidden_state=None,
                )
            )
            self.assertLessEqual(ratio, 0.05)
            self.assertAlmostEqual(float(controller.risk_threshold), 1.0)
            risks = [
                item["predicted_risk"]
                for item in controller.last_decision["candidate_scores"]
            ]
            self.assertEqual(risks, sorted(risks))
            self.assertEqual(
                controller.choose_ratio(
                    RuntimeObservation(
                        generated_tokens=64,
                        entropy=0.1,
                        confidence=0.9,
                        hidden_state=None,
                    )
                ),
                0.0,
            )
            self.assertIn("outside_policy_horizon", controller.last_decision["cap_reasons"])

    def test_stage_probe_split_is_problem_disjoint_and_seeded(self) -> None:
        rows = [
            {
                "dataset": dataset,
                "id": str(problem),
                "stage": ("setup", "reasoning", "final")[segment],
            }
            for dataset in ("gsm8k", "math500")
            for problem in range(20)
            for segment in range(3)
        ]
        first = problem_stage_split(rows, seed=1)
        second = problem_stage_split(rows, seed=2)
        validate_stage_manifest(rows, first, seed=1)
        self.assertNotEqual(first["split_problem_keys"], second["split_problem_keys"])
        split_sets = {
            split: {tuple(key) for key in first["split_problem_keys"][split]}
            for split in ("train", "validation", "test")
        }
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["validation"]))
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["test"]))
        self.assertTrue(split_sets["validation"].isdisjoint(split_sets["test"]))
        self.assertEqual(len(indices_for_stage_split(rows, first, "test")), 18)

    def test_stage_transform_is_train_only_and_variants_have_expected_dims(self) -> None:
        rows = [
            {
                "position": index / 9,
                "entropy": float(index),
                "confidence": 1.0 - index / 10,
            }
            for index in range(10)
        ]
        hidden = torch.arange(60, dtype=torch.float32).reshape(10, 6)
        transform = fit_stage_transform(rows, hidden, list(range(7)), pca_dim=3)
        self.assertEqual(transform["fit_split"], "train")
        self.assertEqual(transform["fit_row_count"], 7)
        self.assertEqual(tuple(transform_stage_features(rows, hidden, transform, "position_only").shape), (10, 1))
        self.assertEqual(tuple(transform_stage_features(rows, hidden, transform, "uncertainty_only").shape), (10, 2))
        self.assertEqual(tuple(transform_stage_features(rows, hidden, transform, "hidden_pca_linear").shape), (10, 3))
        self.assertEqual(tuple(transform_stage_features(rows, hidden, transform, "hidden_uncertainty").shape), (10, 5))

    def test_stage_probe_metrics_include_all_stages(self) -> None:
        metrics = stage_metrics([0, 1, 2, 3, 4], [0, 1, 2, 2, 4])
        self.assertEqual(len(metrics["per_stage_recall"]), 4)
        self.assertEqual(len(metrics["confusion_matrix"]), 4)
        model = StageProbeNet(dim=3, model_type="linear")
        self.assertEqual(tuple(model(torch.zeros(2, 3)).shape), (2, 4))

    def test_operational_stage_classifier_avoids_keyword_false_positives(self) -> None:
        self.assertEqual(classify_operational_stage("Final answer: \\\\boxed{4}", 4, 5), "final")
        self.assertEqual(classify_operational_stage("Step 1: Understand the given information", 0, 5), "setup")
        self.assertEqual(classify_operational_stage("Second month: 3 * 60 = 180", 1, 5), "reasoning")
        self.assertEqual(classify_operational_stage("Therefore, 20 + 4 = 24", 3, 5), "reasoning")
        self.assertEqual(classify_operational_stage("Step 5: Verify both sides are equal", 4, 6), "verification")


if __name__ == "__main__":
    unittest.main()
