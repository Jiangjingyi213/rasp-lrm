from __future__ import annotations

import unittest

import torch
from torch import nn

from src.data.format_prompt import build_assistant_continuation_prompt
from src.pruning.mlp_pruner import mlp_intermediate_channel_mask
from src.rasp.activation_ranker import keep_mask_from_ranking, rank_intermediate_neurons
from src.rasp.mlp_runtime import RuntimeMaskedQwen3MLP
from src.main_collect_aligned_window_bank import boundary_positions, token_divergence
from src.segmentation.rule_segmenter import segment_text


class FakeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)
        self.act_fn = nn.Identity()
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.eye(4))
            self.up_proj.weight.copy_(torch.eye(4))
            self.down_proj.weight.copy_(torch.eye(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = FakeMlp()


class FakeInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer()])


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeInnerModel()


class RaspZeroRuntimeTest(unittest.TestCase):
    def test_assistant_continuation_prompt_appends_prefix_directly(self) -> None:
        prompt = build_assistant_continuation_prompt("What is 1 + 1?", "\nStep 1: add the values.\n")
        self.assertTrue(prompt.endswith("Reasoning:\n\nStep 1: add the values.\n"))
        self.assertNotIn("Reasoning so far", prompt)

    def test_segmenter_recognizes_markdown_step_headings(self) -> None:
        text = (
            "We need to calculate the result.\n\n"
            "### **Step 1: First calculation**\n\nCompute one value.\n\n"
            "**Step 2: Final calculation**\n\nCompute the answer.\n\n"
            "**Final answer:** 3"
        )
        segments = segment_text(text, min_chars=10)
        self.assertEqual(len(segments), 3)
        self.assertIn("Step 1", segments[0]["text"])
        self.assertTrue(any("Step 2" in str(segment["text"]) for segment in segments))
        self.assertEqual(segments[-1]["segment_type"], "final")
        self.assertTrue(str(segments[-1]["text"]).startswith("**Final answer:**"))

    def test_short_final_answer_is_not_merged_with_short_previous_segment(self) -> None:
        segments = segment_text("Work.\n\nFinal answer: 3", min_chars=24)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[-1]["segment_type"], "final")
        self.assertEqual(segments[-1]["text"], "Final answer: 3")

    def test_activation_ranker_builds_nested_masks(self) -> None:
        states = torch.tensor([[[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]]])
        ranking = rank_intermediate_neurons(states)
        light = keep_mask_from_ranking(ranking, 0.25)
        heavy = keep_mask_from_ranking(ranking, 0.50)
        self.assertEqual(int(light.sum()), 3)
        self.assertEqual(int(heavy.sum()), 2)
        self.assertTrue(torch.all(~heavy | light))

    def test_runtime_wrapper_ratio_zero_matches_dense(self) -> None:
        dense = FakeMlp()
        wrapped = RuntimeMaskedQwen3MLP(dense, ratios=[0.25, 0.50])
        prefix = torch.tensor([[[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]]])
        decode = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        self.assertTrue(torch.allclose(wrapped(prefix), dense(prefix)))
        wrapped.set_ratio(0.0)
        self.assertTrue(torch.allclose(wrapped(decode), dense(decode)))

    def test_runtime_wrapper_masks_only_decode(self) -> None:
        dense = FakeMlp()
        wrapped = RuntimeMaskedQwen3MLP(dense, ratios=[0.50])
        prefix = torch.tensor([[[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]]])
        decode = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        self.assertTrue(torch.allclose(wrapped(prefix), dense(prefix)))
        wrapped.set_ratio(0.50)
        output = wrapped(decode)
        self.assertTrue(torch.allclose(output, torch.tensor([[[1.0, 4.0, 0.0, 0.0]]])))

    def test_counterfactual_context_keeps_prefill_dense_and_masks_decode(self) -> None:
        model = FakeModel()
        mlp = model.model.layers[0].mlp
        prefix = torch.tensor([[[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]]])
        decode = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        with mlp_intermediate_channel_mask(model, [0], 0.50):
            self.assertTrue(torch.allclose(mlp(prefix), torch.tensor([[[16.0, 9.0, 4.0, 1.0], [16.0, 9.0, 4.0, 1.0]]])))
            self.assertTrue(torch.allclose(mlp(decode), torch.tensor([[[1.0, 4.0, 0.0, 0.0]]])))

    def test_aligned_bank_uses_fixed_token_boundaries(self) -> None:
        self.assertEqual(boundary_positions(50, 16, None), [0, 16, 32, 48])
        self.assertEqual(boundary_positions(50, 16, 2), [0, 16])
        self.assertAlmostEqual(token_divergence([1, 2, 3], [1, 4, 3]), 1 / 3)


if __name__ == "__main__":
    unittest.main()
