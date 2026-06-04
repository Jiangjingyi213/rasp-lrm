from __future__ import annotations

import unittest

from src.rasp.budget_controller import RuntimeObservation, conservative_ratio_cap


class ActionRouterPolicyTest(unittest.TestCase):
    def test_early_token_cap_applies_before_boundary(self) -> None:
        obs = RuntimeObservation(generated_tokens=32, entropy=0.2, confidence=0.9)
        cap, reasons = conservative_ratio_cap(
            obs,
            default_max_ratio=0.4,
            early_tokens=96,
            early_max_ratio=0.1,
        )
        self.assertEqual(cap, 0.1)
        self.assertEqual(reasons, ["early_tokens"])

    def test_uncertainty_caps_stack_conservatively(self) -> None:
        obs = RuntimeObservation(generated_tokens=128, entropy=1.5, confidence=0.5)
        cap, reasons = conservative_ratio_cap(
            obs,
            default_max_ratio=0.4,
            high_entropy_threshold=1.0,
            high_entropy_max_ratio=0.2,
            low_confidence_threshold=0.7,
            low_confidence_max_ratio=0.1,
        )
        self.assertEqual(cap, 0.1)
        self.assertEqual(reasons, ["high_entropy", "low_confidence"])


if __name__ == "__main__":
    unittest.main()
