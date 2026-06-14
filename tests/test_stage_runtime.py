from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.rasp.stage_probe import StageProbeNet
from src.rasp.stage_runtime import RuntimeStageProbe


def standardizer(dim: int) -> dict[str, torch.Tensor]:
    return {"mean": torch.zeros(dim), "scale": torch.ones(dim)}


class RuntimeStageProbeTest(unittest.TestCase):
    def test_rejects_position_checkpoint_for_causal_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "position.pt"
            torch.save(
                {
                    "metadata": {"variant": "position_only"},
                    "transform": {},
                },
                path,
            )
            with self.assertRaises(ValueError):
                RuntimeStageProbe(str(path), 0.8, require_causal_features=True)

    def test_low_confidence_hidden_prediction_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hidden.pt"
            model = StageProbeNet(1, "linear", 4)
            for value in model.parameters():
                torch.nn.init.zeros_(value)
            torch.save(
                {
                    "metadata": {
                        "variant": "hidden_pca_linear",
                        "dim": 1,
                        "model_type": "linear",
                        "model_dim": 4,
                    },
                    "transform": {
                        "position": standardizer(1),
                        "uncertainty": standardizer(2),
                        "hidden": standardizer(2),
                        "hidden_projection": torch.tensor([[1.0], [0.0]]),
                    },
                    "model": model.state_dict(),
                },
                path,
            )
            probe = RuntimeStageProbe(
                str(path),
                0.8,
                confidence_threshold=0.8,
                require_causal_features=True,
            )
            result = probe.classify(
                hidden_state=torch.tensor([[1.0, 2.0]]),
                entropy=1.0,
                confidence=0.5,
                position=0.25,
                recent_text="continue solving",
                boundary_index=1,
                num_boundaries=4,
            )
            self.assertEqual(result["trusted_stage"], "unknown")
            self.assertTrue(result["stage_probe_causal"])
            causal_last_boundary = probe.classify(
                hidden_state=torch.tensor([[1.0, 2.0]]),
                entropy=1.0,
                confidence=0.5,
                position=0.99,
                recent_text="we now have an answer candidate",
                boundary_index=3,
                num_boundaries=4,
            )
            self.assertNotEqual(
                causal_last_boundary["stage_source"],
                "explicit_verification_rule",
            )


if __name__ == "__main__":
    unittest.main()
