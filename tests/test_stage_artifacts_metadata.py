from __future__ import annotations

import unittest

from src.stage_calibration.artifacts import assert_metadata_matches


class StageArtifactsMetadataTest(unittest.TestCase):
    def test_metadata_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "config_hash"):
            assert_metadata_matches(
                {"config_hash": "old", "prompt_hash": "same"},
                {"config_hash": "new", "prompt_hash": "same"},
            )

    def test_ignored_metadata_key_does_not_raise(self) -> None:
        assert_metadata_matches(
            {"config_hash": "old", "prompt_hash": "same"},
            {"config_hash": "new", "prompt_hash": "same"},
            ignored_keys=("config_hash",),
        )

    def test_non_ignored_metadata_key_still_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt_hash"):
            assert_metadata_matches(
                {"config_hash": "old", "prompt_hash": "old"},
                {"config_hash": "new", "prompt_hash": "new"},
                ignored_keys=("config_hash",),
            )


if __name__ == "__main__":
    unittest.main()
