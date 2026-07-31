from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.stage_calibration.artifacts import file_sha256
from src.stage_calibration.mask_bank import (
    add_stage_residual_policies,
    load_mask_bank,
    save_mask_bank,
    validate_mask_bank,
)
from src.utils.io import write_json


def _copy_tree(source: Path, target: Path, force: bool) -> None:
    if target.exists():
        if not force:
            raise FileExistsError(f"Target already exists: {target}; use --force to replace it")
        shutil.rmtree(target)
    shutil.copytree(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse a calibrated 08 bank and add deterministic stage-residual policies."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    source_selected = source_root / "03_selected"
    source_masks = source_root / "04_masks"
    source_bank = source_masks / "mask_bank.pt"
    if not source_selected.is_dir() or not source_bank.is_file():
        raise FileNotFoundError(
            "Source root must contain 03_selected/ and 04_masks/mask_bank.pt: "
            f"{source_root}"
        )

    _copy_tree(source_selected, target_root / "03_selected", args.force)
    _copy_tree(source_masks, target_root / "04_masks", args.force)
    target_bank = target_root / "04_masks" / "mask_bank.pt"
    bank = load_mask_bank(target_bank)
    add_stage_residual_policies(bank)
    validate_mask_bank(bank)
    save_mask_bank(target_bank, bank)
    write_json(
        target_root / "04_masks" / "stage_residual_bank_manifest.json",
        {
            "schema": "stage_residual_bank_reuse_v1",
            "source_root": str(source_root),
            "source_bank_sha256": file_sha256(source_bank),
            "target_bank_sha256": file_sha256(target_bank),
            "reused_calibration": True,
            "new_trajectories_collected": False,
            "new_counterfactual_bank_collected": False,
            "added_policies": ["stage_residual_025", "stage_residual_050"],
        },
    )


if __name__ == "__main__":
    main()
