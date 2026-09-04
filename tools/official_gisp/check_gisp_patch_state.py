from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check patched official GISP source state.")
    parser.add_argument("--gisp-repo-dir", required=True)
    args = parser.parse_args()

    repo_dir = Path(args.gisp_repo_dir).resolve()
    required_files = {
        "hf_loader": repo_dir / "modules/model/hf.py",
        "data_prune": repo_dir / "modules/data/data_prune.py",
        "system": repo_dir / "modules/system/system.py",
    }
    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        print(f"ERROR: missing official GISP files: {missing}", file=sys.stderr)
        sys.exit(2)

    errors = []
    qwen2_hits = []
    unpatched_mask_hits = []
    for path in sorted(repo_dir.rglob("*.py")):
        try:
            source = _read(path)
        except UnicodeDecodeError:
            continue
        if "Qwen2ForCausalLM" in source:
            qwen2_hits.append(path)
        if "Attention mask should be of size" in source and "RASP-LRM Qwen attention mask shape compatibility patch" not in source:
            unpatched_mask_hits.append(path)
    if qwen2_hits:
        errors.append("Qwen2ForCausalLM remains in: " + ", ".join(str(path) for path in qwen2_hits))
    if unpatched_mask_hits:
        errors.append(
            "attention mask shape checks remain unpatched in: "
            + ", ".join(str(path) for path in unpatched_mask_hits)
        )

    hf_source = _read(required_files["hf_loader"])
    if "AutoModelForCausalLM" not in hf_source:
        errors.append("modules/model/hf.py does not use AutoModelForCausalLM")

    data_source = _read(required_files["data_prune"])
    if "GISP_LOCAL_C4_JSONL" not in data_source:
        errors.append("modules/data/data_prune.py does not contain local C4 JSONL loader patch")

    system_source = _read(required_files["system"])
    for token in (
        "world_size = 1",
        "rank = 0",
        "local_rank = 0",
        "_RaspLrmLegacyRotaryEmbedding",
        "_rasp_lrm_patch_qwen_attention_attrs(model)",
    ):
        if token not in system_source:
            errors.append(f"modules/system/system.py missing patch token: {token}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(3)

    print("official GISP patch state looks ready for Qwen3 + local C4 single-process pruning")


if __name__ == "__main__":
    main()
