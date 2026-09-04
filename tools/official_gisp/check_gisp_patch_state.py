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
    missing_qwen3_norm_hits = []
    missing_output_reshape_hits = []
    missing_return_arity_hits = []
    for path in sorted(repo_dir.rglob("*.py")):
        try:
            source = _read(path)
        except UnicodeDecodeError:
            continue
        if "Qwen2ForCausalLM" in source:
            qwen2_hits.append(path)
        if "Attention mask should be of size" in source and "RASP-LRM Qwen attention mask shape compatibility patch" not in source:
            unpatched_mask_hits.append(path)
        if (
            "query_states = self.q_proj(hidden_states)" in source
            and "value_states = self.v_proj(hidden_states)" in source
            and "RASP-LRM Qwen3 q/k norm compatibility patch" not in source
        ):
            missing_qwen3_norm_hits.append(path)
        if (
            "attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)" in source
            and "RASP-LRM Qwen attention output reshape compatibility patch" not in source
        ):
            missing_output_reshape_hits.append(path)
        if (
            "return attn_output, attn_weights, past_key_value" in source
            and "RASP-LRM Qwen3 attention return arity compatibility patch" not in source
        ):
            missing_return_arity_hits.append(path)
    if qwen2_hits:
        errors.append("Qwen2ForCausalLM remains in: " + ", ".join(str(path) for path in qwen2_hits))
    if unpatched_mask_hits:
        errors.append(
            "attention mask shape checks remain unpatched in: "
            + ", ".join(str(path) for path in unpatched_mask_hits)
        )
    if missing_qwen3_norm_hits:
        errors.append(
            "Qwen attention hooks remain without q/k norm compatibility in: "
            + ", ".join(str(path) for path in missing_qwen3_norm_hits)
        )
    if missing_output_reshape_hits:
        errors.append(
            "Qwen attention hooks remain with hidden_size reshape in: "
            + ", ".join(str(path) for path in missing_output_reshape_hits)
        )
    if missing_return_arity_hits:
        errors.append(
            "Qwen attention hooks remain without Qwen3 return arity compatibility in: "
            + ", ".join(str(path) for path in missing_return_arity_hits)
        )

    hf_source = _read(required_files["hf_loader"])
    if "AutoModelForCausalLM" not in hf_source:
        errors.append("modules/model/hf.py does not use AutoModelForCausalLM")
    for token in (
        "RASP-LRM explicit Qwen3 HF loader compatibility patch",
        "_rasp_lrm_from_pretrained(",
        "Qwen3 checkpoint was not loaded by a Qwen3 model class",
    ):
        if token not in hf_source:
            errors.append(f"modules/model/hf.py missing patch token: {token}")

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
        "module.hidden_size",
        "module.attention_dropout",
        "module.is_causal",
    ):
        if token not in system_source:
            errors.append(f"modules/system/system.py missing patch token: {token}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(3)

    print("official GISP patch state looks ready for Qwen3 + local C4 pruning")


if __name__ == "__main__":
    main()
