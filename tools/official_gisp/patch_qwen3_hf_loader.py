from __future__ import annotations

import argparse
import re
from pathlib import Path


HF_LOADER_RELATIVE_FILES = (
    "modules/model/hf.py",
    "external_code/GISP/modules/model/hf.py",
)

DATA_PRUNE_RELATIVE_FILES = (
    "modules/data/data_prune.py",
    "external_code/GISP/modules/data/data_prune.py",
)


LOCAL_C4_MARKER = "# RASP-LRM local C4 JSONL patch for offline official GISP runs"

LOCAL_C4_HELPER = r'''
# RASP-LRM local C4 JSONL patch for offline official GISP runs
def _rasp_lrm_load_local_c4_jsonl(path, nsamples, seed, seqlen, tokenizer):
    import json
    import random
    import torch

    rng = random.Random(seed)
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if text:
                records.append(text)

    if not records:
        raise RuntimeError(f"Local C4 calibration JSONL has no usable text rows: {path}")

    encoded = []
    for text in records:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        if ids.shape[1] >= seqlen:
            encoded.append(ids)

    if not encoded:
        all_ids = [
            tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
            for text in records
        ]
        flat = torch.cat(all_ids, dim=1)
        if flat.shape[1] < seqlen:
            raise RuntimeError(
                f"Local C4 calibration JSONL is too short after tokenization: {path}; "
                f"need at least {seqlen} tokens, got {flat.shape[1]}"
            )
        encoded = [flat]

    trainloader = []
    for _ in range(nsamples):
        ids = rng.choice(encoded)
        max_start = ids.shape[1] - seqlen
        start = rng.randint(0, max_start) if max_start > 0 else 0
        inp = ids[:, start : start + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None

'''

def _ensure_auto_import(source: str) -> str:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("from transformers import "):
            continue
        imported = [
            part.strip()
            for part in line.split("import", 1)[1].split(",")
            if part.strip() != "Qwen2ForCausalLM"
        ]
        if "AutoModelForCausalLM" not in imported:
            imported.append("AutoModelForCausalLM")
        lines[index] = "from transformers import " + ", ".join(imported)
        return "\n".join(lines) + ("\n" if source.endswith("\n") else "")

    insert_at = 0
    for index, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = index + 1
    lines.insert(insert_at, "from transformers import AutoModelForCausalLM")
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def _force_auto_model_calls(source: str) -> str:
    patched = re.sub(r"\bcustom_package_module\.Qwen2ForCausalLM\b", "AutoModelForCausalLM", source)
    patched = re.sub(r"\bQwen2ForCausalLM\b", "AutoModelForCausalLM", patched)
    patched = re.sub(r"\bAutoModelForCausalLM\s*,\s*AutoModelForCausalLM\b", "AutoModelForCausalLM", patched)
    return patched


def _ensure_trust_remote_code_for_auto_model(source: str) -> str:
    lines = source.splitlines()
    output = []
    pending_call = False
    pending_has_trust = False

    for line in lines:
        stripped = line.strip()
        if "AutoModelForCausalLM.from_pretrained(" in line:
            pending_call = True
            pending_has_trust = "trust_remote_code" in line

        if pending_call and "trust_remote_code" in line:
            pending_has_trust = True

        if pending_call and stripped.startswith(")") and not pending_has_trust:
            indent = line[: len(line) - len(line.lstrip())]
            output.append(f"{indent}    trust_remote_code=True,")
            pending_call = False
            pending_has_trust = False
        elif pending_call and stripped.startswith(")") and pending_has_trust:
            pending_call = False
            pending_has_trust = False

        output.append(line)

    return "\n".join(output) + ("\n" if source.endswith("\n") else "")


def patch_hf_loader(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    patched = _force_auto_model_calls(source)
    patched = _ensure_trust_remote_code_for_auto_model(patched)

    if patched != source:
        patched = _ensure_auto_import(patched)
        path.write_text(patched, encoding="utf-8")
        return True
    return False


def patch_data_prune(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if LOCAL_C4_MARKER in source:
        return False

    match = re.search(r"^def get_c4\(([^)]*)\):", source, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find get_c4(...) in {path}")

    wrapper = (
        "def get_c4(nsamples, seed, seqlen, tokenizer):\n"
        "    import os\n"
        "    local_path = os.environ.get(\"GISP_LOCAL_C4_JSONL\") or os.environ.get(\"C4_CALIBRATION_PATH\")\n"
        "    if local_path:\n"
        "        print(f\"loading local C4 calibration data from {local_path}\")\n"
        "        return _rasp_lrm_load_local_c4_jsonl(local_path, nsamples, seed, seqlen, tokenizer)\n"
        "    return _rasp_lrm_original_get_c4(nsamples, seed, seqlen, tokenizer)\n\n"
        f"def _rasp_lrm_original_get_c4({match.group(1)}):"
    )
    patched = source[: match.start()] + LOCAL_C4_HELPER + wrapper + source[match.end() :]
    path.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch official GISP for Qwen3 and local C4 calibration."
    )
    parser.add_argument("--gisp-repo-dir", required=True)
    args = parser.parse_args()

    repo_dir = Path(args.gisp_repo_dir).resolve()
    hf_candidates = [repo_dir / relative for relative in HF_LOADER_RELATIVE_FILES]
    hf_existing = [path for path in hf_candidates if path.exists()]
    if not hf_existing:
        raise FileNotFoundError(
            "Could not find official GISP HF loader. Checked: "
            + ", ".join(str(path) for path in hf_candidates)
        )

    data_candidates = [repo_dir / relative for relative in DATA_PRUNE_RELATIVE_FILES]
    data_existing = [path for path in data_candidates if path.exists()]
    if not data_existing:
        raise FileNotFoundError(
            "Could not find official GISP data_prune.py. Checked: "
            + ", ".join(str(path) for path in data_candidates)
        )

    hf_changed = []
    hf_untouched = []
    for path in hf_existing:
        if patch_hf_loader(path):
            hf_changed.append(path)
        else:
            hf_untouched.append(path)

    data_changed = []
    data_untouched = []
    for path in data_existing:
        if patch_data_prune(path):
            data_changed.append(path)
        else:
            data_untouched.append(path)

    if hf_changed:
        print("Patched official GISP HF loader for Qwen3:")
        for path in hf_changed:
            print(f"  {path}")
    else:
        print("Official GISP HF loader already appears Qwen3-safe or has no Qwen2 direct loader call.")
    for path in hf_untouched:
        print(f"Checked without changes: {path}")

    if data_changed:
        print("Patched official GISP C4 loader for local JSONL calibration:")
        for path in data_changed:
            print(f"  {path}")
    else:
        print("Official GISP C4 loader already has local JSONL calibration patch.")
    for path in data_untouched:
        print(f"Checked without changes: {path}")


if __name__ == "__main__":
    main()
