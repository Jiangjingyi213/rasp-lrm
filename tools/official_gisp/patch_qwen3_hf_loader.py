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

SYSTEM_RELATIVE_FILES = (
    "modules/system/system.py",
    "external_code/GISP/modules/system/system.py",
)


LOCAL_C4_MARKER = "# RASP-LRM local C4 JSONL patch for offline official GISP runs"
ATTENTION_ATTR_MARKER = "# RASP-LRM Qwen attention attribute compatibility patch"
ATTENTION_MASK_MARKER = "# RASP-LRM Qwen attention mask shape compatibility patch"
QWEN3_LOADER_MARKER = "# RASP-LRM explicit Qwen3 HF loader compatibility patch"

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

ATTENTION_ATTR_HELPER = r'''
# RASP-LRM Qwen attention attribute compatibility patch
rank = 0
local_rank = 0
world_size = 1


class _RaspLrmLegacyRotaryEmbedding:
    def __init__(self, dim, max_position_embeddings=131072, base=10000.0):
        import torch

        self.dim = int(dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.inv_freq = inv_freq

    def __call__(self, x, seq_len=None):
        import torch

        seq_len = int(seq_len or x.shape[-2])
        inv_freq = self.inv_freq.to(device=x.device)
        t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def _rasp_lrm_patch_qwen_attention_attrs(model):
    for layer_index, module in enumerate(model.modules()):
        if not all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")):
            continue
        head_dim = getattr(module, "head_dim", None)
        config = getattr(module, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if head_dim is None:
            num_attention_heads = getattr(config, "num_attention_heads", None)
            if hidden_size and num_attention_heads:
                head_dim = int(hidden_size) // int(num_attention_heads)
        if not head_dim:
            continue
        if hidden_size is None:
            hidden_size = int(getattr(module.q_proj, "out_features", 0))
        if not hasattr(module, "hidden_size") and hidden_size:
            module.hidden_size = int(hidden_size)
        q_out = int(getattr(module.q_proj, "out_features", 0))
        k_out = int(getattr(module.k_proj, "out_features", 0))
        if not hasattr(module, "num_heads") and q_out:
            module.num_heads = max(1, q_out // int(head_dim))
        if not hasattr(module, "num_key_value_heads") and k_out:
            module.num_key_value_heads = max(1, k_out // int(head_dim))
        if not hasattr(module, "num_key_value_groups") and hasattr(module, "num_heads"):
            kv_heads = int(getattr(module, "num_key_value_heads", module.num_heads))
            module.num_key_value_groups = max(1, int(module.num_heads) // max(1, kv_heads))
        if not hasattr(module, "rotary_emb"):
            max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
            rope_theta = getattr(config, "rope_theta", 10000.0)
            module.rotary_emb = _RaspLrmLegacyRotaryEmbedding(
                int(head_dim),
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
            )
        if not hasattr(module, "attention_dropout"):
            module.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        if not hasattr(module, "layer_idx"):
            module.layer_idx = getattr(module, "layer_idx", layer_index)
        if not hasattr(module, "is_causal"):
            module.is_causal = True

'''


QWEN3_LOADER_HELPER = r'''
# RASP-LRM explicit Qwen3 HF loader compatibility patch
def _rasp_lrm_from_pretrained(model_name_or_path, *args, **kwargs):
    name = str(model_name_or_path).lower()
    kwargs.setdefault("trust_remote_code", True)
    loader = AutoModelForCausalLM
    if "qwen3" in name:
        try:
            from transformers import Qwen3ForCausalLM

            loader = Qwen3ForCausalLM
        except Exception:
            loader = AutoModelForCausalLM
    model = loader.from_pretrained(model_name_or_path, *args, **kwargs)
    class_name = type(model).__name__
    print(f"RASP-LRM loaded HF class: {class_name}", flush=True)
    if "qwen3" in name and "Qwen3" not in class_name:
        raise RuntimeError(
            "Qwen3 checkpoint was not loaded by a Qwen3 model class. "
            f"Got {class_name}; refusing to run GISP because pruning would be invalid."
        )
    return model

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
    patched = re.sub(
        r"from\s+transformers\.models\.qwen2\.[^\n]+\s+import\s+(Qwen2ForCausalLM|AutoModelForCausalLM)",
        "from transformers import AutoModelForCausalLM",
        source,
    )
    patched = re.sub(r"\bcustom_package_module\.Qwen2ForCausalLM\b", "AutoModelForCausalLM", patched)
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


def _ensure_qwen3_loader_helper(source: str) -> str:
    if QWEN3_LOADER_MARKER in source:
        return source
    lines = source.splitlines()
    insert_at = 0
    for index, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = index + 1
    lines.insert(insert_at, QWEN3_LOADER_HELPER.strip("\n"))
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def _route_auto_model_calls_through_qwen3_helper(source: str) -> str:
    return re.sub(
        r"(?<![\w.])AutoModelForCausalLM\.from_pretrained\(",
        "_rasp_lrm_from_pretrained(",
        source,
    )


def patch_hf_loader(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    patched = _force_auto_model_calls(source)
    patched = _ensure_trust_remote_code_for_auto_model(patched)
    patched = _ensure_auto_import(patched)
    patched = _ensure_qwen3_loader_helper(patched)
    patched = _route_auto_model_calls_through_qwen3_helper(patched)

    if patched != source:
        path.write_text(patched, encoding="utf-8")
        return True
    return False


def patch_python_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    interesting_tokens = (
        "Qwen2ForCausalLM",
        "Attention mask should be of size",
        "query_states = self.q_proj(hidden_states)",
    )
    if not any(token in source for token in interesting_tokens):
        return False
    needs_auto_import = "Qwen2ForCausalLM" in source
    patched = _force_auto_model_calls(source)
    patched = _ensure_trust_remote_code_for_auto_model(patched)
    patched = _patch_qwen3_norms_in_attention_hooks(patched)
    patched = _patch_attention_mask_shape_check(patched)
    if needs_auto_import:
        patched = _ensure_auto_import(patched)
    if patched != source:
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


def _patch_attention_mask_shape_check(source: str) -> str:
    if "Attention mask should be of size" not in source:
        return source

    lines = source.splitlines()
    output = []
    for line in lines:
        stripped = line.strip()
        if (
            "attention_mask.size() !=" in stripped
            and "(bsz, 1, q_len, kv_seq_len)" in stripped
            and (not output or ATTENTION_MASK_MARKER not in "\n".join(output[-16:]))
        ):
            indent = line[: len(line) - len(line.lstrip())]
            output.extend(
                [
                    f"{indent}{ATTENTION_MASK_MARKER}",
                    f"{indent}if attention_mask.size(-1) != kv_seq_len:",
                    f"{indent}    if attention_mask.size(-1) > kv_seq_len:",
                    f"{indent}        attention_mask = attention_mask[..., :kv_seq_len]",
                    f"{indent}    else:",
                    f"{indent}        attention_mask = torch.nn.functional.pad(",
                    f"{indent}            attention_mask, (0, kv_seq_len - attention_mask.size(-1))",
                    f"{indent}        )",
                    f"{indent}if attention_mask.size(-2) != q_len:",
                    f"{indent}    if attention_mask.size(-2) > q_len:",
                    f"{indent}        attention_mask = attention_mask[..., -q_len:, :]",
                    f"{indent}    else:",
                    f"{indent}        attention_mask = torch.nn.functional.pad(",
                    f"{indent}            attention_mask, (0, 0, q_len - attention_mask.size(-2), 0)",
                    f"{indent}        )",
                ]
            )
        output.append(line)

    return "\n".join(output) + ("\n" if source.endswith("\n") else "")


QWEN3_NORM_MARKER = "# RASP-LRM Qwen3 q/k norm compatibility patch"


def _patch_qwen3_norms_in_attention_hooks(source: str) -> str:
    if "query_states = self.q_proj(hidden_states)" not in source:
        return source

    lines = source.splitlines()
    output = []
    inserted_count = 0
    for index, line in enumerate(lines):
        output.append(line)
        if "value_states = self.v_proj(hidden_states)" not in line:
            continue
        lookahead = "\n".join(lines[index + 1 : index + 8])
        if QWEN3_NORM_MARKER in lookahead:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        output.extend(
            [
                f"{indent}{QWEN3_NORM_MARKER}",
                f"{indent}if hasattr(self, \"q_norm\"):",
                f"{indent}    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)",
                f"{indent}    query_states = self.q_norm(query_states).reshape(bsz, q_len, -1)",
                f"{indent}if hasattr(self, \"k_norm\"):",
                f"{indent}    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)",
                f"{indent}    key_states = self.k_norm(key_states).reshape(bsz, q_len, -1)",
            ]
        )
        inserted_count += 1

    if inserted_count == 0:
        return source
    return "\n".join(output) + ("\n" if source.endswith("\n") else "")


def patch_system(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    patched = source
    if ATTENTION_ATTR_MARKER in patched:
        start = patched.index(ATTENTION_ATTR_MARKER)
        match = re.search(r"\ndef\s+(?!_rasp_lrm_|_RaspLrm)\w+", patched[start:])
        end = start + match.start() if match else len(patched)
        patched = patched[:start] + ATTENTION_ATTR_HELPER.strip("\n") + "\n" + patched[end:]
    else:
        insert_at = 0
        lines = patched.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = index + 1
        lines.insert(insert_at, ATTENTION_ATTR_HELPER.strip("\n"))
        patched = "\n".join(lines) + ("\n" if source.endswith("\n") else "")

    lines = patched.splitlines()
    output = []
    for index, line in enumerate(lines):
        output.append(line)
        if line.strip() == "model.to(device)":
            indent = line[: len(line) - len(line.lstrip())]
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            if "_rasp_lrm_patch_qwen_attention_attrs(model)" not in next_line:
                output.append(f"{indent}_rasp_lrm_patch_qwen_attention_attrs(model)")
    patched = "\n".join(output) + ("\n" if source.endswith("\n") else "")

    if patched != source:
        path.write_text(patched, encoding="utf-8")
        return True
    return False


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

    system_candidates = [repo_dir / relative for relative in SYSTEM_RELATIVE_FILES]
    system_existing = [path for path in system_candidates if path.exists()]
    if not system_existing:
        raise FileNotFoundError(
            "Could not find official GISP system.py. Checked: "
            + ", ".join(str(path) for path in system_candidates)
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

    system_changed = []
    system_untouched = []
    for path in system_existing:
        if patch_system(path):
            system_changed.append(path)
        else:
            system_untouched.append(path)

    recursive_changed = []
    for path in sorted(repo_dir.rglob("*.py")):
        if path in hf_existing:
            continue
        if patch_python_file(path):
            recursive_changed.append(path)

    remaining_qwen2 = []
    for path in sorted(repo_dir.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "Qwen2ForCausalLM" in source:
            remaining_qwen2.append(path)

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

    if recursive_changed:
        print("Patched additional official GISP Python files for Qwen3 compatibility:")
        for path in recursive_changed:
            print(f"  {path}")
    if system_changed:
        print("Patched official GISP system setup for Qwen attention attributes:")
        for path in system_changed:
            print(f"  {path}")
    else:
        print("Official GISP system setup already has Qwen attention attribute patch.")
    for path in system_untouched:
        print(f"Checked without changes: {path}")
    if remaining_qwen2:
        raise RuntimeError(
            "Qwen2ForCausalLM still remains in official GISP Python files after patch: "
            + ", ".join(str(path) for path in remaining_qwen2)
        )


if __name__ == "__main__":
    main()
