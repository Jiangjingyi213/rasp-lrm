from __future__ import annotations

import argparse
from pathlib import Path


TARGET_RELATIVE_FILES = (
    "modules/model/hf.py",
    "external_code/GISP/modules/model/hf.py",
)


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


def patch_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    patched = source

    patched = patched.replace(
        "custom_package_module.Qwen2ForCausalLM.from_pretrained",
        "AutoModelForCausalLM.from_pretrained",
    )
    patched = patched.replace(
        "Qwen2ForCausalLM.from_pretrained",
        "AutoModelForCausalLM.from_pretrained",
    )

    if patched != source:
        patched = _ensure_auto_import(patched)
        path.write_text(patched, encoding="utf-8")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch official GISP HF loader so Qwen3 uses AutoModelForCausalLM."
    )
    parser.add_argument("--gisp-repo-dir", required=True)
    args = parser.parse_args()

    repo_dir = Path(args.gisp_repo_dir).resolve()
    candidates = [repo_dir / relative for relative in TARGET_RELATIVE_FILES]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(
            "Could not find official GISP HF loader. Checked: "
            + ", ".join(str(path) for path in candidates)
        )

    changed = []
    untouched = []
    for path in existing:
        if patch_file(path):
            changed.append(path)
        else:
            untouched.append(path)

    if changed:
        print("Patched official GISP HF loader for Qwen3:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Official GISP HF loader already appears Qwen3-safe or has no Qwen2 direct loader call.")
    for path in untouched:
        print(f"Checked without changes: {path}")


if __name__ == "__main__":
    main()
