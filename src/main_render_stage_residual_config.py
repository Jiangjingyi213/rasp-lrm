from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the output-aware config after the residual gate.")
    parser.add_argument("--template", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selection = read_json(args.selection).get("selection", {})
    if selection.get("status") != "passed":
        raise RuntimeError("Cannot render output-aware config before the residual selection gate passes")
    policy = str(selection["prior_policy"])
    text = Path(args.template).read_text(encoding="utf-8")
    rendered = text.replace("__SELECTED_RESIDUAL_POLICY__", policy)
    if "__SELECTED_RESIDUAL_POLICY__" in rendered:
        raise RuntimeError("Unresolved residual-policy placeholder")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
