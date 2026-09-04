from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Qwen3 AutoModelForCausalLM mapping without loading weights.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    try:
        model_class = AutoModelForCausalLM._model_mapping[type(config)]
    except Exception as exc:
        raise RuntimeError(
            f"AutoModelForCausalLM cannot resolve a model class for {args.model} "
            f"with config type {type(config).__name__}."
        ) from exc

    class_name = getattr(model_class, "__name__", str(model_class))
    print(f"{args.model} config model_type={getattr(config, 'model_type', None)} auto_class={class_name}")
    if class_name == "Qwen2ForCausalLM":
        print(
            "ERROR: this transformers environment maps Qwen3 to Qwen2ForCausalLM. "
            "Upgrade transformers before running official GISP.",
            file=sys.stderr,
        )
        sys.exit(3)
    if "Qwen3" not in class_name:
        print(
            f"WARNING: expected a Qwen3 model class, got {class_name}. "
            "Verify this before trusting pruning results.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
