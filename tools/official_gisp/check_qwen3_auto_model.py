from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Qwen3 AutoModelForCausalLM mapping without loading weights.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    class_name = type(model).__name__
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
            f"ERROR: expected a Qwen3 model class, got {class_name}. "
            "Do not trust pruning results from this environment.",
            file=sys.stderr,
        )
        sys.exit(4)


if __name__ == "__main__":
    main()
