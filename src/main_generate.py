from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from src.data.format_prompt import reasoning_prompt
from src.data.load_gsm8k import load_tasks
from src.metrics.answer_match import answer_match, extract_answer
from src.models.hooks import model_device
from src.models.load_model import load_model_bundle
from src.utils.io import append_jsonl, ensure_dir, read_yaml
from src.utils.seed import set_seed


@torch.no_grad()
def generate_text(bundle, prompt: str, generation_config: dict) -> str:
    tokenizer = bundle.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=generation_config.get("max_input_tokens", 2048))
    inputs = inputs.to(model_device(bundle.model))
    out = bundle.model.generate(
        **inputs,
        max_new_tokens=generation_config.get("max_new_tokens", 512),
        do_sample=generation_config.get("temperature", 0.0) > 0,
        temperature=max(generation_config.get("temperature", 0.0), 1e-5),
        top_p=generation_config.get("top_p", 1.0),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    set_seed(cfg.get("seed", 1))
    output = args.output or cfg["paths"]["trajectories"]
    ensure_dir(cfg["paths"]["run_dir"])
    bundle = load_model_bundle(cfg["model"])
    tasks = load_tasks(cfg["data"])

    for task in tqdm(tasks, desc="generate"):
        prompt = reasoning_prompt(task["question"])
        completion = generate_text(bundle, prompt, cfg.get("generation", {}))
        row = {
            **task,
            "prompt": prompt,
            "completion": completion,
            "prediction": extract_answer(completion),
            "correct": answer_match(completion, task.get("gold", "")),
        }
        append_jsonl(output, row)


if __name__ == "__main__":
    main()
