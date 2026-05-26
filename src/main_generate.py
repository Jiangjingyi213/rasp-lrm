from __future__ import annotations

import argparse
import random

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.baselines.flap_mlp_qwen3 import apply_flap_mlp_pruning_qwen3, summary_to_dict
from src.data.format_prompt import build_prompt
from src.data.load_gsm8k import load_tasks
from src.metrics.answer_match import answer_match, extract_answer
from src.models.hooks import model_device
from src.models.load_model import load_model_bundle
from src.utils.io import append_jsonl, ensure_dir, read_yaml, write_json
from src.utils.seed import set_seed


DEFAULT_STOP_STRINGS = (
    "\nHuman:",
    "\nUser:",
    "\nAssistant:",
    "\nProblem:",
    "Human:",
    "User:",
)


def truncate_completion(text: str, stop_strings: list[str] | tuple[str, ...] = DEFAULT_STOP_STRINGS) -> str:
    cut = len(text)
    for stop in stop_strings:
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def build_flap_calibration_texts(bundle, tasks: list[dict], cfg: dict) -> list[str]:
    model_cfg = cfg["model"]
    source = model_cfg.get("flap_calibration_dataset", "wikitext2")
    n = int(model_cfg.get("flap_calibration_samples", 32))
    max_input_tokens = int(cfg.get("generation", {}).get("max_input_tokens", 2048))
    if source == "task":
        return [build_prompt(task["question"], bundle.tokenizer, cfg.get("prompt", {})) for task in tasks[:n]]
    if source == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = " ".join(row["text"].strip() for row in dataset if row.get("text", "").strip())
        token_ids = bundle.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if token_ids.numel() <= max_input_tokens:
            return [text]
        rng = random.Random(int(cfg.get("seed", 1)))
        chunks = []
        for _ in range(n):
            start = rng.randint(0, int(token_ids.numel()) - max_input_tokens - 1)
            chunk_ids = token_ids[start : start + max_input_tokens]
            chunks.append(bundle.tokenizer.decode(chunk_ids, skip_special_tokens=True))
        return chunks
    raise ValueError(f"Unsupported FLAP calibration dataset: {source}")


@torch.no_grad()
def generate_text(bundle, prompt: str, generation_config: dict) -> str:
    tokenizer = bundle.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=generation_config.get("max_input_tokens", 2048))
    inputs = inputs.to(model_device(bundle.model))
    do_sample = generation_config.get("temperature", 0.0) > 0
    generate_kwargs = {
        "max_new_tokens": generation_config.get("max_new_tokens", 512),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = generation_config.get("temperature", 0.7)
        generate_kwargs["top_p"] = generation_config.get("top_p", 1.0)
    out = bundle.model.generate(
        **inputs,
        **generate_kwargs,
    )
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return truncate_completion(completion, generation_config.get("stop_strings", DEFAULT_STOP_STRINGS))


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
    if cfg["model"].get("adapter") == "flap_mlp_qwen3":
        calibration_texts = build_flap_calibration_texts(bundle, tasks, cfg)
        summary = apply_flap_mlp_pruning_qwen3(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            calibration_texts=calibration_texts,
            ratio=float(cfg["model"].get("flap_pruning_ratio", cfg["model"].get("pruning_ratio", 0.2))),
            calibration_dataset=cfg["model"].get("flap_calibration_dataset", "wikitext2"),
            metric=cfg["model"].get("flap_metric", "WIFV"),
            structure=cfg["model"].get("flap_structure", "AL-AM"),
            calibration_samples=int(cfg["model"].get("flap_calibration_samples", 32)),
            max_input_tokens=int(cfg.get("generation", {}).get("max_input_tokens", 2048)),
            layers=cfg["model"].get("flap_layers") or cfg["model"].get("pruning_layers"),
            bias_compensation=bool(cfg["model"].get("flap_bias_compensation", True)),
        )
        write_json(
            cfg["paths"].get("flap_mlp_summary", f'{cfg["paths"]["run_dir"]}/00_flap_mlp_summary.json'),
            summary_to_dict(summary),
        )

    for task in tqdm(tasks, desc="generate"):
        prompt = build_prompt(task["question"], bundle.tokenizer, cfg.get("prompt", {}))
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
