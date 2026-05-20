from __future__ import annotations

import argparse

from src.probes.train_probe import train_probe
from src.utils.io import read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    probe_cfg = cfg.get("probe", {})
    best = train_probe(
        jsonl_path=cfg["paths"]["probe_dataset"],
        hidden_path=cfg["paths"]["probe_hidden_states"],
        output_path=cfg["paths"]["probe_model"],
        epochs=probe_cfg.get("epochs", 20),
        batch_size=probe_cfg.get("batch_size", 64),
        lr=probe_cfg.get("lr", 1e-3),
        val_fraction=probe_cfg.get("val_fraction", 0.2),
        seed=cfg.get("seed", 1),
        split=probe_cfg.get("split", "problem"),
    )
    write_json(cfg["paths"]["probe_metrics"], best)


if __name__ == "__main__":
    main()
