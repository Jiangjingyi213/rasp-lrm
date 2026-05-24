# External Baselines

This directory contains only our reproducibility scaffolding for external pruning baselines. Do not commit cloned third-party repositories, external environments, generated pruned models, or large outputs.

The intended workflow is:

1. Push this repository to GitHub.
2. Pull it on the remote server.
3. Clone external repositories into the gitignored `external_repos/` directory.
4. Run compatibility smoke scripts.
5. Record whether each method can load and prune `Qwen/Qwen3-1.7B`.
6. Evaluate any exported pruned model with this repository's own generation/evaluation pipeline.

Current baseline candidates:

- `FLAP`: zero-shot/static structured pruning baseline.
- `LLM-Pruner`: static structured pruning baseline, possibly with recovery.
- `GRIFFIN`: dynamic/prompt-conditioned FFN pruning baseline candidate.
- `IFPruning`: pending exact public repository confirmation.

The main caution is that external methods may primarily support LLaMA-family models. Qwen3 compatibility must be verified before treating a method as an executable baseline.
