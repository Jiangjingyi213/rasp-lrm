"""Runtime RASP-Zero components for deployment-aligned experiments."""

__all__ = ["RuntimeMaskedQwen3MLP", "apply_runtime_mlp_masking_qwen3"]


def __getattr__(name: str):
    if name in __all__:
        from src.rasp import mlp_runtime

        return getattr(mlp_runtime, name)
    raise AttributeError(name)
