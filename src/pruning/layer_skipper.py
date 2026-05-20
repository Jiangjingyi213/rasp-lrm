from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from src.models.hooks import skip_layers


@contextmanager
def layer_skip(model, layer_ids: list[int]) -> Iterator[None]:
    with skip_layers(model, layer_ids):
        yield
