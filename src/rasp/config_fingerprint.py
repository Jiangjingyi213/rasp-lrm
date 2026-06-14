from __future__ import annotations

import hashlib
import json
from typing import Any


def config_fingerprint(config: dict[str, Any], sections: tuple[str, ...]) -> str:
    payload = {name: config.get(name) for name in sections}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
