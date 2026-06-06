from __future__ import annotations

from typing import Any


def summarize_runtime_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Need at least one runtime result row")
    total_tokens = sum(int(row["runtime"]["generated_tokens"]) for row in rows)
    total_decode_seconds = sum(float(row["runtime"]["decode_seconds"]) for row in rows)
    total_seconds = sum(float(row["runtime"]["total_seconds"]) for row in rows)
    average_ratio = sum(
        float(row["runtime"]["runtime_mlp"]["average_decode_pruning_ratio"]) * int(row["runtime"]["generated_tokens"])
        for row in rows
    ) / max(total_tokens, 1)
    return {
        "examples": len(rows),
        "correct": sum(int(bool(row["correct"])) for row in rows),
        "accuracy": sum(int(bool(row["correct"])) for row in rows) / len(rows),
        "generated_tokens": total_tokens,
        "average_generated_tokens": total_tokens / len(rows),
        "total_seconds": total_seconds,
        "decode_seconds": total_decode_seconds,
        "decode_tokens_per_second": total_tokens / total_decode_seconds if total_decode_seconds > 0 else None,
        "average_decode_pruning_ratio": average_ratio,
        "logical_mlp_channel_mask_ratio": average_ratio,
        "idealized_mlp_decode_flops_reduction_if_physically_pruned": average_ratio,
        "theoretical_mlp_decode_flops_reduction": average_ratio,
        "theoretical_mlp_decode_activated_parameter_reduction": average_ratio,
        "efficiency_metric_note": (
            "Logical masking keeps dense projections in the current backend; "
            "the ratio is an idealized proxy, not measured compute reduction."
        ),
        "real_speedup_claimed": False,
    }
