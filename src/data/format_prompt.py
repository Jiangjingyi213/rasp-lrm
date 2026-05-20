from __future__ import annotations


def reasoning_prompt(question: str) -> str:
    return (
        "Solve the problem step by step. Put the final answer after 'Final answer:'.\n\n"
        f"Problem: {question.strip()}\n\n"
        "Reasoning:\n"
    )


def prompt_with_prefix(question: str, prefix: str) -> str:
    return reasoning_prompt(question) + prefix.strip() + "\n"
