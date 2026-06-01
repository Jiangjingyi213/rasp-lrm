from __future__ import annotations

from typing import Any


def reasoning_prompt(question: str) -> str:
    return (
        "Solve the problem step by step. Put the final answer after 'Final answer:'.\n\n"
        f"Problem: {question.strip()}\n\n"
        "Reasoning:\n"
    )


def prompt_with_prefix(question: str, prefix: str) -> str:
    return reasoning_prompt(question) + prefix.strip() + "\n"


def build_assistant_continuation_prompt(
    question: str,
    prefix: str,
    tokenizer=None,
    prompt_config: dict[str, Any] | None = None,
) -> str:
    """Build the original assistant prompt followed by its generated prefix.

    This differs from `build_prompt(..., prefix=...)`, which asks the model to
    reconsider a reasoning prefix inside a new user message. Runtime
    counterfactual collection needs the faithful autoregressive form: keep the
    original user prompt and continue directly from the assistant tokens that
    have already been generated.
    """

    base = build_prompt(question, tokenizer, prompt_config)
    return base + prefix.strip()


def build_prompt(
    question: str,
    tokenizer=None,
    prompt_config: dict[str, Any] | None = None,
    prefix: str | None = None,
) -> str:
    prompt_config = prompt_config or {}
    if not prompt_config.get("use_chat_template", False):
        return prompt_with_prefix(question, prefix) if prefix else reasoning_prompt(question)

    user_content = (
        "Solve the problem step by step. Put the final answer after 'Final answer:'.\n\n"
        f"Problem: {question.strip()}"
    )
    if prefix:
        user_content += f"\n\nReasoning so far:\n{prefix.strip()}\n\nContinue the reasoning and give the final answer."
    messages = []
    system = prompt_config.get("system")
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})

    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return user_content + "\n\nReasoning:\n"
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if "enable_thinking" in prompt_config:
        kwargs["enable_thinking"] = bool(prompt_config["enable_thinking"])
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)
