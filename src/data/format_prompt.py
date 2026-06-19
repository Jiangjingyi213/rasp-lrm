from __future__ import annotations

from typing import Any


def answer_instruction(prompt_config: dict[str, Any] | None = None) -> str:
    prompt_config = prompt_config or {}
    if prompt_config.get("explicit_stage_protocol", False):
        prefix = forced_assistant_prefix(prompt_config)
        prefix_sentence = (
            f"The assistant response will begin with {prefix.strip()}; continue from there. "
            if prefix
            else "Begin your response with <STAGE_SETUP>. "
        )
        return (
            "Solve the problem using exactly these four stage markers, each exactly once and in order:\n"
            "<STAGE_SETUP>\n<STAGE_REASONING>\n<STAGE_VERIFY>\n<STAGE_FINAL>\n"
            "Write the final answer in \\boxed{} inside <STAGE_FINAL>. "
            f"{prefix_sentence}"
            "After the setup section, write <STAGE_REASONING> on its own line. "
            "After the reasoning section, write <STAGE_VERIFY> on its own line. "
            "After the verification section, write <STAGE_FINAL> on its own line. "
            "Do not discuss these marker instructions in the solution. "
            "Do not repeat a stage marker after it has appeared. "
            "Do not write any other <STAGE_...> marker."
        )
    if prompt_config.get("answer_format") == "boxed":
        return "Solve the problem step by step. Put the final answer in \\boxed{}."
    return "Solve the problem step by step. Put the final answer after 'Final answer:'."


def forced_assistant_prefix(prompt_config: dict[str, Any] | None = None) -> str:
    prompt_config = prompt_config or {}
    return str(prompt_config.get("forced_assistant_prefix") or "")


def reasoning_prompt(question: str, prompt_config: dict[str, Any] | None = None) -> str:
    return (
        f"{answer_instruction(prompt_config)}\n\n"
        f"Problem: {question.strip()}\n\n"
        "Reasoning:\n"
    )


def prompt_with_prefix(question: str, prefix: str, prompt_config: dict[str, Any] | None = None) -> str:
    return reasoning_prompt(question, prompt_config) + prefix.strip() + "\n"


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
    return base + prefix


def build_prompt(
    question: str,
    tokenizer=None,
    prompt_config: dict[str, Any] | None = None,
    prefix: str | None = None,
) -> str:
    prompt_config = prompt_config or {}
    if not prompt_config.get("use_chat_template", False):
        return (
            prompt_with_prefix(question, prefix, prompt_config)
            if prefix
            else reasoning_prompt(question, prompt_config)
        )

    user_content = (
        f"{answer_instruction(prompt_config)}\n\n"
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
