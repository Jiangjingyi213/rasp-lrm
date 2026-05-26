# Qwen3 FLAP / LLM-Pruner Porting Report

Date: 2026-05-26

This note records what the original repositories actually implement and what must be changed before reporting FLAP / LLM-Pruner as formal Qwen3 baselines.

## Source Repositories

- FLAP: `external_repos/FLAP`, commit `3bb57db3449dd2fa04a5c2192de80e87e33be2b1`
- LLM-Pruner: `external_repos/LLM-Pruner`, commit `128a07d977f9b205d60ab14cfbc6a78f8a8e39d2`

The external repositories are used as reference code and should remain outside git-tracked project code.

## FLAP: What The Official Code Does

Core files:

- `external_repos/FLAP/main.py`
- `external_repos/FLAP/lib/prune.py`
- `external_repos/FLAP/lib/layerwrapper.py`
- `external_repos/FLAP/lib/data.py`

Official loading path:

- `main.py` imports `models.hf_llama.modeling_llama.LlamaForCausalLM`.
- The code is written around `model.model.layers[i].self_attn` and `model.model.layers[i].mlp`.
- The shipped LLaMA implementation adds bias to `self_attn.o_proj` and `mlp.down_proj` for bias compensation.

Official pruning flow:

1. Load calibration data from Wikitext2 by default.
2. `prepare_calibration_input` catches layer-0 inputs and stores a fixed calibration tensor of shape `(nsamples, seqlen, hidden_size)`.
3. For each transformer layer, wrap:
   - `self_attn.o_proj`
   - `mlp.down_proj`
4. Use `BiasGPT.add_batch` to collect input statistics for these output projections:
   - baseline mean input
   - input fluctuation, or input norm depending on metric
5. Compute one of the official metrics:
   - `IFV`: input feature variance
   - `WIFV`: input feature variance weighted by output projection weight norm
   - `WIFN`: weighted input feature norm
6. Convert metrics to attention-head and MLP-channel masks according to structure:
   - `UL-UM`: uniform layer, uniform module
   - `UL-MM`: uniform layer, mixed module
   - `AL-MM`: adaptive layer, mixed module
   - `AL-AM`: adaptive layer, adaptive module
7. Call `compress` to either:
   - apply unstructured-style masks when `--unstr` is set, or
   - physically prune q/k/v/o and up/gate/down weights.
8. Optionally apply bias compensation using the removed-feature mean input.

Important implementation details:

- FLAP computes MLP importance on `mlp.down_proj` input channels, which correspond to the MLP intermediate dimension.
- FLAP computes attention importance on `self_attn.o_proj` input channels and groups them by head.
- Original code assumes LLaMA head dimension `128` in several places.
- Original real-pruning path mutates module dimensions and attention attributes.
- Original bias compensation assumes output projection bias exists.

## FLAP: Qwen3 Porting Feasibility

Qwen3 has a compatible high-level block layout:

- `model.model.layers`
- `layer.self_attn.q_proj/k_proj/v_proj/o_proj`
- `layer.mlp.gate_proj/up_proj/down_proj`
- `Qwen3MLP`: `gate_proj`, `up_proj`, `down_proj`, `SiLU`

But it is not a drop-in target:

- Qwen3 attention uses GQA, so `num_heads` and `num_key_value_heads` differ.
- Head dimension must be read dynamically as `layer.self_attn.head_dim`, not hard-coded as 128.
- Qwen3 projections are usually bias-free. FLAP's bias compensation cannot directly assign to missing bias tensors unless we explicitly add bias or disable compensation.
- Physical attention-head pruning for GQA is risky because q heads and kv heads are coupled differently.

Recommended formal baseline scope:

- First implement **FLAP-Qwen3 MLP-only** using official `BiasGPT` statistics, official metrics, and official AL/UL allocation logic for MLP channels.
- Do not claim full official FLAP attention+MLP compression unless Qwen3 GQA attention pruning is fully ported and validated.
- Report as `FLAP-MLP (Qwen3 port)` or `FLAP-style MLP structured pruning`, with commit and deviations documented.

Recommended implementation choices:

- Use our GSM8K/MATH500 prompts as calibration by default for task-matched evaluation; optionally also support Wikitext2 to match official calibration.
- Collect statistics on `layer.mlp.down_proj` input exactly as FLAP does.
- Use `IFV/WIFV/WIFN`.
- Support at least `UL-UM` and `AL-AM` for MLP-only:
  - `UL-UM`: prune the same ratio in every layer.
  - `AL-AM` MLP-only: standardize per-layer MLP metrics and choose a global threshold across all MLP channels.
- Prefer masking or reduced MLP wrapper for stable generation on Qwen3. Physical shape pruning can be added after smoke tests.

## LLM-Pruner: What The Official Code Does

Core files:

- `external_repos/LLM-Pruner/hf_prune.py`
- `external_repos/LLM-Pruner/llama3.py`
- `external_repos/LLM-Pruner/LLMPruner/pruner/hf_llama_pruner.py`
- `external_repos/LLM-Pruner/LLMPruner/torch_pruning/pruner/algorithms/metapruner.py`

Official pruning flow:

1. Load a LLaMA-family model.
   - `hf_prune.py` imports custom `LLMPruner.models.hf_llama.modeling_llama.LlamaForCausalLM`.
   - `llama3.py` imports `transformers.LlamaForCausalLM`.
2. Build a `torch_pruning.MetaPruner` dependency graph from dummy prompts.
3. Select importance estimator:
   - random
   - L1 / L2 magnitude
   - Taylor
4. Configure pruning mode:
   - block-wise
   - channel-wise
   - layer-wise
5. For block-wise pruning, official code uses root instances:
   - attention projection root, originally `q_proj`, changed to `k_proj` in `llama3.py` for grouped KV
   - MLP `gate_proj`
6. Custom pruners are provided for:
   - LLaMA RMSNorm
   - LLaMA Attention
   - Linear modules
7. For Taylor pruning, calibration examples are used to accumulate gradients.
8. After pruning, attention attributes such as `num_heads` and `num_key_value_heads` are updated.
9. Optional recovery is a separate LoRA post-training script (`post_training.py`).

Important implementation details:

- LLM-Pruner does real structural pruning through dependency graph propagation, not simple masking.
- Correctness depends on model-specific custom pruners and graph tracing.
- `llama3.py` includes GQA-aware hints, but still assumes LLaMA classes and LLaMA RMSNorm.

## LLM-Pruner: Qwen3 Porting Feasibility

This is possible but materially harder than FLAP.

Required Qwen3 work:

- Replace `LlamaForCausalLM` loading with `AutoModelForCausalLM` or Qwen3 classes.
- Add Qwen3 custom pruners:
  - Qwen3 RMSNorm equivalent
  - Qwen3 attention equivalent, including GQA-safe q/k/v/o pruning
  - Qwen3 MLP dependencies
- Build dummy prompts with Qwen3 tokenizer and verify `MetaPruner` can trace Qwen3 forward.
- Validate root instances:
  - MLP: `layer.mlp.gate_proj`
  - Attention: likely `k_proj` for GQA-aware block pruning, but must be verified.
- Update model config after pruning:
  - `hidden_size` if channel-wise pruning changes residual width
  - `num_attention_heads`
  - `num_key_value_heads`
  - `intermediate_size`
- Decide whether to include LoRA recovery. If recovery is used, it is a different baseline from zero-shot pruning and must be reported separately.

Recommended formal baseline scope:

- Start with **LLM-Pruner-Magnitude Qwen3 MLP-only or block-MLP-only** if we want a reliable first result.
- A full official-equivalent Qwen3 LLM-Pruner port should be treated as a separate engineering task because it needs custom dependency graph validation.
- Do not call a simple magnitude mask "LLM-Pruner" without qualification.

Suggested naming:

- Conservative: `LLM-Pruner-Mag-MLP (Qwen3 port)`
- Full port only after dependency graph validation: `LLM-Pruner (Qwen3 port)`

## Comparison To Current GRIFFIN Baseline

GRIFFIN-Qwen3 is a prompt-conditioned dynamic FFN-neuron selection baseline. It does not physically compress the model but uses reduced FFN weights during generation. It is already suitable as:

- `GRIFFIN-style Qwen3 FFN dynamic pruning`

FLAP and LLM-Pruner should be treated differently:

- FLAP is calibration/statistics-based static structured pruning.
- LLM-Pruner is dependency-graph-based real structural pruning, optionally with recovery.

## Recommended Next Steps

1. Implement FLAP-Qwen3 first.
   - MLP-only.
   - Official `BiasGPT` statistics.
   - Official `IFV/WIFV/WIFN` metrics.
   - Official uniform/adaptive allocation logic.
   - Generate GSM8K/MATH500 using our existing evaluator.

2. Validate FLAP-Qwen3 with small smoke tests:
   - GSM8K limit 2.
   - MATH500 limit 2.
   - ratios 0.2 and 0.4.
   - compare against dense and GRIFFIN.

3. Then decide LLM-Pruner path:
   - Short-term: implement and report `LLM-Pruner-Mag-MLP (Qwen3 port)` only if it follows official magnitude importance and pruning roots closely.
   - Long-term: port `MetaPruner` with Qwen3 custom pruners and validate graph tracing.

4. Document all deviations in method tables:
   - official repo commit
   - model architecture support
   - modules pruned
   - whether real pruning or mask/reduced wrapper
   - whether recovery is used

## Bottom Line

- FLAP can be ported to Qwen3 in a credible first version if we restrict to MLP-channel structured pruning and preserve official statistics/metrics.
- Full FLAP attention+MLP port is harder due to Qwen3 GQA and bias compensation assumptions.
- LLM-Pruner full port is significantly harder because it depends on Qwen3-specific dependency graph pruning.
- For formal baseline quality, avoid unqualified claims. Use precise names such as `FLAP-MLP (Qwen3 port)` and `LLM-Pruner-Mag-MLP (Qwen3 port)` until full official-equivalent ports are validated.
