#!/usr/bin/env python3
"""Golden activations for the GLM-5.3 (`GlmMoeDsa`) CPU oracle (src/core/glm53/reference).

The pinned semantics reference is transformers 5.16.0's modeling_glm_moe_dsa.py (the
transformers kitaniai converted with; see ref/glm-5.3/README.md). This script builds a tiny
model with the real architecture (MLA + DSA indexer with full/shared layers, sigmoid noaux_tc
router, shared expert, leading dense layers) and random FP32 weights, runs one causal prefill
on CPU with eager attention, and writes everything the C++ test needs to one safetensors file:

  * every weight under its HF name (FP32; routed experts as HF's fused 3-D tensors),
  * `input_ids` [S] (I64),
  * `hidden.L` [S, H]: the residual stream after layer L (`hidden.-1` is the embedding),
  * `topk.L` [S, K] (I32): the attention's selected key positions at layer L, ascending,
  * `latent.L` [S, kv_lora]: the KV latent row each position caches at layer L,
  * `index_key_norm.L` [S, index_head_dim] (Full layers): the index key before RoPE,
  * `final_norm` [S, H] and `logits` [S, V],
  * `__metadata__`: `config_json` (transformers' own config.json text), the seed and versions.

The indexer and the router select by top-k, so a score difference below FP32 noise between the
k-th and (k+1)-th candidate would make the golden depend on summation order. The script records
every such margin and retries seeds until each one is comfortably above the noise floor.

Usage (desktop, CPU only, in the dev image):
  docker run --rm -v $PWD:/work ds4-native-dev:2 bash -c \
    "pip install -q transformers==5.16.0 && python3 /work/make_golden.py --out /work/golden.safetensors"
"""

import argparse

import torch
import transformers
from safetensors.torch import save_file
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

PINNED_TRANSFORMERS = "5.16.0"

# GLM-5.3's architecture at toy width. Every structural feature of the real config.json is
# present: interleaved RoPE (theta 8e6), q/kv LoRA, nope/rope split, v_head != qk_head, an
# indexer whose top-k (5) is smaller than the sequence (16) so DSA actually drops keys, the
# freq/offset full/shared indexer schedule, leading dense layers, sigmoid routing with an
# e_score_correction_bias, top-k renormalization and the 2.5 routed scaling.
TINY = dict(
    vocab_size=64,
    hidden_size=64,
    intermediate_size=48,
    moe_intermediate_size=24,
    num_hidden_layers=7,
    first_k_dense_replace=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=32,
    kv_lora_rank=16,
    qk_rope_head_dim=8,
    qk_nope_head_dim=12,
    v_head_dim=16,
    index_n_heads=4,
    index_head_dim=16,
    index_topk=5,
    index_topk_freq=2,
    index_skip_topk_offset=2,
    n_routed_experts=8,
    num_experts_per_tok=3,
    n_shared_experts=1,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    rms_norm_eps=1e-5,
    rope_parameters={"rope_theta": 8000000.0, "rope_type": "default"},
    max_position_embeddings=4096,
    tie_word_embeddings=False,
    attention_bias=False,
    pad_token_id=0,
)
SEQ = 16
MIN_MARGIN = 1e-3  # relative gap between the k-th and (k+1)-th selection score


class Margins:
    """Records the smallest relative top-k margin of every indexer / router selection."""

    def __init__(self, index_topk, router_topk):
        self.index_topk, self.router_topk = index_topk, router_topk
        self.worst = float("inf")

    def observe(self, scores, k):
        rows = scores.reshape(-1, scores.shape[-1])
        floor = torch.finfo(scores.dtype).min / 2
        for row in rows:
            # -inf and finfo.min mark masked (future / other-group) candidates: never selectable.
            vals = row[torch.isfinite(row) & (row > floor)]
            if vals.numel() <= k:
                continue  # every valid candidate is selected: no boundary to straddle
            top = torch.sort(vals, descending=True).values
            gap = (top[k - 1] - top[k]).abs().item()
            scale = max(top[k - 1].abs().item(), 1e-6)
            self.worst = min(self.worst, gap / scale)


def patch_topk(margins):
    """Wrap torch.topk / Tensor.topk to observe the indexer's and router's selection margins."""
    fn, method = torch.topk, torch.Tensor.topk

    def observed(scores, k):
        if k in (margins.index_topk, margins.router_topk) and scores.dtype == torch.float32:
            margins.observe(scores.detach(), k)

    def topk_fn(input, k, *args, **kwargs):
        observed(input, k)
        return fn(input, k, *args, **kwargs)

    def topk_method(self, k, *args, **kwargs):
        observed(self, k)
        return method(self, k, *args, **kwargs)

    torch.topk, torch.Tensor.topk = topk_fn, topk_method
    return fn, method


def build(seed):
    torch.manual_seed(seed)
    config = GlmMoeDsaConfig(**TINY)
    config._attn_implementation = "eager"
    model = GlmMoeDsaForCausalLM(config).to(torch.float32).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif name.endswith("norm.bias"):
                p.copy_(0.1 * torch.randn_like(p))
            elif name.endswith("indexer.wk.weight"):
                # Index keys enter their LayerNorm with variance ~1e-5, where its eps (1e-6, the
                # module default, not rms_norm_eps) changes the result by tens of percent: the
                # golden then pins the eps, which unit-variance keys would hide under FP32 noise.
                p.copy_(3e-3 * torch.randn_like(p) / p.shape[-1] ** 0.5)
            else:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
        for layer in model.model.layers:
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                gate.e_score_correction_bias.copy_(0.05 * torch.randn_like(gate.e_score_correction_bias))
    return config, model


def run(seed):
    config, model = build(seed)
    margins = Margins(config.index_topk, config.num_experts_per_tok)
    captured = {}

    def keep_topk(idx):
        def hook(_module, _inputs, outputs):
            captured[f"topk.{idx}"] = torch.sort(outputs[2][0].to(torch.int64), dim=-1).values.to(torch.int32)

        return hook

    def keep(name):
        def hook(_module, _inputs, outputs):
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            captured[name] = out[0].detach().clone()

        return hook

    def keep_hidden(idx):
        return keep(f"hidden.{idx}")

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_hook(keep_topk(i))
        layer.register_forward_hook(keep_hidden(i))
        # Cache contents: the KV latent row and (Full layers) the index key before RoPE.
        layer.self_attn.kv_a_layernorm.register_forward_hook(keep(f"latent.{i}"))
        if layer.self_attn.indexer is not None:
            layer.self_attn.indexer.k_norm.register_forward_hook(keep(f"index_key_norm.{i}"))
    model.model.embed_tokens.register_forward_hook(keep_hidden(-1))
    model.model.norm.register_forward_hook(lambda _m, _i, o: captured.__setitem__("final_norm", o[0].clone()))

    g = torch.Generator().manual_seed(seed + 1)
    input_ids = torch.randint(1, config.vocab_size, (1, SEQ), generator=g)
    fn, method = patch_topk(margins)
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits[0]
    finally:
        torch.topk, torch.Tensor.topk = fn, method
    return config, model, input_ids[0], captured, logits, margins.worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise SystemExit(f"transformers {transformers.__version__} is not the pinned {PINNED_TRANSFORMERS}")
    torch.set_num_threads(1)  # one summation order for the margins we just measured

    for seed in range(args.seed, args.seed + 200):
        config, model, input_ids, captured, logits, worst = run(seed)
        if worst >= MIN_MARGIN:
            break
        print(f"seed {seed}: worst top-k margin {worst:.2e} < {MIN_MARGIN:.0e}, retrying")
    else:
        raise SystemExit("no seed met the margin")

    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    tensors["input_ids"] = input_ids.to(torch.int64).contiguous()
    tensors.update({k: v.contiguous() for k, v in captured.items()})
    tensors["logits"] = logits.contiguous()

    # The config exactly as transformers serializes it (the config.json format), so the C++
    # side reads the tiny model through the same parser as the released config.json.
    meta = {"config_json": config.to_json_string(use_diff=False)}
    meta["seq"] = str(SEQ)
    meta["seed"] = str(seed)
    meta["min_topk_margin"] = f"{worst:.6e}"
    meta["transformers"] = transformers.__version__
    meta["torch"] = torch.__version__
    save_file(tensors, args.out, metadata=meta)
    print(f"seed {seed}, worst top-k margin {worst:.3e}, {len(tensors)} tensors -> {args.out}")
    print("indexer_types", config.indexer_types)
    print("mlp_layer_types", config.mlp_layer_types)
    for name, t in sorted(tensors.items()):
        if not name.startswith("model.layers.") or ".0." in name or ".2." in name:
            print(f"  {name} {tuple(t.shape)} {t.dtype}")


if __name__ == "__main__":
    main()
