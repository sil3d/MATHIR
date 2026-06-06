"""
convert_lfm2_to_gguf.py — Convert merged LFM2-700M model to GGUF format

Uses correct LFM2 tensor names from original GGUF:
- shortconv for convolution layers
- Proper layer ordering
"""

import sys
import io
import json
import site
from pathlib import Path
import numpy as np

site_packages = site.getsitepackages()[0]
sys.path.insert(0, site_packages)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

MERGED_MODEL = "C:/Users/So-i-learn-3D/Desktop/SECRET_CODE/mindtree/mycerise-700m-gguf-v8/merged_model"
OUTPUT_GGUF = "C:/Users/So-i-learn-3D/Desktop/SECRET_CODE/mindtree/mycerise-700m-gguf-v8/LFM2-700M-Mycerise-V8-F16.gguf"

print("=" * 70)
print("CONVERT LFM2-700M TO GGUF (FIXED TENSOR NAMES)")
print("=" * 70)
print(f"Input: {MERGED_MODEL}")
print(f"Output: {OUTPUT_GGUF}")
print()

import gguf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
print("[1/3] Loading merged model...")
model_path = Path(MERGED_MODEL)
if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(str(model_path))

total_params = sum(p.numel() for p in model.parameters())
print(f"   Model loaded: {total_params / 1e9:.2f}B params")
print(f"   Vocab size: {tokenizer.vocab_size}")

# Get config info
config = model.config
hidden_size = config.hidden_size
num_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads
num_layers = config.num_hidden_layers
vocab_size = tokenizer.vocab_size  # Use tokenizer's actual vocab_size (64400)
intermediate_size = config.intermediate_size
rope_theta = (
    config.rope_parameters.get("rope_theta", 1000000.0)
    if hasattr(config, "rope_parameters")
    else 1000000.0
)

# Layer types from config
layer_types = config.layer_types  # ['conv', 'conv', 'full_attention', ...]
full_attn_indices = [i for i, t in enumerate(layer_types) if t == "full_attention"]

print(f"   Hidden size: {hidden_size}")
print(f"   Num layers: {num_layers}")
print(f"   Attention heads: {num_heads}")
print(f"   KV heads: {num_kv_heads}")
print(f"   Intermediate size: {intermediate_size}")
print(f"   Full attention indices: {full_attn_indices}")

# Create GGUF writer with LFM2 architecture
print("[2/3] Setting up GGUF writer...")

# Get full state dict and determine actual FFN size from weights
state_dict = model.state_dict()
ffn_gate_weight = state_dict["model.layers.0.feed_forward.w1.weight"]
actual_ffn_size = ffn_gate_weight.shape[0]
print(f"   Actual FFN size from weights: {actual_ffn_size}")
print(f"   Config intermediate_size: {intermediate_size}")

writer = gguf.GGUFWriter(str(OUTPUT_GGUF), "lfm2")

# Add metadata
print("   Adding model metadata...")
writer.add_name("LFM2-700M-Mycerise-V8")
writer.add_description("LFM2-700M fine-tuned on Mycerise orchestrator data")
writer.add_block_count(num_layers)
writer.add_embedding_length(hidden_size)
writer.add_head_count(num_heads)
writer.add_head_count_kv(num_kv_heads)
writer.add_rope_freq_base(rope_theta)
writer.add_feed_forward_length(actual_ffn_size)
writer.add_context_length(config.max_position_embeddings)
writer.add_layer_norm_eps(config.norm_eps)
writer.add_layer_norm_rms_eps(config.norm_eps)
writer.add_vocab_size(vocab_size)
writer.add_shortconv_l_cache(config.conv_L_cache)

# Tokenizer
print("   Adding tokenizer...")
writer.add_tokenizer_model("llama")
writer.add_bos_token_id(config.bos_token_id)
writer.add_eos_token_id(config.eos_token_id)
writer.add_pad_token_id(config.pad_token_id)
writer.add_add_bos_token(True)
writer.add_add_eos_token(False)

# Add special tokens to vocab
print("   Adding vocabulary...")
vocab = tokenizer.get_vocab()
token_list = list(vocab.keys())
writer.add_token_list(token_list)

# Convert tensors
print("[3/3] Converting tensors...")


# Map tensor names: HF -> GGUF LFM2 (CORRECTED NAMES)
def map_tensor_name(name):
    """Map HuggingFace tensor names to GGUF LFM2 tensor names."""

    # Embeddings
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    elif name == "model.embedding_norm.weight":
        return "token_embd_norm.weight"
    elif name == "lm_head.weight":
        return "output.weight"

    # Extract layer number
    if ".layers." in name:
        layer = name.split(".layers.")[1].split(".")[0]
        layer_int = int(layer)

    # Convolution layers (shortconv)
    if ".conv.conv.weight" in name:
        return f"blk.{layer}.shortconv.conv.weight"
    elif ".conv.in_proj.weight" in name:
        return f"blk.{layer}.shortconv.in_proj.weight"
    elif ".conv.out_proj.weight" in name:
        return f"blk.{layer}.shortconv.out_proj.weight"

    # Feed forward (same order as original: gate, down, up)
    if ".feed_forward.w1.weight" in name:
        return f"blk.{layer}.ffn_gate.weight"
    elif ".feed_forward.w3.weight" in name:
        return f"blk.{layer}.ffn_up.weight"
    elif ".feed_forward.w2.weight" in name:
        return f"blk.{layer}.ffn_down.weight"

    # Layer norms
    if ".operator_norm.weight" in name:
        return f"blk.{layer}.attn_norm.weight"
    if ".ffn_norm.weight" in name:
        return f"blk.{layer}.ffn_norm.weight"

    # Attention (layers with full_attention)
    if ".self_attn.q_proj.weight" in name:
        return f"blk.{layer}.attn_q.weight"
    if ".self_attn.k_proj.weight" in name:
        return f"blk.{layer}.attn_k.weight"
    if ".self_attn.v_proj.weight" in name:
        return f"blk.{layer}.attn_v.weight"
    if ".self_attn.out_proj.weight" in name:
        return f"blk.{layer}.attn_output.weight"
    if ".self_attn.q_layernorm.weight" in name:
        return f"blk.{layer}.attn_q_norm.weight"
    if ".self_attn.k_layernorm.weight" in name:
        return f"blk.{layer}.attn_k_norm.weight"

    return name


# Get full state dict and convert
state_dict = model.state_dict()
tensor_count = len(state_dict)

for i, (name, tensor) in enumerate(state_dict.items()):
    gguf_name = map_tensor_name(name)

    # Convert to numpy float16
    tensor_np = tensor.to(dtype=torch.float16).cpu().numpy()

    # Transpose embeddings (HuggingFace: [vocab, hidden], GGUF: [hidden, vocab])
    # Truncate to vocab_size (64400) to match tokenizer
    if gguf_name == "token_embd.weight" or gguf_name == "output.weight":
        # GGUF expects [vocab, hidden] format for these weights
        # HF model has [vocab, hidden] already, so no transpose needed
        if tensor_np.shape[0] > vocab_size:
            tensor_np = tensor_np[:vocab_size, :]
            print(f"          (truncated to {vocab_size})")

    print(f"   [{i + 1}/{tensor_count}] {name}")
    print(f"          -> {gguf_name}: {list(tensor_np.shape)}")

    # For embedding and output weights, explicitly pass shape
    if gguf_name == "token_embd.weight" or gguf_name == "output.weight":
        writer.add_tensor(gguf_name, tensor_np, raw_shape=list(tensor_np.shape))
    else:
        writer.add_tensor(gguf_name, tensor_np)

# Write file
print()
print(f"   Writing to {OUTPUT_GGUF}...")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

print()
print("=" * 70)
print("DONE!")
print(f"Output: {Path(OUTPUT_GGUF).stat().st_size / 1e9:.2f} GB")
print("=" * 70)
