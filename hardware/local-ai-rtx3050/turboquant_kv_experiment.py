import numpy as np
from turboquant.main.mse import TurboQuantMSE
from turboquant.main.prod import TurboQuantProd

# ─── SIMULATE A REAL KV CACHE ───────────────────────────────────────────────
# Llama 3.2 3B uses head_dim=128, 8 attention heads
# We simulate 512 tokens (like a real conversation)

HEAD_DIM = 128
N_HEADS = 8
SEQ_LEN = 512

rng = np.random.default_rng(42)

# Simulate key and value vectors as they would arrive from Llama
# In real inference, these come from the attention layer one token at a time
keys   = rng.standard_normal((SEQ_LEN, HEAD_DIM)).astype(np.float32)
values = rng.standard_normal((SEQ_LEN, HEAD_DIM)).astype(np.float32)

# Normalize (TurboQuant requires unit-norm vectors)
keys   /= np.linalg.norm(keys,   axis=1, keepdims=True)
values /= np.linalg.norm(values, axis=1, keepdims=True)

# Simulate a query vector
query = rng.standard_normal(HEAD_DIM).astype(np.float32)
query /= np.linalg.norm(query)

print("=" * 60)
print("  QuantEdge Phase 4 — TurboQuant KV Cache Experiment")
print(f"  head_dim={HEAD_DIM}, heads={N_HEADS}, seq_len={SEQ_LEN}")
print("=" * 60)

# ─── BASELINE: fp16 attention scores ────────────────────────────────────────
scores_fp16 = query @ keys.T  # shape (SEQ_LEN,)
top5_exact  = np.argsort(-scores_fp16)[:5]

print(f"\n[Baseline fp16]")
print(f"  Memory for keys: {keys.nbytes / 1024:.1f} KB")
print(f"  Top-5 attended tokens: {list(top5_exact)}")

# ─── TURBO 4-BIT ────────────────────────────────────────────────────────────
tq4 = TurboQuantMSE(dim=HEAD_DIM, bits=4)
idx4 = tq4.quantize(keys)
keys_hat4 = tq4.dequantize(idx4)

scores_4bit = query @ keys_hat4.T
top5_4bit   = np.argsort(-scores_4bit)[:5]
mse_4bit    = float(np.mean((scores_fp16 - scores_4bit) ** 2))
recall_4bit = len(set(top5_4bit) & set(top5_exact)) / 5

print(f"\n[TurboQuant 4-bit]")
print(f"  Memory for keys: {idx4.nbytes / 1024:.1f} KB  (4x smaller)")
print(f"  Attention score MSE: {mse_4bit:.6f}")
print(f"  Top-5 recall: {recall_4bit*100:.0f}%")
print(f"  Top-5 attended tokens: {list(top5_4bit)}")

# ─── TURBO 3-BIT ────────────────────────────────────────────────────────────
tq3 = TurboQuantMSE(dim=HEAD_DIM, bits=3)
idx3 = tq3.quantize(keys)
keys_hat3 = tq3.dequantize(idx3)

scores_3bit = query @ keys_hat3.T
top5_3bit   = np.argsort(-scores_3bit)[:5]
mse_3bit    = float(np.mean((scores_fp16 - scores_3bit) ** 2))
recall_3bit = len(set(top5_3bit) & set(top5_exact)) / 5

print(f"\n[TurboQuant 3-bit]")
print(f"  Memory for keys: {idx3.nbytes / 1024:.1f} KB  (5.3x smaller)")
print(f"  Attention score MSE: {mse_3bit:.6f}")
print(f"  Top-5 recall: {recall_3bit*100:.0f}%")
print(f"  Top-5 attended tokens: {list(top5_3bit)}")

# ─── TURBO 2-BIT ────────────────────────────────────────────────────────────
tq2 = TurboQuantMSE(dim=HEAD_DIM, bits=2)
idx2 = tq2.quantize(keys)
keys_hat2 = tq2.dequantize(idx2)

scores_2bit = query @ keys_hat2.T
top5_2bit   = np.argsort(-scores_2bit)[:5]
mse_2bit    = float(np.mean((scores_fp16 - scores_2bit) ** 2))
recall_2bit = len(set(top5_2bit) & set(top5_exact)) / 5

print(f"\n[TurboQuant 2-bit]")
print(f"  Memory for keys: {idx2.nbytes / 1024:.1f} KB  (8x smaller)")
print(f"  Attention score MSE: {mse_2bit:.6f}")
print(f"  Top-5 recall: {recall_2bit*100:.0f}%")
print(f"  Top-5 attended tokens: {list(top5_2bit)}")

# ─── CONTEXT WINDOW PROJECTION ──────────────────────────────────────────────
vram_gb     = 1.9  # available VRAM after model weights on RTX 3050
fp16_bytes  = HEAD_DIM * N_HEADS * 2 * 2  # keys + values, 2 bytes per fp16
turbo3_bytes = HEAD_DIM * N_HEADS * 2 * (3/8)  # 3 bits packed

ctx_fp16   = int((vram_gb * 1024**3) / fp16_bytes)
ctx_turbo3 = int((vram_gb * 1024**3) / turbo3_bytes)

print(f"\n{'='*60}")
print(f"  Context Window Projection (RTX 3050, 1.9GB free VRAM)")
print(f"{'='*60}")
print(f"  fp16 KV cache:        ~{ctx_fp16:,} tokens")
print(f"  TurboQuant 3-bit:     ~{ctx_turbo3:,} tokens")
print(f"  Improvement:          {ctx_turbo3 // ctx_fp16}x more context")