from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import sys

sys.path.insert(0, r"C:\Users\pooji\OneDrive\Desktop\Local LLM\turboq+llama_cpp\turboquant")
from turboquant.main.mse import TurboQuantMSE

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

BITS = 3
HEAD_DIM = 128
NUM_KV_HEADS = 8
tq = TurboQuantMSE(dim=HEAD_DIM, bits=BITS)

stats = {
    "v_vectors": 0,
    "v_mse_total": 0.0,
}

def make_proj_hook(layer_idx, proj_type):
    def hook(module, args, output):
        # Skip K compression — RoPE encoding creates outliers
        if proj_type == "k":
            return output

        x = output.detach().float().cpu().numpy()
        batch, seq_len, total_dim = x.shape
        
        # Correctly reshape: split into (batch * seq_len * num_heads, head_dim)
        x_reshaped = x.reshape(batch * seq_len * NUM_KV_HEADS, HEAD_DIM)

        # Compress and decompress
        indices = tq.quantize(x_reshaped)
        x_reconstructed = tq.dequantize(indices)

        # Measure MSE
        mse = float(np.mean(np.sum((x_reshaped - x_reconstructed)**2, axis=-1)))
        stats["v_vectors"] += x_reshaped.shape[0]
        stats["v_mse_total"] += mse

        if layer_idx == 0 and seq_len > 1:
            print(f"  Layer {layer_idx} | v_proj | shape: {x.shape} | MSE: {mse:.6f}")

        # Return correctly shaped output
        x_out = x_reconstructed.reshape(batch, seq_len, total_dim)
        return torch.tensor(x_out, dtype=output.dtype, device=output.device)

    return hook

hooks = []
for i, layer in enumerate(model.model.layers):
    hv = layer.self_attn.v_proj.register_forward_hook(make_proj_hook(i, "v"))
    hooks.append(hv)

# Run WITHOUT TurboQuant first to get baseline response
for h in hooks:
    h.remove()
hooks = []

prompt = "Explain what gravity is in simple words."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print(f"Prompt: {prompt}")
print("\n--- BASELINE (no compression) ---")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
baseline_response = tokenizer.decode(out[0], skip_special_tokens=True)
print(baseline_response)

# Now run WITH TurboQuant
for i, layer in enumerate(model.model.layers):
    hk = layer.self_attn.k_proj.register_forward_hook(make_proj_hook(i, "k"))
    hv = layer.self_attn.v_proj.register_forward_hook(make_proj_hook(i, "v"))
    hooks.append(hk)
    hooks.append(hv)

print("\n--- WITH TURBOQUANT 3-bit KV compression ---")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
turbo_response = tokenizer.decode(out[0], skip_special_tokens=True)
print(turbo_response)

for h in hooks:
    h.remove()

print(f"\n--- TurboQuant Stats ---")
print(f"V vectors compressed: {stats['v_vectors']}")
if stats['v_vectors'] > 0:
    print(f"Average V MSE: {stats['v_mse_total'] / stats['v_vectors']:.6f}")

print(f"\n--- Response Comparison ---")
print(f"Same response: {baseline_response == turbo_response}")