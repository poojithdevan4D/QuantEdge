# Hardware: NVIDIA RTX 3050 Laptop GPU

> NVIDIA Ampere | CUDA Compute 8.6 | 4GB VRAM | 2048 CUDA cores | Windows 11

---

## Device Specifications

| Spec | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Architecture | Ampere (GA107) |
| CUDA Compute Capability | 8.6 |
| VRAM | 4095 MiB (4GB GDDR6) |
| CUDA Cores | 2048 |
| Memory Bandwidth | 192 GB/s |
| TDP | 35–80W (configurable) |
| Host CPU | Intel Core (x86-64) |
| Host RAM | 16GB |
| OS | Windows 11 |
| CUDA Toolkit | 13.2 |
| Driver | Latest (via NVIDIA App) |

---

## Experiment — Llama 3.2 3B Q4_K_M Local Inference

**Goal:** Run a full language model locally on a consumer laptop GPU. Study how a quantized LLM behaves on constrained VRAM. Build llama.cpp from source with CUDA support.

**Date:** April 2026

### Setup

| Component | Detail |
|-----------|--------|
| Model | Llama 3.2 3B Instruct Q4_K_M |
| Model size | ~2GB on disk |
| Parameters | 3.21B |
| Framework | llama.cpp (built from source, CUDA 13.2) |
| Build system | Ninja (bypasses MSVC toolset compatibility issue) |
| Interface | llama-server web UI at localhost:8080 |

---

## Build — llama.cpp from Source on Windows with CUDA

### Prerequisites

- MSYS2 + g++ (via `pacman -S mingw-w64-ucrt-x86_64-gcc`)
- Visual Studio 2026 Build Tools (Desktop development with C++)
- CUDA Toolkit 13.2
- CMake 4.3+

### Critical: Use x64 Native Tools Command Prompt

Do NOT use regular CMD or VS Code terminal. Search Start Menu for:
```
x64 Native Tools Command Prompt for VS
```

This pre-loads the 64-bit MSVC compiler. CUDA requires 64-bit — using the 32-bit version causes an `ACCESS_VIOLATION` in `cudafe++`.

### CMake Configure

```cmd
cmake -B build -G "Ninja" -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe"
```

Using `-G "Ninja"` instead of the default MSVC generator is required because CUDA 13.2 does not have a registered toolset for VS 2026 Insiders.

### Build

```cmd
cmake --build build --config Release
```

558 files compiled. Takes 10–20 minutes. Output: `build\bin\llama-server.exe`

---

## Results

### CUDA Detection

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 4095 MiB)
Device 0: NVIDIA GeForce RTX 3050 Laptop GPU, compute capability 8.6, VMM: yes
```

### Model Run

```cmd
bin\llama-server.exe -m "Llama-3.2-3B-Instruct-Q4_K_M.gguf" -ngl 33
```

| Metric | Value |
|--------|-------|
| Model loaded | Yes |
| VRAM used | ~2.1 GB |
| GPU layers offloaded | 33 / 33 (full GPU) |
| Inference speed | Real-time |
| Interface | http://127.0.0.1:8080 |
| Internet required | No |

Full GPU offload at `-ngl 33` means all 33 transformer layers of the model run on the RTX 3050. Nothing falls back to CPU.

---

## Errors Encountered and Fixed

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `nmake not found` / `CMAKE_C_COMPILER not set` | MSVC Build Tools not installed | Install VS Build Tools, Desktop dev with C++ workload |
| 2 | `cudafe++ ACCESS_VIOLATION (0xC0000005)` | 32-bit MSVC compiler (`HostX86\x86`) loaded instead of 64-bit | Use x64 Native Tools Command Prompt |
| 3 | `No CUDA toolset found` | CUDA 13.2 has no toolset entry for VS 2026 Insiders | Switch to `-G "Ninja"` build generator |
| 4 | `OpenSSL not found` (warning) | Optional HTTPS support not installed | Ignored — not needed for local inference |
| 5 | `ggml-cuda.dll ggml_backend_init not found` | Shared library linking issue | Model still runs; full fix: `-DBUILD_SHARED_LIBS=OFF` |

---

## Key Observations

### Q4_K_M on 4GB VRAM

A 3.21B parameter model quantized to Q4_K_M fits comfortably in 4GB VRAM with ~2GB to spare. This headroom allows for a longer context window and faster KV cache operations.

The Q4_K_M format uses 4-bit K-quants — a block-wise quantization scheme that applies different precision to different layers based on their sensitivity. This is more sophisticated than flat Q4 and preserves more accuracy at the same file size.

### GGUF Format

The model is stored as a single `.gguf` file containing weights, architecture metadata, and tokenizer in one binary. llama.cpp memory-maps this file directly — loading is fast and VRAM usage is predictable.

### GPU Layer Offloading (-ngl)

`-ngl N` offloads N transformer layers to GPU. For Llama 3.2 3B (28 layers), `-ngl 33` means all layers go to GPU. If VRAM is insufficient, reduce `-ngl` to offload fewer layers — the rest run on CPU. This is the primary knob for fitting a model into available VRAM.

---

## What This Means for QuantEdge

This experiment directly informs how QuantEdge handles GPU targets:

1. **VRAM budget:** A 3B Q4_K_M model uses ~2GB VRAM. QuantEdge must predict VRAM usage from model size and quantization format before deployment.

2. **Layer offloading:** The `-ngl` parameter is the VRAM vs speed tradeoff lever. QuantEdge should automatically compute the maximum `-ngl` value that fits within the target device's VRAM.

3. **Format selection:** Q4_K_M is a good default for 4GB VRAM. Q8_0 would use ~4GB (fills the VRAM). Q2_K would use ~1.4GB but loses noticeable quality. QuantEdge selects the format that maximises quality within the VRAM constraint.

---

## Recommended Models for RTX 3050 (4GB VRAM)

| Model | Format | Size | VRAM | Notes |
|-------|--------|------|------|-------|
| Llama 3.2 3B | Q4_K_M | ~2GB | ~2.1GB | Recommended — fits with headroom |
| Llama 3.2 3B | Q8_0 | ~3.4GB | ~3.5GB | Higher quality, tight fit |
| Llama 3.1 7B | Q2_K | ~2.7GB | ~2.8GB | Larger model, lower precision |
| Llama 3.1 7B | Q4_K_M | ~4.1GB | ~4.2GB | Exceeds VRAM — needs partial CPU offload |

---

## Quick Start

```cmd
# 1. Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 2. Open x64 Native Tools Command Prompt for VS, then:
cmake -B build -G "Ninja" -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe"
cmake --build build --config Release

# 3. Download a model from HuggingFace (bartowski/Llama-3.2-3B-Instruct-GGUF)

# 4. Run
cd build
bin\llama-server.exe -m "path\to\model.gguf" -ngl 33

# 5. Open browser at http://127.0.0.1:8080
```
