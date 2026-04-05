# Hardware: Samsung Galaxy On Nxt (SM-G610F)

> ARM Cortex-A53 × 8 | ARMv8-A | 1.6GHz | 3GB RAM | Android 6.0.1 | 2017

---

## Device Specifications

| Spec | Value |
|------|-------|
| Device | Samsung Galaxy On Nxt (SM-G610F) |
| SoC | Samsung Exynos 7870 |
| CPU | ARM Cortex-A53 × 8 @ 1.6GHz |
| Architecture | ARMv8-A (64-bit chip, 32-bit Android) |
| RAM | 3GB |
| Storage | 16GB internal |
| Android version | 6.0.1 (Marshmallow) |
| Year | 2017 |
| NEON | Yes (128-bit, ARMv8 NEON) |
| NNAPI | No (Android 6 — NNAPI requires Android 8.1+) |
| FP16 SIMD | Limited |
| SDOT instruction | No (requires ARMv8.4-A) |

### Architecture Note

The Exynos 7870 is a 64-bit (ARMv8-A) chip. However, Samsung shipped Android 6.0.1 — a 32-bit OS. This means `/system/bin/linker64` is absent, and all native binaries must be compiled for `armeabi-v7a` (32-bit ARM), not `arm64-v8a` despite the hardware being 64-bit capable. This was one of the first errors encountered during setup.

### Why This Device?

The Cortex-A53 is the most common CPU core in budget and mid-range Android phones from 2015–2020. This is the device class used by the majority of smartphone users in emerging markets. LLM inference benchmarks on this chip are not published anywhere — this experiment fills that gap.

---

## Experiment — TinyLlama 1.1B Q4_K_M Inference

**Goal:** First published LLM inference benchmark on Cortex-A53. Thread scaling sweep to find the optimal configuration and identify the memory bandwidth ceiling.

**Date:** April 2026

### Setup

| Component | Detail |
|-----------|--------|
| Model | TinyLlama 1.1B Chat v1.0 Q4_K_M |
| Model size | 636.18 MiB |
| Parameters | 1.10B |
| Framework | llama.cpp (build 9c699074c) |
| Compiled for | armeabi-v7a (32-bit ARM) |
| Benchmark tool | llama-bench |

### Benchmark Command

```bash
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
  /data/local/tmp/llama-bench \
  -m /data/local/tmp/tinyllama-q4_k_m.gguf \
  -t 4 -n 64"
```

---

## Results — Thread Scaling Sweep

> pp = prompt processing (tokens/sec encoding input prompt)  
> tg = token generation (tokens/sec generating each output token)

| Threads | pp t/s | tg t/s | pp Speedup vs t=1 | tg Speedup vs t=1 |
|---------|--------|--------|-------------------|-------------------|
| 1 | 0.64 | 0.57 | 1.00x (baseline) | 1.00x (baseline) |
| 2 | 1.27 | 1.11 | 1.98x | 1.95x |
| **4** | **1.69** | **1.83** | **2.64x** | **3.21x** |
| 8 | 2.29 | 1.13 | 3.58x | 1.98x |

**Optimal configuration: 4 threads → 1.83 tokens/sec generation**

At 1.83 t/s, a 64 token response takes approximately 35 seconds. Slow but functional for offline edge applications.

---

## Key Findings

### Finding 1 — pp and tg have different scaling behaviour

Prompt processing (pp) scales consistently from t=1 to t=8. More threads always help.

Token generation (tg) peaks at t=4 (1.83 t/s) and **drops at t=8** (1.13 t/s). Adding threads beyond 4 actively hurts generation speed.

This is because pp can process multiple input tokens in parallel (compute-bound), while tg must generate tokens sequentially — each token requires loading the entire model weight matrix from RAM (memory-bound).

### Finding 2 — Memory bandwidth saturation at t=8 for tg

At t=8, all 8 Cortex-A53 cores compete for the Exynos 7870's single shared memory controller. The memory bus becomes the bottleneck. This is why tg drops from 1.83 → 1.13 t/s at t=8 while pp continues to scale.

This is a hardware characteristic of the Exynos 7870 that any deployment targeting this chip must account for.

### Finding 3 — LLM inference is viable on 2017 hardware

1.83 t/s is functional for:
- Offline text classification
- Edge summarization
- On-device translation
- Any application where latency is acceptable and privacy is required

Human reading speed is approximately 4-5 tokens/sec. This phone gets you to 40% of reading speed on a model trained on internet-scale text, with zero internet connection required.

### Finding 4 — tg > pp at t=4 (unusual)

At t=4, token generation (1.83 t/s) is slightly faster than prompt processing (1.69 t/s). This is counterintuitive — normally pp is faster because it batches tokens in parallel. On this constrained hardware without FP16 SIMD support, the batching advantage of pp is partially neutralised.

---

## Build Issues and Fixes

| Error | Root Cause | Fix |
|-------|-----------|-----|
| `arm64-v8a binary refused to execute` | Samsung shipped 32-bit Android 6. `/system/bin/linker64` absent | Recompile for `armeabi-v7a` |
| `libomp.so not found at runtime` | OpenMP runtime not bundled on Android | Push `libomp.so` from NDK `arm/` directory via ADB |
| `vld1q_f16` / `vld1_f16` undeclared | FP16 NEON intrinsics not available in ARMv7 32-bit mode | Add `-DGGML_LLAMAFILE=OFF` |
| `httplib` linker error | Server component links libraries unavailable on old Android | Add `-DLLAMA_BUILD_SERVER=OFF` |

---

## Setup Guide

### Cross-compile llama.cpp for armeabi-v7a

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

export NDK=~/android-ndk-r25c

cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=armeabi-v7a \
  -DANDROID_PLATFORM=android-23 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_LLAMAFILE=OFF

cmake --build build-android --config Release -j$(nproc)
```

### Push to device

```bash
# Push shared libraries
adb push build-android/bin/libggml-base.so /data/local/tmp/
adb push build-android/bin/libggml-cpu.so /data/local/tmp/
adb push build-android/bin/libggml.so /data/local/tmp/
adb push build-android/bin/libllama.so /data/local/tmp/

# Push OpenMP runtime (required)
adb push $NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7/lib/linux/arm/libomp.so /data/local/tmp/

# Push binaries
adb push build-android/bin/llama-bench /data/local/tmp/
adb push build-android/bin/llama-cli /data/local/tmp/

# Push model
adb push tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf /data/local/tmp/

# Set permissions
adb shell "chmod +x /data/local/tmp/llama-bench"
adb shell "chmod +x /data/local/tmp/llama-cli"
```

### Run benchmark

```bash
# Full thread sweep (run each separately)
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/llama-bench -m /data/local/tmp/tinyllama-q4_k_m.gguf -t 1 -n 32 -p 64"
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/llama-bench -m /data/local/tmp/tinyllama-q4_k_m.gguf -t 2 -n 32 -p 64"
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/llama-bench -m /data/local/tmp/tinyllama-q4_k_m.gguf -t 4 -n 32 -p 64"
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/llama-bench -m /data/local/tmp/tinyllama-q4_k_m.gguf -t 8 -n 32 -p 64"
```

### Run inference

```bash
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
  /data/local/tmp/llama-cli \
  -m /data/local/tmp/tinyllama-q4_k_m.gguf \
  -t 4 -n 64 -p 'What is machine learning?'"
```

---

## What This Means for QuantEdge

The thread sweep directly answers the most important deployment question for this device class: **what thread count to use?**

The answer (t=4) is not obvious without running the experiment. At t=8 performance degrades — a naive deployment that uses all available cores would be 38% slower than optimal.

QuantEdge's goal is to answer this question automatically for any target device, without requiring the developer to run the sweep manually.
