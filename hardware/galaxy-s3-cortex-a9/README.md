# Hardware: Samsung GT-S7392 (Galaxy S Duos 2)

> ARM Cortex-A9 | ARMv7-A | 1GHz | 512MB RAM | Android 4.1.2 | 2013

---

## Device Specifications

| Spec | Value |
|------|-------|
| Device | Samsung Galaxy S Duos 2 (GT-S7392) |
| SoC | Broadcom BCM21664 |
| CPU | ARM Cortex-A9 @ 1GHz |
| Architecture | ARMv7-A (32-bit) |
| RAM | 512MB |
| Storage | 4GB internal |
| Android version | 4.1.2 (Jelly Bean) |
| Year | 2013 |
| NEON | Yes (128-bit, ARMv7 NEON) |
| NNAPI | No (Android too old) |
| FP16 SIMD | No |
| SDOT instruction | No |

### Why This Device?

This device is a **documentation gap in the ecosystem.** ARMv7-A devices running Android 4.x are completely absent from published TinyML benchmarks. ARM's own developer platform rejected a learning path submission on this hardware because they no longer support 32-bit Android — confirming the gap is real.

The Cortex-A9 represents the floor of what is deployable. Understanding this floor is essential for building a quantization platform that makes real hardware decisions.

---

## Experiment 1 — NEON Intrinsics Benchmark

**Goal:** Measure raw hardware capability for vectorized integer arithmetic.  
**Date:** April 2026

### What Was Measured

Three kernel implementations of the same matrix multiply operation:

| Implementation | Description |
|----------------|-------------|
| Scalar INT8 | No vectorization — baseline |
| FP32 NEON | Compiler auto-vectorized using NEON |
| INT8 NEON | Manual NEON intrinsics (`vld1_s8`, `vmull_s8`, `vpaddl_s16`, `vpadd_s32`) |

### Results

| Implementation | Time (ms) | Speedup vs Scalar |
|----------------|-----------|-------------------|
| Scalar INT8 | 384.67ms | 1.00x (baseline) |
| FP32 NEON | 74.87ms | 5.14x |
| **INT8 NEON (manual)** | **29.90ms** | **12.86x** |

**INT8 NEON is 2.5x faster than FP32 NEON on this hardware.**

### Key Finding

Manual INT8 NEON intrinsics outperform compiler-auto-vectorized FP32 NEON by 2.5x on Cortex-A9. The compiler cannot generate optimal NEON code automatically for this workload — manual intrinsics are required to reach hardware capability.

> **Correction note:** An earlier version of this benchmark measured compiler auto-vectorization quality rather than true hardware capability. The corrected benchmark uses explicit NEON intrinsics to isolate actual hardware throughput. The direction (INT8 > FP32) was always correct — the magnitude was understated. Correction made following feedback from Majed Shakir (Senior AI Engineer, Edge AI/Automotive/IoT).

### Why This Matters for QuantEdge

This confirms the quantization target: INT8 is the correct precision for ARMv7-A inference. FP32 quantization is 2.5x slower for no accuracy benefit on this chip. QuantEdge should automatically select INT8 as the output format when targeting Cortex-A9 class hardware.

---

## Experiment 2 — TFLite BlazeFace Deployment

**Goal:** Run a real face detection model end-to-end on this device.  
**Date:** April 2026

### Setup

| Component | Detail |
|-----------|--------|
| Model | BlazeFace (full model, FP32) |
| Framework | TensorFlow Lite |
| Compilation | Cross-compiled for ARMv7-A (Android NDK, softfp ABI) |
| Deployment | ADB push to /data/local/tmp/ |

### Conversion Pipeline

```
PyTorch BlazeFace → ONNX → TFLite FP32 → ARMv7-A Android
```

### Results

| Metric | Value |
|--------|-------|
| Inference time (steady-state) | 92ms |
| Effective FPS | 10.8 FPS |
| Model format | TFLite FP32 |
| Warmup runs discarded | 3 |

### Key Notes

- `/sdcard/` is `noexec` on this device — all binaries must be in `/data/local/tmp/`
- `LD_LIBRARY_PATH` must be set explicitly for shared library resolution
- CLOCK_MONOTONIC used for nanosecond-precision timing
- First 3 warmup runs discarded to avoid cold-start effects

### Next Step

Run the same BlazeFace model in INT8 quantized format. Based on the NEON benchmark ratio (2.5x), the expected result is ~37ms / ~27 FPS. This experiment validates whether the synthetic benchmark ratio holds on a real model.

---

## ARM Learning Path Submission

A complete learning path documenting this setup was written and submitted as **PR #3054** to [ArmDeveloperEcosystem/arm-learning-paths](https://github.com/ArmDeveloperEcosystem/arm-learning-paths).

**Status: Rejected**

**Reason:** ARM's developer platform no longer supports 32-bit Android (ARMv7-A).

**Reviewer:** Waheed Brown (Developer Relations Engineer, ARM) flagged the submission to the review team directly.

This rejection confirmed that the documentation gap is real and that this work sits outside the official ecosystem. That is exactly why it is worth building.

---

## Files

| File | Description |
|------|-------------|
| `neon_bench.c` | Three-kernel NEON benchmark (scalar, FP32 NEON, INT8 NEON) |
| `inference.c` | TFLite BlazeFace inference runner |
| `inference_bench.c` | Benchmarked version with timing |

---

## Setup Guide

### Cross-compile for ARMv7-A

```bash
export NDK=~/android-ndk-r25c
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=armeabi-v7a \
  -DANDROID_PLATFORM=android-16 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android --config Release
```

### Deploy via ADB

```bash
adb push your_binary /data/local/tmp/
adb shell "chmod +x /data/local/tmp/your_binary"
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./your_binary"
```
