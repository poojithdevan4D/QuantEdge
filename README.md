# QuantEdge

> **Automated ML model quantization for specific hardware targets.**  
> Give it a model. Give it a hardware spec. Get a deployment-ready quantized model back.

---

## What is QuantEdge?

QuantEdge is a platform that takes any ML model and a target hardware specification and automatically determines the optimal quantization strategy — balancing model size, inference speed, and accuracy without manual intervention.

The long-term vision: you point it at a model and a device, and it figures out the rest.

The current phase: deploying real models on real hardware to build ground truth data — understanding how quantized models actually behave across different hardware targets before automating the decision.

---

## Repo Structure

Each hardware target gets its own folder with full specs, setup guide, results, and analysis. This is the ground truth database that QuantEdge's automation layer will eventually be built on top of.

```
QuantEdge/
├── README.md                          ← you are here
└── hardware/
    ├── galaxy-s3-cortex-a9/           ← Samsung GT-S7392, 2013, ARMv7-A
    ├── galaxy-on-nxt-cortex-a53/      ← Samsung SM-G610F, 2017, ARMv8-A (32-bit Android)
    └── local-ai-rtx3050/              ← RTX 3050 Laptop GPU, CUDA 8.6, Windows
```

---

## Hardware Tested

| Folder | Device | Chip | Architecture | Year |
|--------|--------|------|--------------|------|
| [galaxy-s3-cortex-a9](hardware/galaxy-s3-cortex-a9/) | Samsung GT-S7392 | ARM Cortex-A9 @ 1GHz | ARMv7-A, 32-bit | 2013 |
| [galaxy-on-nxt-cortex-a53](hardware/galaxy-on-nxt-cortex-a53/) | Samsung SM-G610F | ARM Cortex-A53 × 8 @ 1.6GHz | ARMv8-A (32-bit Android) | 2017 |
| [local-ai-rtx3050](hardware/local-ai-rtx3050/) | RTX 3050 Laptop GPU | NVIDIA Ampere, 2048 CUDA cores | CUDA Compute 8.6 | 2021 |

---

## Results Summary

| Device | Model | Framework | Key Result |
|--------|-------|-----------|------------|
| Cortex-A9 | BlazeFace (FP32) | TFLite | 92ms / 10.8 FPS |
| Cortex-A9 | NEON kernel (INT8) | Manual intrinsics | 29.90ms — 2.5x faster than FP32 NEON |
| Cortex-A53 | TinyLlama 1.1B Q4_K_M | llama.cpp | 1.83 t/s at t=4 (optimal) |
| RTX 3050 | Llama 3.2 3B Q4_K_M | llama.cpp + CUDA | Real-time inference, full GPU offload |

---

## Why These Devices?

**The 2013 phone (Cortex-A9):**  
ARMv7-A devices are entirely absent from published TinyML benchmarks. Understanding the absolute floor of what's possible on truly constrained hardware is central to QuantEdge's design.

**The 2017 phone (Cortex-A53):**  
Represents the largest class of Android devices still in active use globally. The thread scaling data collected here (memory bandwidth saturation at t=8) is not published anywhere for this chip.

**The RTX 3050:**  
The development machine — used to study quantized LLM inference at scale. Understanding how Q4_K_M behaves on 4GB VRAM directly informs the VRAM budget constraints QuantEdge applies to GPU targets.

---

## Roadmap

- [x] NEON intrinsics benchmark (Cortex-A9)
- [x] TFLite FP32 BlazeFace deployment (Cortex-A9)
- [x] TinyLlama 1.1B Q4_K_M — full thread sweep (Cortex-A53)
- [x] Llama 3.2 3B Q4_K_M local inference (RTX 3050, CUDA)
- [ ] TFLite INT8 BlazeFace on Cortex-A9 (validate NEON speedup on real model)
- [ ] Cortex-A9 vs Cortex-A53 comparison paper (identical benchmarks, both devices)
- [ ] Automated quantization format selector (model + hardware → optimal format)
- [ ] STM32 Nucleo F746ZG (TFLite Micro, bare metal, 320KB RAM)
- [ ] TinyML Symposium paper submission

---

## About

Built by Poojith (MCA / MSc AI & Data Science, JGU SRM) as part of research into hardware-aware neural network deployment.

GitHub: [github.com/poojithdevan4D/QuantEdge](https://github.com/poojithdevan4D/QuantEdge)
