/*
 * neon_bench.c
 * QuantEdge — NEON Intrinsics Benchmark
 * Answers Majed Shakir's challenge:
 *   "Does INT8 NEON beat FP32 NEON on Cortex-A9?"
 *
 * Three kernels compared:
 *   1. Scalar INT8     — plain C, no SIMD (your original benchmark)
 *   2. FP32 NEON       — vmlaq_f32, 4 floats/instruction
 *   3. INT8 NEON       — vmull_s8 + vpadalq_s16, honest widening
 *
 * Compile for ARM (on your laptop):
 *   armv7a-linux-androideabi21-clang ^
 *     -O2 -mfpu=neon -mfloat-abi=softfp ^
 *     -o neon_bench_arm neon_bench.c -lm
 *
 * Run on Samsung:
 *   adb push neon_bench_arm /data/local/tmp/
 *   adb shell chmod 777 /data/local/tmp/neon_bench_arm
 *   adb shell /data/local/tmp/neon_bench_arm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

#define N    256
#define RUNS 10

/* ──────────────────────────────────────────────────
   TIMING — nanosecond resolution
   clock() has 10ms resolution on Android 4.1, useless
   CLOCK_MONOTONIC gives real hardware time
   ────────────────────────────────────────────────── */
double get_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* ──────────────────────────────────────────────────
   KERNEL 1: SCALAR INT8
   Plain C, no SIMD — this is what your original
   matrix_3x3_mul.c was doing
   ────────────────────────────────────────────────── */
void matmul_scalar_int8(int8_t* A, int8_t* B, int32_t* C) {
    int i, j, k;
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            int32_t sum = 0;
            for(k = 0; k < N; k++) {
                sum += (int32_t)A[i*N+k] * (int32_t)B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
}

/* ──────────────────────────────────────────────────
   KERNEL 2: FP32 NEON
   Uses vmlaq_f32 — 4 FP32 multiply-accumulate per instruction
   This is the HONEST FP32 NEON path
   ────────────────────────────────────────────────── */
void matmul_fp32_neon(float* A, float* B, float* C) {
    int i, j, k;
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            for(k = 0; k < N; k += 4) {
                float32x4_t a = vld1q_f32(&A[i*N + k]);
                float32x4_t b = vld1q_f32(&B[k*N + j]);
                sum = vmlaq_f32(sum, a, b);  /* sum += a * b, 4 at once */
            }
            /* horizontal sum of 4 lanes → single float */
            /* Note: vaddvq_f32 is ARMv8 only — use vpadd for ARMv7 */
            float32x2_t lo = vget_low_f32(sum);
            float32x2_t hi = vget_high_f32(sum);
            float32x2_t s  = vadd_f32(lo, hi);
            s = vpadd_f32(s, s);
            C[i*N+j] = vget_lane_f32(s, 0);
        }
    }
}

/* ──────────────────────────────────────────────────
   KERNEL 3: INT8 NEON
   Uses vmull_s8 — multiplies 8 INT8 pairs → 8 INT16 results
   Then vpadalq_s16 — pair-add INT16 into INT32 accumulator
   This is EXACTLY what Majed is asking about.
   The "widening" chain: INT8 → INT16 → INT32
   ────────────────────────────────────────────────── */
void matmul_int8_neon(int8_t* A, int8_t* B, int32_t* C) {
    int i, j, k;
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            int32x4_t sum = vdupq_n_s32(0);
            for(k = 0; k < N; k += 8) {
                int8x8_t  a   = vld1_s8(&A[i*N + k]);
                int8x8_t  b   = vld1_s8(&B[k*N + j]);
                /* vmull_s8: INT8 × INT8 → INT16 (widening multiply) */
                int16x8_t mul = vmull_s8(a, b);
                /* vpadalq_s16: pairwise-add INT16 → accumulate into INT32 */
                sum = vpadalq_s16(sum, mul);
            }
            /* horizontal sum of 4 INT32 lanes */
            int32x2_t lo = vget_low_s32(sum);
            int32x2_t hi = vget_high_s32(sum);
            int32x2_t s  = vadd_s32(lo, hi);
            s = vpadd_s32(s, s);
            C[i*N+j] = vget_lane_s32(s, 0);
        }
    }
}

/* ──────────────────────────────────────────────────
   MAIN
   ────────────────────────────────────────────────── */
int main() {
    printf("=== QuantEdge NEON Benchmark ===\n");
    printf("Device : ARM Cortex-A9 Samsung GT-S7392\n");
    printf("Matrix : %dx%d\n", N, N);
    printf("Runs   : %d (after 2 warmup)\n\n", RUNS);

    /* allocate matrices */
    int8_t*  Ai = (int8_t*)  malloc(N*N*sizeof(int8_t));
    int8_t*  Bi = (int8_t*)  malloc(N*N*sizeof(int8_t));
    int32_t* Ci = (int32_t*) malloc(N*N*sizeof(int32_t));
    float*   Af = (float*)   malloc(N*N*sizeof(float));
    float*   Bf = (float*)   malloc(N*N*sizeof(float));
    float*   Cf = (float*)   malloc(N*N*sizeof(float));

    if(!Ai || !Bi || !Ci || !Af || !Bf || !Cf) {
        printf("ERROR: malloc failed\n");
        return 1;
    }

    /* fill with deterministic values */
    int x;
    for(x = 0; x < N*N; x++) {
        Ai[x] = (int8_t)(x % 7);
        Bi[x] = (int8_t)(x % 5);
        Af[x] = (float)(x % 7) / 7.0f;
        Bf[x] = (float)(x % 5) / 5.0f;
    }

    double t0, t1, total;
    int r;

    /* ── KERNEL 1: Scalar INT8 ── */
    printf("Running Scalar INT8...\n");
    /* warmup */
    matmul_scalar_int8(Ai, Bi, Ci);
    matmul_scalar_int8(Ai, Bi, Ci);
    /* timed runs */
    total = 0;
    for(r = 0; r < RUNS; r++) {
        t0 = get_ms();
        matmul_scalar_int8(Ai, Bi, Ci);
        t1 = get_ms();
        total += (t1 - t0);
    }
    double scalar_int8_ms = total / RUNS;
    /* force compiler to keep result — print checksum */
    printf("  Checksum: %d\n", Ci[0] + Ci[N*N-1]);
    printf("  Result: %.2f ms/run\n\n", scalar_int8_ms);

    /* ── KERNEL 2: FP32 NEON ── */
    printf("Running FP32 NEON...\n");
    matmul_fp32_neon(Af, Bf, Cf);
    matmul_fp32_neon(Af, Bf, Cf);
    total = 0;
    for(r = 0; r < RUNS; r++) {
        t0 = get_ms();
        matmul_fp32_neon(Af, Bf, Cf);
        t1 = get_ms();
        total += (t1 - t0);
    }
    double fp32_neon_ms = total / RUNS;
    printf("  Checksum: %.2f\n", Cf[0] + Cf[N*N-1]);
    printf("  Result: %.2f ms/run\n\n", fp32_neon_ms);

    /* ── KERNEL 3: INT8 NEON ── */
    printf("Running INT8 NEON...\n");
    matmul_int8_neon(Ai, Bi, Ci);
    matmul_int8_neon(Ai, Bi, Ci);
    total = 0;
    for(r = 0; r < RUNS; r++) {
        t0 = get_ms();
        matmul_int8_neon(Ai, Bi, Ci);
        t1 = get_ms();
        total += (t1 - t0);
    }
    double int8_neon_ms = total / RUNS;
    printf("  Checksum: %d\n", Ci[0] + Ci[N*N-1]);
    printf("  Result: %.2f ms/run\n\n", int8_neon_ms);

    /* ── SUMMARY ── */
    printf("========================================\n");
    printf("RESULTS — %dx%d matmul, %d runs\n", N, N, RUNS);
    printf("========================================\n");
    printf("Scalar INT8 (plain C)   : %7.2f ms\n", scalar_int8_ms);
    printf("FP32   NEON (vmlaq_f32) : %7.2f ms\n", fp32_neon_ms);
    printf("INT8   NEON (vmull_s8)  : %7.2f ms\n", int8_neon_ms);
    printf("----------------------------------------\n");
    printf("INT8 NEON vs FP32 NEON  : %.3fx %s\n",
           int8_neon_ms / fp32_neon_ms,
           int8_neon_ms < fp32_neon_ms ? "(INT8 WINS)" : "(FP32 WINS)");
    printf("INT8 NEON vs Scalar INT8: %.3fx %s\n",
           int8_neon_ms / scalar_int8_ms,
           int8_neon_ms < scalar_int8_ms ? "(NEON faster)" : "(scalar faster?!)");
    printf("========================================\n");
    printf("This answers Majed Shakir's challenge.\n");

    free(Ai); free(Bi); free(Ci);
    free(Af); free(Bf); free(Cf);
    return 0;
}
