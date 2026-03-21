/*
 * inference_bench.c
 * QuantEdge — TFLite Model Inference Benchmark
 * Runs a .tflite model 10 times on the Samsung and reports mean timing.
 *
 * Usage:
 *   /data/local/tmp/inference_bench /sdcard/blazeface.tflite
 *   /data/local/tmp/inference_bench /sdcard/blazeface_int8.tflite
 *
 * Compile:
 *   armv7a-linux-androideabi21-clang.cmd ^
 *     -O2 -mfpu=neon -mfloat-abi=softfp ^
 *     -o inference_bench ^
 *     inference_bench.c ^
 *     -I tflite_old/headers ^
 *     -L tflite_old/jni/armeabi-v7a ^
 *     -ltensorflowlite_jni -llog -lz -lm -ldl -landroid
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "tensorflow/lite/c/c_api.h"

#define WARMUP_RUNS 3
#define BENCH_RUNS  10

/* ── timing ─────────────────────────────────────── */
double get_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* ── stats ───────────────────────────────────────── */
double mean(double* arr, int n) {
    double s = 0;
    int i;
    for(i = 0; i < n; i++) s += arr[i];
    return s / n;
}

double stddev(double* arr, int n, double m) {
    double s = 0;
    int i;
    for(i = 0; i < n; i++) s += (arr[i]-m)*(arr[i]-m);
    return sqrt(s / n);
}

double min_val(double* arr, int n) {
    double m = arr[0];
    int i;
    for(i = 1; i < n; i++) if(arr[i] < m) m = arr[i];
    return m;
}

double max_val(double* arr, int n) {
    double m = arr[0];
    int i;
    for(i = 1; i < n; i++) if(arr[i] > m) m = arr[i];
    return m;
}

int main(int argc, char** argv) {
    const char* model_path = "/sdcard/blazeface.tflite";
    if(argc > 1) model_path = argv[1];

    printf("========================================\n");
    printf("  QuantEdge TFLite Inference Benchmark\n");
    printf("  Samsung GT-S7392 | Cortex-A9 @ 1GHz\n");
    printf("========================================\n\n");
    printf("Model  : %s\n", model_path);
    printf("Warmup : %d runs (discarded)\n", WARMUP_RUNS);
    printf("Bench  : %d runs\n\n", BENCH_RUNS);

    /* ── load model ─────────────────────────────── */
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
    if(!model) {
        printf("ERROR: could not load model from %s\n", model_path);
        printf("Make sure you pushed the model:\n");
        printf("  adb push blazeface.tflite /sdcard/blazeface.tflite\n");
        return 1;
    }
    printf("Model loaded OK\n");

    /* ── create interpreter ─────────────────────── */
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    TfLiteInterpreter* interp = TfLiteInterpreterCreate(model, options);
    if(!interp) { printf("ERROR: interpreter create failed\n"); return 1; }
    TfLiteInterpreterAllocateTensors(interp);
    printf("Interpreter ready\n");

    /* ── print tensor info ──────────────────────── */
    const TfLiteTensor* in_t = TfLiteInterpreterGetInputTensor(interp, 0);
    printf("Input  : type=%d  dims=[", TfLiteTensorType(in_t));
    int d;
    for(d = 0; d < TfLiteTensorNumDims(in_t); d++) {
        printf("%d", TfLiteTensorDim(in_t, d));
        if(d < TfLiteTensorNumDims(in_t)-1) printf(",");
    }
    printf("]\n\n");

    /* ── fill input with dummy data ─────────────── */
    /* For FP32 model: input is float32 */
    /* For INT8 model: input may be int8 — we fill with zeros either way */
    void* input_ptr = TfLiteTensorData(in_t);
    int input_bytes = (int)TfLiteTensorByteSize(in_t);
    memset(input_ptr, 0, input_bytes);
    printf("Input filled (%d bytes)\n\n", input_bytes);

    /* ── warmup ─────────────────────────────────── */
    printf("Warming up (%d runs)...\n", WARMUP_RUNS);
    int r;
    for(r = 0; r < WARMUP_RUNS; r++) {
        TfLiteInterpreterInvoke(interp);
    }
    printf("Warmup done\n\n");

    /* ── benchmark ──────────────────────────────── */
    printf("Benchmarking (%d runs)...\n", BENCH_RUNS);
    double times[BENCH_RUNS];
    for(r = 0; r < BENCH_RUNS; r++) {
        double t0 = get_ms();
        TfLiteInterpreterInvoke(interp);
        double t1 = get_ms();
        times[r] = t1 - t0;
        printf("  Run %2d: %.2f ms\n", r+1, times[r]);
    }

    /* ── stats ──────────────────────────────────── */
    double m   = mean(times, BENCH_RUNS);
    double sd  = stddev(times, BENCH_RUNS, m);
    double mn  = min_val(times, BENCH_RUNS);
    double mx  = max_val(times, BENCH_RUNS);
    double fps = 1000.0 / m;

    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    printf("Model     : %s\n", model_path);
    printf("Mean      : %.2f ms\n", m);
    printf("Std dev   : %.2f ms\n", sd);
    printf("Min       : %.2f ms\n", mn);
    printf("Max       : %.2f ms\n", mx);
    printf("FPS       : %.1f\n", fps);
    printf("========================================\n");

    /* ── cleanup ────────────────────────────────── */
    TfLiteInterpreterDelete(interp);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return 0;
}
