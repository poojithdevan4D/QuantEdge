#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tensorflow/lite/c/c_api.h"

double get_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    printf("========================================\n");
    printf("  QuantEdge On-Device Inference\n");
    printf("  Samsung GT-S7392 | armeabi-v7a\n");
    printf("========================================\n\n");

    const char* model_path = "/sdcard/blazeface.tflite";

    // Load model
    printf("Loading model: %s\n", model_path);
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
    if(!model) {
        printf("ERROR: Could not load model\n");
        return 1;
    }
    printf("Model loaded OK\n");

    // Create interpreter
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    if(!interpreter) {
        printf("ERROR: Could not create interpreter\n");
        return 1;
    }
    printf("Interpreter created OK\n");

    // Allocate tensors
    if(TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        printf("ERROR: AllocateTensors failed\n");
        return 1;
    }
    printf("Tensors allocated OK\n\n");

    // Get input tensor
    TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
    printf("Input shape: ");
    int i;
    for(i = 0; i < TfLiteTensorNumDims(input); i++)
        printf("%d ", TfLiteTensorDim(input, i));
    printf("\n\n");

    // Fill input with dummy face-like data
    int input_size = 1 * 128 * 128 * 3;
    float* input_data = (float*)TfLiteTensorData(input);
    for(i = 0; i < input_size; i++)
        input_data[i] = 0.5f + (float)(i % 20) * 0.01f;

    // Warmup
    printf("Warmup run...\n");
    TfLiteInterpreterInvoke(interpreter);
    printf("Warmup done\n\n");

    // Benchmark
    int RUNS = 10;
    printf("Running %d inference passes...\n", RUNS);
    double start = get_ms();
    for(i = 0; i < RUNS; i++) {
        TfLiteInterpreterInvoke(interpreter);
    }
    double avg_ms = (get_ms() - start) / RUNS;

    // Get output
    const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    float* out_data = (float*)TfLiteTensorData(output);

    printf("\n========================================\n");
    printf("  QuantEdge On-Device Results\n");
    printf("========================================\n");
    printf("Model:      BlazeFace 224KB\n");
    printf("Device:     Samsung GT-S7392\n");
    printf("CPU:        ARM Cortex-A9 @ 1GHz\n");
    printf("Avg/frame:  %.2f ms\n", avg_ms);
    printf("Max FPS:    %.1f\n", 1000.0/avg_ms);
    printf("Output[0]:  %.4f\n", out_data[0]);
    printf("========================================\n");
    printf("TFLite running on ARM Cortex-A9!\n");
    printf("QuantEdge on-device SUCCESS!\n");
    printf("========================================\n");

    // Cleanup
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
    return 0;
}