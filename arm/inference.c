#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tensorflow/lite/c/c_api.h"
#include <math.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

// Minimal JPEG loader - reads raw pixels from file
// We'll use a simple PPM format instead for simplicity

double get_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int main() {
    printf("========================================\n");
    printf("  QuantEdge On-Device Face Detection\n");
    printf("  Samsung GT-S7392 | armeabi-v7a\n");
    printf("========================================\n\n");

    const char* model_path  = "/sdcard/blazeface.tflite";
    const char* result_path = "/sdcard/result.txt";

    // Load model
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
    if(!model) { printf("ERROR: model load failed\n"); return 1; }
    printf("Model loaded OK\n");

    // Create interpreter
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterAllocateTensors(interpreter);
    printf("Interpreter ready\n\n");

    // Get input tensor
    TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
    float* input_data = (float*)TfLiteTensorData(input);
    int input_size = 1 * 128 * 128 * 3;

    printf("Waiting for frames at /sdcard/frame.ppm ...\n");
    printf("Press Ctrl+C to stop\n\n");

    int frame_count = 0;

    while(1) {
        // Check if frame file exists
        FILE* f = fopen("/sdcard/frame.ppm", "rb");
        if(!f) {
            usleep(50000); // wait 50ms
            continue;
        }

        // Read PPM file (P6 format: binary RGB)
        char magic[3];
        int width, height, maxval;
        fscanf(f, "%s %d %d %d\n", magic, &width, &height, &maxval);

        if(magic[0] != 'P' || magic[1] != '6') {
            fclose(f);
            printf("ERROR: Not a valid PPM file\n");
            continue;
        }

        // Read pixel data
        int pixels = width * height * 3;
        unsigned char* img = (unsigned char*)malloc(pixels);
        fread(img, 1, pixels, f);
        fclose(f);

        // Delete frame file (signal laptop to send next frame)
        remove("/sdcard/frame.ppm");

        // Resize to 128x128 and normalize
        int i, j;
        for(i = 0; i < 128; i++) {
            for(j = 0; j < 128; j++) {
                int src_x = j * width / 128;
                int src_y = i * height / 128;
                int src_idx = (src_y * width + src_x) * 3;
                int dst_idx = (i * 128 + j) * 3;
                input_data[dst_idx+0] = img[src_idx+0] / 255.0f;
                input_data[dst_idx+1] = img[src_idx+1] / 255.0f;
                input_data[dst_idx+2] = img[src_idx+2] / 255.0f;
            }
        }
        free(img);

        // Run inference on ARM CPU
        double t0 = get_ms();
        TfLiteInterpreterInvoke(interpreter);
        double ms = get_ms() - t0;

        // Get outputs
        const TfLiteTensor* out_boxes  = TfLiteInterpreterGetOutputTensor(interpreter, 0);
        const TfLiteTensor* out_scores = TfLiteInterpreterGetOutputTensor(interpreter, 1);
        float* boxes  = (float*)TfLiteTensorData(out_boxes);
        float* scores = (float*)TfLiteTensorData(out_scores);

        // Find best detection
        int num_anchors = 896;
        float best_score = 0;
        int best_idx = -1;
        int k;
        for(k = 0; k < num_anchors; k++) {
            float s = sigmoid(scores[k]);
            if(s > best_score) {
                best_score = s;
                best_idx = k;
            }
        }

        // Write result to file
        FILE* r = fopen(result_path, "w");
        if(r) {
            if(best_score > 0.4f && best_idx >= 0) {
                // Decode box
                float cx = boxes[best_idx*16 + 0];
                float cy = boxes[best_idx*16 + 1];
                float w  = boxes[best_idx*16 + 2];
                float h  = boxes[best_idx*16 + 3];

                // Anchor
                float anchor_x, anchor_y;
                if(best_idx < 512) {
                    anchor_x = ((best_idx/2) % 16 + 0.5f) / 16.0f;
                    anchor_y = ((best_idx/2) / 16 + 0.5f) / 16.0f;
                } else {
                    int idx2 = best_idx - 512;
                    anchor_x = ((idx2/6) % 8 + 0.5f) / 8.0f;
                    anchor_y = ((idx2/6) / 8 + 0.5f) / 8.0f;
                }

                cx = cx / 128.0f + anchor_x;
                cy = cy / 128.0f + anchor_y;
                w  = w  / 128.0f;
                h  = h  / 128.0f;

                float x1 = cx - w/2;
                float y1 = cy - h/2;
                float x2 = cx + w/2;
                float y2 = cy + h/2;

                fprintf(r, "FACE %.4f %.4f %.4f %.4f %.4f %.2f\n",
                    x1, y1, x2, y2, best_score, ms);
                printf("Frame %d: FACE DETECTED %.0f%% | %.1fms | ARM CPU\n",
                    frame_count, best_score*100, ms);
            } else {
                fprintf(r, "NONE %.2f\n", ms);
                printf("Frame %3d | No face (best=%.2f) | %.1fms | ARM CPU\n",
                        frame_count, best_score, ms);
            }
            fclose(r);
        }

        frame_count++;
    }

    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
    return 0;
}