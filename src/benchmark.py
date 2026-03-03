import os
import numpy as np
import tensorflow as tf
import time
import cv2

tflite = tf.lite

model_path = r"C:\Users\pooji\OneDrive\Desktop\Face_detect_dumb_phone\blazeface.tflite"

# Model size
size_kb = os.path.getsize(model_path) / 1024
print(f"Model size: {size_kb:.1f} KB")

# Load and measure inference time
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

# Dummy input
dummy = np.random.rand(1, 128, 128, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy)

# Warmup
for _ in range(10):
    interpreter.invoke()

# Measure 100 runs
start = time.time()
for _ in range(100):
    interpreter.invoke()
end = time.time()

avg_ms = (end - start) / 100 * 1000
print(f"Avg inference time: {avg_ms:.2f} ms per frame")
print(f"Max theoretical FPS: {1000/avg_ms:.1f}")
print(f"\n--- Benchmark Entry 1 ---")
print(f"Model:     BlazeFace FP32")
print(f"Size:      {size_kb:.1f} KB")
print(f"Inference: {avg_ms:.2f} ms")
print(f"Max FPS:   {1000/avg_ms:.1f}")