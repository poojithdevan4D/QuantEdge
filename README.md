# QuantEdge 🔬
### Hardware-Aware Face Detection on Samsung GT-S7392 (2013)

> **Research Question:** Can a 224KB neural network run face detection
> in real-time on a 2013 smartphone with no ML accelerator?
> **Answer:** Yes — 8.7 FPS on ARM Cortex-A9. And INT8 quantization
> is actually SLOWER than FP32 on this chip.
> 
**Key Finding:** INT8 is 1.35x SLOWER than FP32 on ARM Cortex-A9.

## Demo

![Face Detection Demo](results/detected1.png)

*BlazeFace running on ARM Cortex-A9 @ 1GHz — face detected at 92% confidence*

Running BlazeFace on ARM Cortex-A9 @ 1GHz | 512MB RAM | Android 4.1.2

---

## What This Project Does
Deploys a 224KB face detection model (BlazeFace) on a 2013 Samsung phone
and benchmarks INT8 vs FP32 quantization on real ARM hardware.

---

## Results
| Stage | Hardware | Time | FPS |
|---|---|---|---|
| Laptop | Intel CPU | 2.08ms | 479 |
| Phone (on-device) | ARM Cortex-A9 | 115ms | 8.7 |

**Key Finding:** INT8 is 1.35x SLOWER than FP32 on ARM Cortex-A9.
Quantization saves memory but not speed on this chip.

---

## Requirements
- Python 3.10+
- Android phone with USB Debugging enabled
- Android NDK (for recompiling ARM binary)
- ADB installed

Install Python dependencies:
```bash
pip install -r requirements.txt
```

---

## How To Run

### 1. Laptop Webcam Detection
```bash
python src/facedetect.py
```

### 2. Live Phone Camera Detection
Connect your Android phone via USB, enable USB debugging, then:
```bash
adb shell am start -a android.media.action.STILL_IMAGE_CAMERA
python src/live_phone.py
```

### 3. On-Device Inference (runs ON the phone)
First push the files to your phone:
```bash
adb push models/blazeface.tflite /sdcard/blazeface.tflite
adb push arm/inference_arm /data/local/tmp/inference_arm
adb push tflite_lib/libtensorflowlite_jni.so /data/local/tmp/libtensorflowlite_jni.so
adb shell chmod 777 /data/local/tmp/inference_arm
adb shell chmod 777 /data/local/tmp/libtensorflowlite_jni.so
```

Then run inference ON the phone:
```bash
adb shell "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/inference_arm"
```

### 4. Recompile ARM Binary (if needed)
Download TFLite 2.4.0 AAR and extract it, then:
```bash
armv7a-linux-androideabi21-clang -o arm/inference_arm arm/inference.c \
  -I tflite_old/headers \
  -L tflite_old/jni/armeabi-v7a \
  -ltensorflowlite_jni -llog -lz -lm -ldl -landroid
```

---

## Hardware Target
| Spec | Value |
|---|---|
| Device | Samsung GT-S7392 |
| CPU | ARM Cortex-A9 @ 1GHz |
| RAM | 512MB |
| Android | 4.1.2 API 16 |
| Architecture | armeabi-v7a |

---

## Stack
Python · TensorFlow Lite 2.4.0 · OpenCV · ONNX · Android NDK 29 · C · ADB
