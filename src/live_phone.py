import cv2
import numpy as np
import subprocess
import threading
import time
import os

print("QuantEdge — Phone CPU Face Detection")
print("Inference: Samsung GT-S7392 ARM Cortex-A9")
print("Display:   Laptop screen")
print("=" * 45)

base = r"C:\Users\pooji\OneDrive\Desktop\Face_detect_dumb_phone"
ppm_path    = os.path.join(base, "tmp_frame.ppm")
photo_path  = os.path.join(base, "tmp_photo.jpg")
result_txt  = os.path.join(base, "result.txt")

frame_lock = threading.Lock()
current_frame = [None]
running = [True]

def capture_frames():
    last_photo = ""
    while running[0]:
        try:
            result = subprocess.run(
                ["adb", "shell", "ls", "/sdcard/DCIM/Camera/"],
                capture_output=True, text=True, timeout=3
            )
            files = [f.strip() for f in result.stdout.strip().split('\n')
                     if f.strip().endswith('.jpg') or f.strip().endswith('.JPG')]
            if not files:
                time.sleep(0.5)
                continue
            latest = sorted(files)[-1]
            if latest == last_photo:
                subprocess.run(["adb", "shell", "input", "keyevent", "27"],
                               capture_output=True, timeout=3)
                time.sleep(1.5)
                continue
            last_photo = latest
            print(f"New photo: {latest}")
            subprocess.run(["adb", "pull", f"/sdcard/DCIM/Camera/{latest}", photo_path],
                           capture_output=True, timeout=5)
            frame = cv2.imread(photo_path)
            if frame is None:
                continue
            print(f"Frame size: {frame.shape}")
            frame_h, frame_w = frame.shape[:2]
            size = min(frame_h, frame_w)
            cx = frame_w // 2
            cy = frame_h // 2
            frame_sq = frame[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
            frame_small = cv2.resize(frame_sq, (128, 128))
            rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            with open(ppm_path, 'wb') as f:
                f.write(b"P6\n")
                f.write(f"128 128\n".encode('ascii'))
                f.write(b"255\n")
                f.write(rgb.tobytes())
            print(f"PPM size: {os.path.getsize(ppm_path)}")
            subprocess.run(["adb", "push", ppm_path, "/sdcard/frame.ppm"],
                           capture_output=True, timeout=5)
            with frame_lock:
                current_frame[0] = frame.copy()
        except Exception as e:
            print(f"Capture error: {e}")
        time.sleep(0.3)

latest_result = [None]
latest_ms = [0.0]
result_lock = threading.Lock()

def read_results():
    while running[0]:
        try:
            r = subprocess.run(["adb", "pull", "/sdcard/result.txt", result_txt],
                               capture_output=True, timeout=3)
            if r.returncode == 0 and os.path.exists(result_txt):
                with open(result_txt, 'r') as f:
                    line = f.read().strip()
                if not line:
                    time.sleep(0.05)
                    continue
                if line.startswith("FACE"):
                    parts = line.split()
                    x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    score = float(parts[5])
                    ms = float(parts[6])
                    with result_lock:
                        latest_result[0] = (x1, y1, x2, y2, score)
                        latest_ms[0] = ms
                    print(f">>> FACE DETECTED {score:.0%} | {ms:.1f}ms | ARM CPU <<<")
                elif line.startswith("NONE"):
                    parts = line.split()
                    with result_lock:
                        latest_result[0] = None
                        latest_ms[0] = float(parts[1])
                try:
                    os.remove(result_txt)
                except:
                    pass
        except:
            pass
        time.sleep(0.05)

print("\nStarting threads...")
t1 = threading.Thread(target=capture_frames, daemon=True)
t2 = threading.Thread(target=read_results, daemon=True)
t1.start()
t2.start()

print("Waiting for first frame from phone...")
timeout = 0
while current_frame[0] is None:
    time.sleep(0.2)
    timeout += 1
    if timeout > 100:
        print("ERROR: No frames received.")
        running[0] = False
        exit(1)

print("Stream started! Point Samsung camera at your face.")
print("Press Q to quit\n")

prev_time = time.time()

while True:
    with frame_lock:
        frame = current_frame[0].copy() if current_frame[0] is not None else None
    if frame is None:
        time.sleep(0.05)
        continue

    h, w = frame.shape[:2]
    with result_lock:
        result = latest_result[0]
        ms = latest_ms[0]

    if result:
        x1, y1, x2, y2, score = result
        size = min(h, w)
        x_off = w // 2 - size // 2
        y_off = h // 2 - size // 2
        px1 = max(0, int(x_off + x1 * size))
        py1 = max(0, int(y_off + y1 * size))
        px2 = min(w, int(x_off + x2 * size))
        py2 = min(h, int(y_off + y2 * size))
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 4)
        cv2.putText(frame, f"Face {score:.0%}", (px1, max(py1-15, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    curr_time = time.time()
    fps = 1.0 / max(curr_time - prev_time, 0.001)
    prev_time = curr_time

    status = "FACE DETECTED" if result else "Scanning..."
    color  = (0, 255, 0) if result else (0, 100, 255)
    ms_str = f"{ms:.0f}ms" if ms > 0 else "--"

    display = cv2.resize(frame, (480, 640))
    cv2.putText(display, f"QuantEdge | {status}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(display, f"ARM Cortex-A9: {ms_str}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(display, f"Samsung GT-S7392", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    cv2.imshow("QuantEdge — ARM CPU Inference", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running[0] = False
cv2.destroyAllWindows()
for f in [ppm_path, photo_path, result_txt]:
    try:
        os.remove(f)
    except:
        pass
print("Done.")