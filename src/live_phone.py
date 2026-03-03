import cv2
import numpy as np
import tensorflow as tf
import subprocess
import threading
import time
import sys

tflite = tf.lite

print("QuantEdge — Live Phone Camera Detection")
print("=" * 45)

# ── Load Model ───────────────────────────────
model_path = r"C:\Users\pooji\OneDrive\Desktop\Face_detect_dumb_phone\models\blazeface.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h, input_w = 128, 128
print("Model loaded OK")

# ── Anchors ───────────────────────────────────
def generate_anchors():
    anchors = []
    for y in range(16):
        for x in range(16):
            for _ in range(2):
                anchors.append([(x+0.5)/16.0, (y+0.5)/16.0])
    for y in range(8):
        for x in range(8):
            for _ in range(6):
                anchors.append([(x+0.5)/8.0, (y+0.5)/8.0])
    return np.array(anchors, dtype=np.float32)

anchors = generate_anchors()

def decode_boxes(raw_boxes, anchors):
    boxes = np.zeros_like(raw_boxes[:, :4])
    boxes[:, 0] = raw_boxes[:, 0] / 128.0 * anchors[:, 0] + anchors[:, 0]
    boxes[:, 1] = raw_boxes[:, 1] / 128.0 * anchors[:, 1] + anchors[:, 1]
    boxes[:, 2] = raw_boxes[:, 2] / 128.0
    boxes[:, 3] = raw_boxes[:, 3] / 128.0
    decoded = np.zeros_like(boxes)
    decoded[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    decoded[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    decoded[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    decoded[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return decoded

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def nms(boxes, scores, threshold=0.3):
    if len(boxes) == 0: return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou = inter/(areas[i]+areas[order[1:]]-inter)
        order = order[1:][iou < threshold]
    return keep

def detect(frame):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (input_w, input_h))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb.astype(np.float32)/255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    raw_boxes  = interpreter.get_tensor(output_details[0]['index'])[0]
    raw_scores = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = sigmoid(raw_scores[:, 0])
    boxes  = decode_boxes(raw_boxes, anchors)
    mask = scores > 0.6
    fb = boxes[mask]
    fs = scores[mask]
    results = []
    if len(fb) > 0:
        keep = nms(fb, fs)
        for idx in keep:
            box = fb[idx]
            score = fs[idx]
            x1 = max(0, int(box[0]*w))
            y1 = max(0, int(box[1]*h))
            x2 = min(w, int(box[2]*w))
            y2 = min(h, int(box[3]*h))
            results.append((x1, y1, x2, y2, score))
    return results

# ── Capture frames from phone via ADB ────────
print("Starting phone camera stream via ADB...")
print("Point your Samsung GT-S7392 camera at your face")
print("Press Q to quit\n")

# Use ADB to continuously capture frames
frame_lock = threading.Lock()
current_frame = [None]
running = [True]

def capture_frames():
    """Continuously pull frames from phone camera via ADB screencap"""
    tmp_path = r"C:\Users\pooji\OneDrive\Desktop\Face_detect_dumb_phone\tmp_frame.png"
    while running[0]:
        try:
            # Save to phone first, then pull
            subprocess.run(
                ["adb", "shell", "screencap", "-p", "/sdcard/tmp_frame.png"],
                capture_output=True, timeout=5
            )
            subprocess.run(
                ["adb", "pull", "/sdcard/tmp_frame.png", tmp_path],
                capture_output=True, timeout=5
            )
            frame = cv2.imread(tmp_path)
            if frame is not None:
                with frame_lock:
                    current_frame[0] = frame
        except Exception as e:
            print(f"Capture error: {e}")
        time.sleep(0.2)

# Start capture thread
t = threading.Thread(target=capture_frames, daemon=True)
t.start()

print("Waiting for first frame...")
while current_frame[0] is None:
    time.sleep(0.1)
print("Stream started!\n")

# ── Main loop ────────────────────────────────
prev_time = time.time()
fps = 0

while True:
    with frame_lock:
        frame = current_frame[0].copy() if current_frame[0] is not None else None

    if frame is None:
        continue

    # Run detection
    detections = detect(frame)

    # Draw results
    for (x1, y1, x2, y2, score) in detections:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(frame, f"Face {score:.0%}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # FPS
    curr_time = time.time()
    fps = 1.0 / max(curr_time - prev_time, 0.001)
    prev_time = curr_time

    # Status overlay
    status = f"FACE DETECTED ({len(detections)})" if detections else "Scanning..."
    color  = (0,255,0) if detections else (0,0,255)
    cv2.putText(frame, f"QuantEdge | {status}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Samsung GT-S7392 | FPS: {fps:.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("QuantEdge — Live Phone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running[0] = False
cv2.destroyAllWindows()
print("Done.")