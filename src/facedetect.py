import cv2
import numpy as np
import tensorflow as tf
import time

tflite = tf.lite

# Load model
model_path = r"C:\Users\pooji\OneDrive\Desktop\Face_detect_dumb_phone\blazeface.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h, input_w = 128, 128

# Generate BlazeFace anchors
def generate_anchors():
    anchors = []
    for y in range(16):
        for x in range(16):
            for _ in range(2):
                anchors.append([(x + 0.5) / 16.0, (y + 0.5) / 16.0])
    for y in range(8):
        for x in range(8):
            for _ in range(6):
                anchors.append([(x + 0.5) / 8.0, (y + 0.5) / 8.0])
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
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip fixes overflow warning

def nms(boxes, scores, threshold=0.3):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < threshold]
    return keep

# Open webcam
cap = cv2.VideoCapture(0)
print("Webcam opened. Press Q to quit.")

# FPS tracking
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Preprocess
    img = cv2.resize(frame, (input_w, input_h))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    # ── FPS calculation ──────────────────────────
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    # ─────────────────────────────────────────────

    raw_boxes  = interpreter.get_tensor(output_details[0]['index'])[0]
    raw_scores = interpreter.get_tensor(output_details[1]['index'])[0]

    scores = sigmoid(raw_scores[:, 0])
    boxes  = decode_boxes(raw_boxes, anchors)

    THRESHOLD = 0.6
    mask = scores > THRESHOLD
    filtered_boxes  = boxes[mask]
    filtered_scores = scores[mask]

    face_found = False

    if len(filtered_boxes) > 0:
        keep = nms(filtered_boxes, filtered_scores)
        face_found = True
        for idx in keep:
            box   = filtered_boxes[idx]
            score = filtered_scores[idx]
            x1 = max(0, int(box[0] * w))
            y1 = max(0, int(box[1] * h))
            x2 = min(w, int(box[2] * w))
            y2 = min(h, int(box[3] * h))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {score:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Status
    status = "FACE DETECTED" if face_found else "Scanning..."
    color  = (0, 255, 0) if face_found else (0, 0, 255)

    cv2.putText(frame, f"QuantEdge | {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # FPS display — yellow text
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("QuantEdge", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()