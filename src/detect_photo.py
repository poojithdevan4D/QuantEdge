import cv2
import numpy as np
import tensorflow as tf
import os

tflite = tf.lite

model_path  = r"C:\Users\pooji\OneDrive\Desktop\Face_detect_dumb_phone\blazeface.tflite"
image_path  = r"C:\Users\pooji\OneDrive\Desktop\Face_detect_dumb_phone\phone_capture.jpg"
output_path = r"C:\Users\pooji\OneDrive\Desktop\Face_detect_dumb_phone\result.jpg"

# Load model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_h, input_w = 128, 128

# Anchors
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

# Load phone photo
print(f"Loading photo from phone: {image_path}")
frame = cv2.imread(image_path)
if frame is None:
    print("ERROR: Could not load image")
    exit()

print(f"Photo size: {frame.shape[1]}x{frame.shape[0]}")
h, w = frame.shape[:2]

# Preprocess
img = cv2.resize(frame, (input_w, input_h))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_input = np.expand_dims(img_rgb.astype(np.float32)/255.0, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img_input)
interpreter.invoke()

raw_boxes  = interpreter.get_tensor(output_details[0]['index'])[0]
raw_scores = interpreter.get_tensor(output_details[1]['index'])[0]

scores = sigmoid(raw_scores[:, 0])
boxes  = decode_boxes(raw_boxes, anchors)

# Filter
mask = scores > 0.6
filtered_boxes  = boxes[mask]
filtered_scores = scores[mask]

print(f"Detections above threshold: {len(filtered_scores)}")

# Draw results
face_count = 0
if len(filtered_boxes) > 0:
    keep = nms(filtered_boxes, filtered_scores)
    face_count = len(keep)
    for idx in keep:
        box   = filtered_boxes[idx]
        score = filtered_scores[idx]
        x1 = max(0, int(box[0]*w))
        y1 = max(0, int(box[1]*h))
        x2 = min(w, int(box[2]*w))
        y2 = min(h, int(box[3]*h))
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(frame, f"Face {score:.0%}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

# Add QuantEdge watermark
cv2.putText(frame, f"QuantEdge | {face_count} face(s) detected", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
cv2.putText(frame, "Samsung GT-S7392 Camera", (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

# Save result
cv2.imwrite(output_path, frame)
print(f"\n{'='*45}")
print(f"  QuantEdge Detection Result")
print(f"{'='*45}")
print(f"  Source:  Samsung GT-S7392 camera")
print(f"  Faces:   {face_count} detected")
print(f"  Saved:   result.jpg")
print(f"{'='*45}")

# Show it
cv2.imshow("QuantEdge — Phone Camera Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()