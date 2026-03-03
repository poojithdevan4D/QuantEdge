import cv2
import numpy as np
import os

print(f"OpenCV version: {cv2.__version__}")
print(f"Model exists: {os.path.exists('blazeface.tflite')}")
print(f"Model size: {os.path.getsize('blazeface.tflite')/1024:.1f} KB")
print("All systems ready. QuantEdge is alive.")