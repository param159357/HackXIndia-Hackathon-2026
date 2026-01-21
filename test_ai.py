from ultralytics import YOLO
import cv2
import numpy as np

print("Testing imports...")
try:
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")
    
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
    print("Tracking successful!")
    
    if results[0].boxes.id is None:
        print("No tracks detected (expected on empty image)")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
