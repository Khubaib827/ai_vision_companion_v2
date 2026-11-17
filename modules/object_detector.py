import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(self, frame, conf_threshold=0.4):
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detected_objects = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                detected_objects.append({
                    "class": cls_name,
                    "confidence": round(conf, 2),
                    "box": [x1, y1, x2, y2]
                })
        
        annotated_bgr = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        return detected_objects, annotated_frame
