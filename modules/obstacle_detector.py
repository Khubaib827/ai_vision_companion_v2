class ObstacleDetector:
    def __init__(self):
        self.hazard_classes = {
            "car", "bus", "truck", "motorcycle", "bicycle", 
            "chair", "table", "couch", "bed", "door", "fire hydrant", "stop sign"
        }

    def analyze_obstacles(self, detections, frame_width, frame_height):
        hazards = []
        frame_center_x = frame_width / 2

        for det in detections:
            cls_name = det["class"]
            box = det["box"]
            x1, y1, x2, y2 = box
            obj_center_x = (x1 + x2) / 2
            
            if obj_center_x < frame_center_x - (frame_width * 0.15):
                position = "on your left"
            elif obj_center_x > frame_center_x + (frame_width * 0.15):
                position = "on your right"
            else:
                position = "directly ahead"
                
            distance = det.get("distance", 5.0)
            is_hazard = (cls_name in self.hazard_classes) or (distance <= 3.0)
            
            if is_hazard:
                hazards.append({
                    "class": cls_name,
                    "position": position,
                    "distance": distance,
                    "confidence": det["confidence"],
                    "box": box
                })
                
        return hazards
