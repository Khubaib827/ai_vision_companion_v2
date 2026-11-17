class DistanceEstimator:
    def __init__(self):
        self.avg_heights = {
            "person": 1.7, "car": 1.5, "bus": 3.0, "truck": 3.0,
            "chair": 0.9, "table": 0.75, "door": 2.0, "bicycle": 1.0,
            "motorcycle": 1.1, "dog": 0.5, "cat": 0.3, "couch": 0.85, "bed": 0.6
        }

    def estimate_distance(self, box, class_name, frame_height):
        x1, y1, x2, y2 = box
        pixel_height = y2 - y1
        if pixel_height <= 0:
            return 0.0
        real_height = self.avg_heights.get(class_name, 1.0)
        focal_length = frame_height * 0.85
        distance = (real_height * focal_length) / pixel_height
        return round(float(distance), 2)
