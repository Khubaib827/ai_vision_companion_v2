import easyocr
import cv2
import numpy as np

class OCRReader:
    def __init__(self, languages=['en', 'ur']):
        try:
            self.reader = easyocr.Reader(languages, gpu=False)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR reader: {e}")

    def read_text(self, frame):
        try:
            results = self.reader.readtext(frame)
            extracted_texts = []
            for (bbox, text, prob) in results:
                if prob >= 0.35:
                    extracted_texts.append({
                        "text": text.strip(),
                        "confidence": round(float(prob), 2),
                        "bbox": bbox
                    })
            return extracted_texts
        except Exception as e:
            print(f"OCR Error: {e}")
            return []

    def draw_ocr_boxes(self, frame, extracted_texts):
        annotated_frame = frame.copy()
        for item in extracted_texts:
            bbox = item["bbox"]
            text = item["text"]
            pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            top_left = (int(bbox[0][0]), max(int(bbox[0][1]) - 10, 10))
            cv2.putText(annotated_frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return annotated_frame
