import json
from datetime import datetime

class ReportGenerator:
    def __init__(self):
        pass

    def generate_summary_report(self, video_metadata, all_frame_results):
        total_frames_analyzed = len(all_frame_results)
        unique_objects_detected = set()
        total_hazards_logged = 0
        total_texts_found = set()
        hazard_log = []
        
        for res in all_frame_results:
            frame_idx = res["frame_idx"]
            timestamp = res["timestamp"]
            for det in res["detections"]:
                unique_objects_detected.add(det["class"])
            for hzd in res["hazards"]:
                total_hazards_logged += 1
                hazard_log.append({
                    "frame": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "hazard": hzd["class"],
                    "position": hzd["position"],
                    "distance": hzd["distance"]
                })
            for txt in res["ocr_results"]:
                total_texts_found.add(txt["text"])

        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_duration_seconds": round(video_metadata.get("duration", 0), 2),
            "resolution": f"{video_metadata.get('width', 0)}x{video_metadata.get('height', 0)}",
            "total_frames_analyzed": total_frames_analyzed,
            "unique_objects_identified": list(unique_objects_detected),
            "total_hazards_detected": total_hazards_logged,
            "unique_text_signs_read": list(total_texts_found),
            "hazard_timeline": hazard_log
        }
        return report

    def export_to_json(self, report_dict):
        return json.dumps(report_dict, indent=4, ensure_ascii=False)
