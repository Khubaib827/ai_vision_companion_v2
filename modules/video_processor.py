import os
import cv2
import tempfile

class VideoProcessor:
    def __init__(self, source=None):
        """
        Initializes VideoProcessor.
        - If source is an UploadedFile object: saves to temporary file.
        - If source is an integer (e.g., 0): uses live camera feed.
        """
        self.source = source
        self.is_live = isinstance(source, int)
        self.temp_file_path = None
        
        if not self.is_live and source is not None:
            self.temp_file_path = self._save_temp_file()

    def _save_temp_file(self):
        """Saves uploaded video file to temporary disk location."""
        try:
            suffix = os.path.splitext(self.source.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(self.source.read())
                return tmp_file.name
        except Exception as e:
            raise RuntimeError(f"Failed to save temporary video file: {e}")

    def get_video_metadata(self):
        """Extracts basic video properties for uploaded videos."""
        if self.is_live or not self.temp_file_path:
            return None
            
        cap = cv2.VideoCapture(self.temp_file_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration
        }

    def extract_frames(self, step_size=1):
        """Generator yielding RGB frames from uploaded video."""
        if self.is_live or not self.temp_file_path:
            return
            
        cap = cv2.VideoCapture(self.temp_file_path)
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % step_size == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_idx, frame_rgb
                
            frame_idx += 1
            
        cap.release()

    def cleanup(self):
        """Deletes temporary video file from disk."""
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
            except Exception:
                pass
