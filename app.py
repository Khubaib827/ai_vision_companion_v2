import streamlit as st
import os
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from modules.video_processor import VideoProcessor
from modules.object_detector import ObjectDetector
from modules.distance_estimator import DistanceEstimator
from modules.obstacle_detector import ObstacleDetector
from modules.ocr_reader import OCRReader
from modules.voice_assistant import VoiceAssistant
from modules.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="AI Vision Companion",
    page_icon="👁️",
    layout="wide"
)

@st.cache_resource
def load_all_modules():
    detector = ObjectDetector("yolov8n.pt")
    dist_estimator = DistanceEstimator()
    obs_detector = ObstacleDetector()
    ocr_reader = OCRReader(languages=['en', 'ur'])
    voice_assistant = VoiceAssistant()
    report_generator = ReportGenerator()
    return detector, dist_estimator, obs_detector, ocr_reader, voice_assistant, report_generator

# Define custom video transformer for real-time WebRTC streaming
class VisionTransformer(VideoTransformerBase):
    def __init__(self, detector, dist_estimator, obs_detector, ocr_reader, conf_threshold):
        self.detector = detector
        self.dist_estimator = dist_estimator
        self.obs_detector = obs_detector
        self.ocr_reader = ocr_reader
        self.conf_threshold = conf_threshold
        self.latest_detections = []
        self.latest_hazards = []
        self.latest_ocr = []

    def transform(self, frame):
        # Convert incoming WebRTC video frame to RGB numpy array
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        # Run AI Pipeline on live frame
        detections, annotated_frame = self.detector.detect(frame_rgb, conf_threshold=self.conf_threshold)
        
        for det in detections:
            det["distance"] = self.dist_estimator.estimate_distance(det["box"], det["class"], h)
            
        hazards = self.obs_detector.analyze_obstacles(detections, w, h)
        ocr_results = self.ocr_reader.read_text(frame_rgb)
        
        if ocr_results:
            annotated_frame = self.ocr_reader.draw_ocr_boxes(annotated_frame, ocr_results)

        # Store latest states for UI display
        self.latest_detections = detections
        self.latest_hazards = hazards
        self.latest_ocr = ocr_results

        # Convert back to BGR for WebRTC output stream
        annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        return annotated_bgr

def main():
    st.title("👁️ AI Vision Companion")
    st.markdown("### Intelligent Visual Assistance System for Blind and Visually Impaired People")

    detector, dist_estimator, obs_detector, ocr_reader, voice_assistant, report_generator = load_all_modules()

    # Sidebar controls
    st.sidebar.header("Control Panel")
    input_source = st.sidebar.radio("Select Input Source", ["Upload Video File", "Live Real-Time Webcam Stream"])
    conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    voice_lang = st.sidebar.selectbox("Voice Assistance Language", ["English (en)", "Urdu (ur)"])

    if input_source == "Live Real-Time Webcam Stream":
        st.subheader("📹 Live Real-Time Webcam Stream")
        st.write("Streaming live from your webcam. The AI model is analyzing frames in real-time.")

        # Initialize WebRTC Streamer
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        webrtc_ctx = webrtc_streamer(
            key="ai-vision-stream",
            video_transformer_factory=lambda: VisionTransformer(detector, dist_estimator, obs_detector, ocr_reader, conf_threshold),
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True
        )

        if webrtc_ctx.state.playing:
            st.success("Webcam stream is active and processing live frames.")
            
            # Display real-time data from transformer if available
            if webrtc_ctx.video_transformer:
                transformer = webrtc_ctx.video_transformer
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.write(f"### Detected Objects ({len(transformer.latest_detections)})")
                    if transformer.latest_detections:
                        st.json(transformer.latest_detections)
                    else:
                        st.info("Scanning for objects...")
                with col_b:
                    st.write(f"### Hazards ({len(transformer.latest_hazards)})")
                    if transformer.latest_hazards:
                        for hzd in transformer.latest_hazards:
                            st.warning(f"⚠️ **{hzd['class'].capitalize()}** {hzd['position']} (~{hzd['distance']}m)")
                    else:
                        st.success("Path looks clear.")
                with col_c:
                    st.write(f"### OCR Text ({len(transformer.latest_ocr)})")
                    if transformer.latest_ocr:
                        for txt in transformer.latest_ocr:
                            st.info(f"📝 **{txt['text']}**")
                    else:
                        st.info("No text detected.")

    else:
        # Standard Video Upload Mode
        step_size = st.sidebar.slider("Frame Skip Rate (Speed vs Precision)", 1, 30, 10, 1)
        uploaded_file = st.file_uploader("Upload Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            st.success(f"Successfully uploaded: **{uploaded_file.name}**")
            
            with st.spinner("Initializing AI modules..."):
                processor = VideoProcessor(uploaded_file)
                metadata = processor.get_video_metadata()

            if metadata:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Duration", f"{metadata['duration']:.2f} s")
                col2.metric("Resolution", f"{metadata['width']}x{metadata['height']}")
                col3.metric("FPS", f"{metadata['fps']:.1f}")
                col4.metric("Total Frames", metadata['total_frames'])

                st.divider()
                st.subheader("Uploaded Video Preview")
                st.video(uploaded_file)
                st.divider()

                if st.button("Start Full Video Processing Pipeline"):
                    fps = metadata["fps"] if metadata["fps"] > 0 else 30
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_frame_results = []
                    frame_gen = processor.extract_frames(step_size=step_size)
                    
                    total_estimated_frames = metadata["total_frames"] // step_size
                    processed_count = 0
                    
                    for idx, frame in frame_gen:
                        processed_count += 1
                        current_progress = min(processed_count / max(total_estimated_frames, 1), 1.0)
                        progress_bar.progress(current_progress)
                        status_text.text(f"Processing frame {idx} / {metadata['total_frames']}...")
                        
                        h, w, _ = frame.shape
                        detections, _ = detector.detect(frame, conf_threshold=conf_threshold)
                        
                        for det in detections:
                            det["distance"] = dist_estimator.estimate_distance(det["box"], det["class"], h)
                            
                        hazards = obs_detector.analyze_obstacles(detections, w, h)
                        ocr_results = ocr_reader.read_text(frame)
                        
                        timestamp = idx / fps
                        all_frame_results.append({
                            "frame_idx": idx,
                            "timestamp": timestamp,
                            "detections": detections,
                            "hazards": hazards,
                            "ocr_results": ocr_results
                        })
                    
                    progress_bar.empty()
                    status_text.success("Video processing complete!")
                    
                    report_dict = report_generator.generate_summary_report(metadata, all_frame_results)
                    json_report = report_generator.export_to_json(report_dict)
                    
                    st.divider()
                    st.subheader("📊 Video Analysis Summary Report")
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("Frames Analyzed", report_dict["total_frames_analyzed"])
                    col_r2.metric("Total Hazards Logged", report_dict["total_hazards_detected"])
                    col_r3.metric("Unique Objects", len(report_dict["unique_objects_identified"]))
                    
                    st.write("**Identified Objects:**", ", ".join(report_dict["unique_objects_identified"]) if report_dict["unique_objects_identified"] else "None")
                    st.write("**Text & Signs Detected:**", ", ".join(report_dict["unique_text_signs_read"]) if report_dict["unique_text_signs_read"] else "None")
                    
                    if report_dict["hazard_timeline"]:
                        st.write("### ⚠️ Hazard Timeline Log")
                        st.dataframe(report_dict["hazard_timeline"])
                    else:
                        st.success("No hazardous obstacles logged throughout the video.")
                        
                    st.download_button(
                        label="Download Full JSON Report",
                        data=json_report,
                        file_name="ai_vision_companion_report.json",
                        mime="application/json"
                    )
                
                processor.cleanup()
            else:
                st.error("Failed to read video metadata.")
        else:
            st.info("Please upload a video file to begin.")

if __name__ == "__main__":
    main()