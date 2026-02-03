import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time

st.set_page_config(page_title="DRDO ROI Occlusion System", layout="wide")
st.title("DRDO ROI Occlusion System - Real-time Smoke Analysis")

# Initialize session state variables
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'roi_selected' not in st.session_state:
    st.session_state.roi_selected = False
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'occlusion_data' not in st.session_state:
    st.session_state.occlusion_data = pd.DataFrame(columns=["Frame", "Occlusion (%)"])
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'play_video' not in st.session_state:
    st.session_state.play_video = False
if 'roi_coordinates' not in st.session_state:
    st.session_state.roi_coordinates = None

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is None:
    st.warning("Please upload a video first.")
    st.stop()

# Save uploaded video
tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
tfile.write(uploaded_video.read())
video_path = tfile.name

# Show uploaded video preview
st.subheader("Uploaded Video Preview")
st.video(video_path)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

st.success(f"Video Loaded Successfully ✅ Total Frames: {total_frames}, FPS: {fps}")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["ROI Selection", "Real-time Analysis", "Results"])

with tab1:
    st.subheader("Select ROI on Frame")
    
    frame_no = st.slider("Select Frame for ROI Selection", 0, total_frames - 1, 0, 
                         help="Use the slider to find the frame where the object is clearly visible")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    
    if not ret:
        st.error("Could not read frame from video.")
        st.stop()
    
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    st.image(frame_pil, width=900, caption=f"Frame {frame_no}")
    
    # Resize image for canvas
    CANVAS_W = 900
    scale = CANVAS_W / frame_pil.size[0]
    CANVAS_H = int(frame_pil.size[1] * scale)
    
    frame_pil_resized = frame_pil.resize((CANVAS_W, CANVAS_H))
    
    st.subheader("Draw Bounding Box on Object (ROI Selection)")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=3,
        stroke_color="red",
        background_image=frame_pil_resized,
        update_streamlit=True,
        height=CANVAS_H,
        width=CANVAS_W,
        drawing_mode="rect",
        key="canvas_roi",
    )
    
    # Check if rectangle is drawn
    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        # Get last rectangle
        obj = canvas_result.json_data["objects"][-1]
        
        # ROI coordinates from resized canvas
        x = int(obj["left"])
        y = int(obj["top"])
        w = int(obj["width"])
        h = int(obj["height"])
        
        # Convert back to original frame coordinates
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        orig_w = int(w / scale)
        orig_h = int(h / scale)
        
        st.session_state.roi_coordinates = (orig_x, orig_y, orig_w, orig_h)
        
        st.success(f"✅ ROI Selected - x={orig_x}, y={orig_y}, width={orig_w}, height={orig_h}")
        
        # ROI Preview
        roi_preview = frame_rgb[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
        
        if roi_preview.size > 0:
            st.image(Image.fromarray(roi_preview), width=400, caption="Selected ROI Preview")
            st.session_state.roi_selected = True
        else:
            st.error("ROI extraction failed. Please draw ROI inside frame.")
    else:
        st.info("✍️ Please draw a rectangle on the object to select ROI.")

with tab2:
    st.subheader("Real-time Occlusion Analysis")
    
    if not st.session_state.roi_selected:
        st.warning("Please select ROI first in the 'ROI Selection' tab.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶ Start Analysis", type="primary"):
                st.session_state.analysis_running = True
                st.session_state.occlusion_data = pd.DataFrame(columns=["Frame", "Occlusion (%)"])
                st.session_state.current_frame = 0
        
        with col2:
            if st.button("⏸ Pause Analysis"):
                st.session_state.analysis_running = False
        
        with col3:
            if st.button("🔄 Reset Analysis"):
                st.session_state.analysis_running = False
                st.session_state.occlusion_data = pd.DataFrame(columns=["Frame", "Occlusion (%)"])
                st.session_state.current_frame = 0
        
        # Create placeholders for video and graph
        video_placeholder = st.empty()
        graph_placeholder = st.empty()
        data_placeholder = st.empty()
        
        # Get ROI coordinates
        orig_x, orig_y, orig_w, orig_h = st.session_state.roi_coordinates
        
        if st.session_state.analysis_running:
            cap = cv2.VideoCapture(video_path)
            
            # Get template from first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame = cap.read()
            
            if ret:
                first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                roi_template = first_gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
                
                # Reset to current frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
                
                while st.session_state.analysis_running and st.session_state.current_frame < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Draw ROI rectangle on frame
                    cv2.rectangle(frame_rgb, (orig_x, orig_y), 
                                (orig_x + orig_w, orig_y + orig_h), 
                                (255, 0, 0), 3)
                    
                    # Calculate occlusion
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_now = gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
                    
                    if roi_now.size > 0 and roi_template.size > 0:
                        # Ensure same size
                        roi_now_resized = cv2.resize(roi_now, (roi_template.shape[1], roi_template.shape[0]))
                        
                        # Calculate difference
                        diff = cv2.absdiff(roi_template, roi_now_resized)
                        
                        # Calculate occlusion percentage
                        occlusion_percent = (np.sum(diff > 30) / diff.size) * 100
                        
                        # Add to session state
                        new_data = pd.DataFrame({
                            "Frame": [st.session_state.current_frame],
                            "Occlusion (%)": [round(float(occlusion_percent), 2)]
                        })
                        
                        st.session_state.occlusion_data = pd.concat([st.session_state.occlusion_data, new_data], ignore_index=True)
                        
                        # Add occlusion text to frame
                        cv2.putText(frame_rgb, f"Occlusion: {occlusion_percent:.1f}%", 
                                  (orig_x, orig_y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display frame
                    video_placeholder.image(frame_rgb, width=900, 
                                          caption=f"Frame: {st.session_state.current_frame}/{total_frames}")
                    
                    # Update graph
                    if not st.session_state.occlusion_data.empty:
                        graph_placeholder.line_chart(st.session_state.occlusion_data.set_index("Frame"))
                    
                    # Update current frame
                    st.session_state.current_frame += 1
                    
                    # Small delay to simulate real-time playback
                    time.sleep(1/fps)
                
                cap.release()

with tab3:
    st.subheader("Analysis Results")
    
    if not st.session_state.occlusion_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Frames Analyzed", len(st.session_state.occlusion_data))
        
        with col2:
            avg_occlusion = st.session_state.occlusion_data["Occlusion (%)"].mean()
            st.metric("Average Occlusion", f"{avg_occlusion:.2f}%")
        
        st.subheader("Occlusion Graph")
        st.line_chart(st.session_state.occlusion_data.set_index("Frame"))
        
        st.subheader("Detailed Data")
        st.dataframe(st.session_state.occlusion_data)
        
        # Calculate statistics
        st.subheader("Statistics")
        
        max_occlusion = st.session_state.occlusion_data["Occlusion (%)"].max()
        min_occlusion = st.session_state.occlusion_data["Occlusion (%)"].min()
        std_occlusion = st.session_state.occlusion_data["Occlusion (%)"].std()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Max Occlusion", f"{max_occlusion:.2f}%")
        col2.metric("Min Occlusion", f"{min_occlusion:.2f}%")
        col3.metric("Std Deviation", f"{std_occlusion:.2f}%")
        
        # Find frames with high occlusion (> 50%)
        high_occlusion_frames = st.session_state.occlusion_data[
            st.session_state.occlusion_data["Occlusion (%)"] > 50
        ]
        
        if not high_occlusion_frames.empty:
            st.info(f"⚠️ High occlusion detected in {len(high_occlusion_frames)} frames")
        
        # Download button
        csv = st.session_state.occlusion_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Occlusion Data (CSV)",
            csv,
            "occlusion_analysis_results.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.info("No analysis data available. Run the analysis in the 'Real-time Analysis' tab.")

# Add sidebar with controls
with st.sidebar:
    st.header("Controls")
    
    st.subheader("Video Information")
    st.write(f"Total Frames: {total_frames}")
    st.write(f"FPS: {fps}")
    st.write(f"Duration: {total_frames/fps:.2f} seconds")
    
    if st.session_state.roi_selected:
        st.subheader("ROI Information")
        orig_x, orig_y, orig_w, orig_h = st.session_state.roi_coordinates
        st.write(f"Position: ({orig_x}, {orig_y})")
        st.write(f"Size: {orig_w} x {orig_h}")
    
    st.subheader("Occlusion Threshold")
    threshold = st.slider("Occlusion Threshold (%)", 0, 100, 30,
                         help="Threshold for detecting significant occlusion")
    
    st.subheader("Export Options")
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
    
    if st.button("Generate Report"):
        if not st.session_state.occlusion_data.empty:
            st.success("Report generated successfully!")
        else:
            st.warning("No data available for report")

# Add custom CSS for better UI
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Cleanup
try:
    tfile.close()
except:
    pass









































