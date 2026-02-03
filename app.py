import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

st.set_page_config(page_title="DRDO ROI Occlusion System", layout="wide")
st.title("DRDO ROI Occlusion System")
st.write("Draw ROI Box on Video Frame")

# Initialize session state
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'roi_coords' not in st.session_state:
    st.session_state.roi_coords = None
if 'frame_no' not in st.session_state:
    st.session_state.frame_no = 0
if 'uploaded_video_name' not in st.session_state:
    st.session_state.uploaded_video_name = None

# Upload video with size limit
st.subheader("Upload Video")
uploaded_video = st.file_uploader(
    "Choose a video file", 
    type=["mp4", "avi", "mov"],
    help="Maximum file size: 200MB"
)

if uploaded_video is None:
    st.warning("Please upload a video first.")
    st.stop()

# Check file size (limit to 200MB for Streamlit Cloud)
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
if uploaded_video.size > MAX_FILE_SIZE:
    st.error(f"File too large! Maximum size is 200MB. Your file is {uploaded_video.size/(1024*1024):.2f}MB")
    st.stop()

# Save uploaded video to temp file
try:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    st.session_state.video_path = video_path
    st.session_state.uploaded_video_name = uploaded_video.name
    
    st.success(f"✅ Video uploaded: {uploaded_video.name} ({uploaded_video.size//1024}KB)")
except Exception as e:
    st.error(f"Error uploading video: {str(e)}")
    st.stop()

# Show uploaded video preview
st.subheader("Uploaded Video Preview")
try:
    st.video(video_path)
except:
    st.warning("Video preview not available")

# Get video properties
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video file")
        st.stop()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0:
        st.error("Invalid video file or cannot read frame count")
        cap.release()
        st.stop()
    
    st.success(f"✅ Video loaded: {total_frames} frames, {width}x{height}, {fps} FPS")
    
except Exception as e:
    st.error(f"Error reading video: {str(e)}")
    st.stop()

# Frame selection
st.subheader("Select Frame for ROI")
frame_no = st.slider("Frame Number", 0, total_frames - 1, min(0, total_frames-1))
st.session_state.frame_no = frame_no

# Read selected frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ret, frame = cap.read()

if not ret:
    st.error("Could not read selected frame")
    cap.release()
    st.stop()

# Convert frame to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_pil = Image.fromarray(frame_rgb)

# Display frame
st.subheader(f"Selected Frame (Frame {frame_no})")
st.image(frame_pil, width=700)

# Draw ROI box
st.subheader("Draw Bounding Box on Object")

# Resize image for canvas
CANVAS_W = 700
scale = CANVAS_W / frame_pil.size[0]
CANVAS_H = int(frame_pil.size[1] * scale)
frame_pil_resized = frame_pil.resize((CANVAS_W, CANVAS_H))

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.2)",
    stroke_width=2,
    stroke_color="red",
    background_image=frame_pil_resized,
    update_streamlit=True,
    height=CANVAS_H,
    width=CANVAS_W,
    drawing_mode="rect",
    key="canvas_roi",
)

# Manual ROI coordinates
st.subheader("Or Enter ROI Coordinates Manually")
col1, col2, col3, col4 = st.columns(4)
with col1:
    manual_x = st.number_input("x (left)", min_value=0, max_value=width-1, value=100)
with col2:
    manual_y = st.number_input("y (top)", min_value=0, max_value=height-1, value=100)
with col3:
    manual_w = st.number_input("width", min_value=1, max_value=width, value=200)
with col4:
    manual_h = st.number_input("height", min_value=1, max_value=height, value=200)

# Get ROI coordinates
if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
    # Use canvas coordinates
    obj = canvas_result.json_data["objects"][-1]
    x = int(obj["left"])
    y = int(obj["top"])
    w = int(obj["width"])
    h = int(obj["height"])
    
    # Convert back to original coordinates
    orig_x = int(x / scale)
    orig_y = int(y / scale)
    orig_w = int(w / scale)
    orig_h = int(h / scale)
    
    st.info(f"📐 Canvas ROI: x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}")
else:
    # Use manual coordinates
    orig_x = manual_x
    orig_y = manual_y
    orig_w = manual_w
    orig_h = manual_h
    st.info(f"📐 Manual ROI: x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}")

# Store ROI coordinates
st.session_state.roi_coords = (orig_x, orig_y, orig_w, orig_h)

# ROI Preview
if (0 <= orig_x < width and 0 <= orig_y < height and 
    orig_w > 0 and orig_h > 0 and
    orig_x + orig_w <= width and orig_y + orig_h <= height):
    
    try:
        roi_preview = frame_rgb[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
        if roi_preview.size > 0:
            st.subheader("ROI Preview")
            st.image(Image.fromarray(roi_preview), width=300)
        else:
            st.warning("Cannot extract ROI preview")
    except:
        st.warning("Cannot extract ROI preview")
else:
    st.warning("⚠️ ROI coordinates are outside frame boundaries")

cap.release()

# Run analysis button
st.markdown("---")
if st.button("🚀 Run Occlusion Analysis", type="primary", use_container_width=True):
    if st.session_state.roi_coords is None:
        st.error("Please select ROI first!")
    elif st.session_state.video_path is None:
        st.error("Video not found!")
    else:
        with st.spinner("Analyzing video frames..."):
            try:
                # Reopen video for analysis
                cap_analysis = cv2.VideoCapture(st.session_state.video_path)
                if not cap_analysis.isOpened():
                    st.error("Cannot open video file for analysis")
                    st.stop()
                
                total_frames_analysis = int(cap_analysis.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames_analysis <= 0:
                    st.error("Invalid video for analysis")
                    cap_analysis.release()
                    st.stop()
                
                # Get ROI coordinates
                orig_x, orig_y, orig_w, orig_h = st.session_state.roi_coords
                
                # Get first frame for template
                cap_analysis.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, first_frame = cap_analysis.read()
                
                if not ret:
                    st.error("Cannot read first frame")
                    cap_analysis.release()
                    st.stop()
                
                # Get frame dimensions
                frame_height, frame_width = first_frame.shape[:2]
                
                # Validate ROI
                if (orig_x >= frame_width or orig_y >= frame_height or 
                    orig_x + orig_w > frame_width or orig_y + orig_h > frame_height or
                    orig_w <= 0 or orig_h <= 0):
                    
                    st.error(f"Invalid ROI. Frame: {frame_width}x{frame_height}, ROI: ({orig_x},{orig_y},{orig_w},{orig_h})")
                    cap_analysis.release()
                    st.stop()
                
                # Create template
                first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                roi_template = first_gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
                
                if roi_template.size == 0:
                    st.error("Cannot extract ROI template")
                    cap_analysis.release()
                    st.stop()
                
                frames_list = []
                occlusion_list = []
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Analyze frames
                for f in range(total_frames_analysis):
                    cap_analysis.set(cv2.CAP_PROP_POS_FRAMES, f)
                    ret, fr = cap_analysis.read()
                    
                    if not ret:
                        break
                    
                    # Calculate occlusion
                    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    
                    # Extract current ROI
                    if (orig_y < gray.shape[0] and orig_y + orig_h <= gray.shape[0] and
                        orig_x < gray.shape[1] and orig_x + orig_w <= gray.shape[1]):
                        
                        roi_now = gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
                        
                        if roi_now.size > 0:
                            roi_now_resized = cv2.resize(roi_now, (roi_template.shape[1], roi_template.shape[0]))
                            diff = cv2.absdiff(roi_template, roi_now_resized)
                            occlusion_percent = (np.sum(diff > 30) / diff.size) * 100
                        else:
                            occlusion_percent = 100
                    else:
                        occlusion_percent = 100
                    
                    frames_list.append(f)
                    occlusion_list.append(round(float(occlusion_percent), 2))
                    
                    # Update progress
                    progress = (f + 1) / total_frames_analysis
                    progress_bar.progress(progress)
                    if f % 10 == 0:  # Update every 10 frames to reduce UI updates
                        status_text.text(f"Processing: {f+1}/{total_frames_analysis} frames")
                
                cap_analysis.release()
                
                # Create results
                df = pd.DataFrame({
                    "Frame": frames_list,
                    "Occlusion (%)": occlusion_list
                })
                
                # Display results
                st.success(f"✅ Analysis Complete! {len(df)} frames analyzed")
                
                # Statistics
                st.subheader("📊 Analysis Statistics")
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Avg Occlusion", f"{df['Occlusion (%)'].mean():.1f}%")
                with cols[1]:
                    st.metric("Max Occlusion", f"{df['Occlusion (%)'].max():.1f}%")
                with cols[2]:
                    st.metric("Min Occlusion", f"{df['Occlusion (%)'].min():.1f}%")
                with cols[3]:
                    st.metric("Frames", len(df))
                
                # Graph
                st.subheader("📈 Occlusion Graph")
                st.line_chart(df.set_index("Frame"))
                
                # Data table
                st.subheader("📋 Data Table")
                st.dataframe(df, use_container_width=True, height=300)
                
                # Download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "occlusion_results.csv",
                    "text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

# Cleanup
try:
    if 'tfile' in locals():
        tfile.close()
        os.unlink(tfile.name)
except:
    pass










































