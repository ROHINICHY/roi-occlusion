import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

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

# Upload video
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is None:
    st.warning("Please upload a video first.")
    st.stop()

# Save uploaded video to temp file
tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
tfile.write(uploaded_video.read())
video_path = tfile.name
st.session_state.video_path = video_path

# Show uploaded video preview
st.subheader("Uploaded Video Preview")
st.video(video_path)

# Get video properties
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

st.success(f"Video Loaded Successfully ✅ Total Frames: {total_frames}, Size: {width}x{height}")

# Frame selection
frame_no = st.slider("Select Frame for ROI Selection", 0, total_frames - 1, 0)
st.session_state.frame_no = frame_no

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ret, frame = cap.read()

if not ret:
    st.error("Could not read frame from video.")
    cap.release()
    st.stop()

# Convert frame to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_pil = Image.fromarray(frame_rgb)

st.subheader(f"Selected Frame Preview (Frame No: {frame_no})")
st.image(frame_pil, width=900)

# Draw ROI box
st.subheader("Draw Bounding Box on Object")

# Resize image for canvas
CANVAS_W = 900
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
st.subheader("Enter ROI Coordinates Manually")
col1, col2, col3, col4 = st.columns(4)
with col1:
    manual_x = st.number_input("x (left)", min_value=0, max_value=width-1, value=100)
with col2:
    manual_y = st.number_input("y (top)", min_value=0, max_value=height-1, value=100)
with col3:
    manual_w = st.number_input("width (w)", min_value=1, max_value=width, value=200)
with col4:
    manual_h = st.number_input("height (h)", min_value=1, max_value=height, value=200)

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
    
    st.info(f"Canvas ROI: x={orig_x}, y={orig_y}, width={orig_w}, height={orig_h}")
else:
    # Use manual coordinates
    orig_x = manual_x
    orig_y = manual_y
    orig_w = manual_w
    orig_h = manual_h
    st.info(f"Manual ROI: x={orig_x}, y={orig_y}, width={orig_w}, height={orig_h}")

# Store ROI coordinates in session state
st.session_state.roi_coords = (orig_x, orig_y, orig_w, orig_h)

# ROI Preview - Ensure coordinates are valid
if (orig_x < width and orig_y < height and 
    orig_x + orig_w <= width and orig_y + orig_h <= height and
    orig_w > 0 and orig_h > 0):
    
    roi_preview = frame_rgb[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
    if roi_preview.size > 0:
        st.subheader("ROI Preview")
        st.image(Image.fromarray(roi_preview), width=400)
    else:
        st.warning("ROI preview extraction failed. Please check coordinates.")
else:
    st.warning("ROI coordinates are outside frame boundaries. Please adjust.")

cap.release()

# Run analysis
if st.button("Run Occlusion Analysis", type="primary"):
    if st.session_state.roi_coords is None:
        st.error("Please select ROI first!")
    elif st.session_state.video_path is None:
        st.error("Video not found!")
    else:
        with st.spinner("Analyzing video frames..."):
            # Reopen video for analysis
            cap_analysis = cv2.VideoCapture(st.session_state.video_path)
            total_frames_analysis = int(cap_analysis.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Get ROI coordinates
            orig_x, orig_y, orig_w, orig_h = st.session_state.roi_coords
            
            # Validate ROI coordinates
            if orig_w <= 0 or orig_h <= 0:
                st.error("Invalid ROI dimensions. Please select a valid ROI.")
                cap_analysis.release()
                st.stop()
            
            # Get template from first frame
            cap_analysis.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame = cap_analysis.read()
            
            if not ret:
                st.error("Failed to read first frame.")
                cap_analysis.release()
                st.stop()
            
            # Check if ROI is within frame boundaries
            frame_height, frame_width = first_frame.shape[:2]
            if (orig_x >= frame_width or orig_y >= frame_height or 
                orig_x + orig_w > frame_width or orig_y + orig_h > frame_height):
                st.error(f"ROI is outside frame boundaries. Frame: {frame_width}x{frame_height}, ROI: ({orig_x},{orig_y},{orig_w},{orig_h})")
                cap_analysis.release()
                st.stop()
            
            first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            roi_template = first_gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
            
            if roi_template.size == 0:
                st.error("ROI template extraction failed. ROI may be outside frame.")
                cap_analysis.release()
                st.stop()
            
            frames_list = []
            occlusion_list = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze all frames
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
                        # Resize to match template size
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
                status_text.text(f"Processing frame {f+1}/{total_frames_analysis}...")
            
            cap_analysis.release()
            
            # Create dataframe
            df = pd.DataFrame({
                "Frame": frames_list,
                "Occlusion (%)": occlusion_list
            })
            
            # Display graph
            st.subheader("Occlusion Graph (Frame vs Occlusion %)")
            st.line_chart(df.set_index("Frame"))
            
            # Display statistics
            st.subheader("Analysis Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Occlusion", f"{df['Occlusion (%)'].mean():.2f}%")
            with col2:
                st.metric("Max Occlusion", f"{df['Occlusion (%)'].max():.2f}%")
            with col3:
                st.metric("Min Occlusion", f"{df['Occlusion (%)'].min():.2f}%")
            with col4:
                st.metric("Frames Analyzed", len(df))
            
            # Display table
            st.subheader("Occlusion Data Table")
            st.dataframe(df)
            
            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Occlusion Data (CSV)", csv, "occlusion_data.csv", "text/csv")
            
            st.success("Occlusion Analysis Completed ✅")

# Cleanup
try:
    tfile.close()
except:
    pass










































