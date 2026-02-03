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

# Upload video
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is None:
    st.warning("Please upload a video first.")
    st.stop()

# Save uploaded video
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_video.read())
video_path = tfile.name

# Show uploaded video preview
st.subheader("Uploaded Video Preview")
st.video(video_path)

# Get video properties
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

st.success(f"Video Loaded Successfully ✅ Total Frames: {total_frames}")

# Frame selection
frame_no = st.slider("Select Frame for ROI Selection", 0, total_frames - 1, 0)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ret, frame = cap.read()

if not ret:
    st.error("Could not read frame from video.")
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
    manual_x = st.number_input("x (left)", min_value=0, max_value=frame.shape[1]-1, value=66)
with col2:
    manual_y = st.number_input("y (top)", min_value=0, max_value=frame.shape[0]-1, value=42)
with col3:
    manual_w = st.number_input("width (w)", min_value=1, max_value=frame.shape[1], value=200)
with col4:
    manual_h = st.number_input("height (h)", min_value=1, max_value=frame.shape[0], value=150)

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
else:
    # Use manual coordinates
    orig_x = manual_x
    orig_y = manual_y
    orig_w = manual_w
    orig_h = manual_h

st.info(f"ROI Coordinates: x={orig_x}, y={orig_y}, width={orig_w}, height={orig_h}")

# ROI Preview
roi_preview = frame_rgb[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
if roi_preview.size > 0:
    st.subheader("ROI Preview")
    st.image(Image.fromarray(roi_preview), width=400)

# Run analysis
if st.button("Run Occlusion Analysis"):
    cap = cv2.VideoCapture(video_path)
    
    # Get template from first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    
    if not ret:
        st.error("Failed to read first frame.")
        st.stop()
    
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    roi_template = first_gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
    
    if roi_template.size == 0:
        st.error("ROI template extraction failed.")
        st.stop()
    
    frames_list = []
    occlusion_list = []
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Analyze all frames
    for f in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, fr = cap.read()
        
        if not ret:
            break
        
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        roi_now = gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
        
        if roi_now.size > 0:
            roi_now = cv2.resize(roi_now, (roi_template.shape[1], roi_template.shape[0]))
            diff = cv2.absdiff(roi_template, roi_now)
            occlusion_percent = (np.sum(diff > 30) / diff.size) * 100
        else:
            occlusion_percent = 100
        
        frames_list.append(f)
        occlusion_list.append(round(float(occlusion_percent), 2))
        
        # Update progress
        progress_bar.progress((f + 1) / total_frames)
    
    cap.release()
    
    # Create dataframe
    df = pd.DataFrame({
        "Frame": frames_list,
        "Occlusion (%)": occlusion_list
    })
    
    # Display graph
    st.subheader("Occlusion Graph (Frame vs Occlusion %)")
    st.line_chart(df.set_index("Frame"))
    
    # Display table
    st.subheader("Occlusion Data Table")
    st.dataframe(df)
    
    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Occlusion Data (CSV)", csv, "occlusion_data.csv", "text/csv")
    
    st.success("Occlusion Analysis Completed ✅")

# Cleanup
try:
    cap.release()
    tfile.close()
except:
    pass









































