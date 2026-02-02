import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="DRDO ROI Occlusion System", layout="wide")
st.title("DRDO ROI Occlusion System")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is None:
    st.warning("Please upload a video first.")
    st.stop()

# Save uploaded video
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_video.read())
video_path = tfile.name

# Show uploaded video preview
st.video(video_path)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

st.success(f"Video Loaded Successfully ✅ Total Frames: {total_frames}")

frame_no = st.slider("Select Frame for ROI Selection", 0, total_frames - 1, 0)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ret, frame = cap.read()

if not ret:
    st.error("Could not read frame from video.")
    st.stop()

# Convert to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ---------------- FIX: Resize Frame for Canvas ----------------
original_h, original_w = frame_rgb.shape[:2]

canvas_width = 900  # fixed width for Streamlit canvas
scale = canvas_width / original_w
canvas_height = int(original_h * scale)

frame_resized = cv2.resize(frame_rgb, (canvas_width, canvas_height))
img_pil = Image.fromarray(frame_resized)

st.subheader("Draw Bounding Box on Object (ROI Selection)")

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.2)",
    stroke_width=2,
    stroke_color="red",
    background_image=img_pil,
    update_streamlit=True,
    height=canvas_height,
    width=canvas_width,
    drawing_mode="rect",
    key="canvas",
)

# Extract ROI from canvas
if canvas_result.json_data is None or len(canvas_result.json_data["objects"]) == 0:
    st.info("Draw a rectangle on the object to select ROI.")
    st.stop()

obj = canvas_result.json_data["objects"][-1]

# ROI from resized frame
x_resized = int(obj["left"])
y_resized = int(obj["top"])
w_resized = int(obj["width"])
h_resized = int(obj["height"])

# Convert ROI back to original frame scale
x = int(x_resized / scale)
y = int(y_resized / scale)
w = int(w_resized / scale)
h = int(h_resized / scale)

st.success(f"ROI Selected ✅ x={x}, y={y}, w={w}, h={h}")

# ROI validation
if w <= 10 or h <= 10:
    st.error("ROI too small. Please draw a bigger box.")
    st.stop()

# Run Analysis
if st.button("Run Occlusion Analysis"):
    cap = cv2.VideoCapture(video_path)

    frames_list = []
    occlusion_per_frame = []

    # Read ROI template from selected frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, first_frame = cap.read()

    if not ret:
        st.error("Failed to read selected frame again.")
        st.stop()

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    roi_template = first_gray[y:y+h, x:x+w]

    if roi_template.size == 0:
        st.error("ROI extraction failed. Draw ROI fully inside the frame.")
        st.stop()

    f = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

        roi_now = gray[y:y+h, x:x+w]

        if roi_now.size == 0:
            occlusion_percent = 100
        else:
            roi_now = cv2.resize(roi_now, (roi_template.shape[1], roi_template.shape[0]))
            diff = cv2.absdiff(roi_template, roi_now)

            occlusion_percent = (np.sum(diff > 30) / diff.size) * 100

        frames_list.append(f)
        occlusion_per_frame.append(round(float(occlusion_percent), 2))
        f += 1

    cap.release()

    if len(occlusion_per_frame) == 0:
        st.error("No frames processed. Please upload another video.")
        st.stop()

    df = pd.DataFrame({
        "Frame": frames_list,
        "Occlusion (%)": occlusion_per_frame
    })

    st.subheader("Occlusion Graph (Frame vs Occlusion %)")
    st.line_chart(df.set_index("Frame"))

    st.subheader("Occlusion Data Table")
    st.dataframe(df)

    st.success("Occlusion Analysis Completed ✅")
















