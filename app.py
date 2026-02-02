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
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded_video.read())
video_path = tfile.name

# ✅ Show video preview
st.subheader("Uploaded Video Preview")
st.video(video_path)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    st.error("❌ OpenCV cannot open this video. Please upload another MP4 video.")
    st.stop()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames == 0:
    st.error("❌ Video frames are not readable. Please upload a different MP4 file.")
    st.stop()

st.success(f"Video Loaded Successfully ✅ Total Frames: {total_frames}")

frame_no = st.slider("Select Frame for ROI Selection", 0, total_frames - 1, 0)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ret, frame = cap.read()

if not ret or frame is None:
    st.error("❌ Could not read selected frame. Please upload another video.")
    st.stop()

# Convert frame to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(frame_rgb)

st.subheader("Draw Bounding Box on Object")

# Resize for Streamlit Cloud
display_width = 700
aspect_ratio = img_pil.size[1] / img_pil.size[0]
display_height = int(display_width * aspect_ratio)

img_display = img_pil.resize((display_width, display_height))

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.2)",
    stroke_width=2,
    stroke_color="red",
    background_image=img_display,
    update_streamlit=True,
    height=display_height,
    width=display_width,
    drawing_mode="rect",
    key="canvas",
)

if canvas_result.json_data is None or len(canvas_result.json_data["objects"]) == 0:
    st.info("Draw a rectangle on the object to select ROI.")
    st.stop()

obj = canvas_result.json_data["objects"][-1]

# Canvas coordinates
x1 = int(obj["left"])
y1 = int(obj["top"])
w1 = int(obj["width"])
h1 = int(obj["height"])

# Convert to original frame coordinates
scale_x = img_pil.size[0] / display_width
scale_y = img_pil.size[1] / display_height

x = int(x1 * scale_x)
y = int(y1 * scale_y)
w = int(w1 * scale_x)
h = int(h1 * scale_y)

st.success(f"ROI Selected ✅ x={x}, y={y}, w={w}, h={h}")

if w <= 5 or h <= 5:
    st.error("ROI too small. Please draw a bigger box.")
    st.stop()

# Run Analysis
if st.button("Run Occlusion Analysis"):
    cap = cv2.VideoCapture(video_path)

    frames_list = []
    occlusion_per_frame = []

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

    df = pd.DataFrame({
        "Frame": frames_list,
        "Occlusion (%)": occlusion_per_frame
    })

    st.subheader("Occlusion Graph (Frame vs Occlusion %)")
    st.line_chart(df.set_index("Frame"))

    st.subheader("Occlusion Data Table")
    st.dataframe(df)

    st.success("Occlusion Analysis Completed ✅")










