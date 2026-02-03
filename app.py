import streamlit as st
import cv2
import numpy as np
import tempfile
import subprocess
import os
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ROI Occlusion Analysis System",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h2 style='margin-bottom:0'>ROI Occlusion Analysis System</h2>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- VIDEO UPLOAD ----------------
st.subheader("Upload Input Video")

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mpeg4"]
)

if uploaded_video is None:
    st.info("Please upload a video to continue")
    st.stop()

# Save uploaded video
temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
temp_video.write(uploaded_video.read())
video_path = temp_video.name

# ---------------- LOAD VIDEO ----------------
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ---------------- FRAME SELECT ----------------
st.subheader("Select Frame for ROI Selection")
frame_id = st.slider("Frame Number", 0, total_frames - 1, 0)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
ret, frame = cap.read()

if not ret:
    st.error("Could not read frame")
    st.stop()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(frame_rgb)

# ---------------- LAYOUT ----------------
left_col, right_col = st.columns([1, 1])

# ---------------- LEFT: DRAW ROI ----------------
with left_col:
    st.subheader("Draw Bounding Box on Object")

    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=pil_image,   # MUST be PIL Image
        update_streamlit=True,
        height=pil_image.height,
        width=pil_image.width,
        drawing_mode="rect",
        key="canvas",
    )

    if canvas.json_data and len(canvas.json_data["objects"]) > 0:
        rect = canvas.json_data["objects"][0]

        x = int(rect["left"])
        y = int(rect["top"])
        w = int(rect["width"])
        h = int(rect["height"])

        roi = frame_rgb[y:y+h, x:x+w]

        st.success("ROI Selected")
        st.image(roi, caption="Selected Object", use_column_width=True)

        if st.button("Run Occlusion Analysis"):
            st.info("Processing...")

            subprocess.run(
                ["python", "index.py", video_path, str(x), str(y), str(w), str(h)],
                capture_output=True,
                text=True
            )

# ---------------- RIGHT: OUTPUT ----------------
with right_col:
    st.subheader("Output")

    if os.path.exists("occlusion_graph.png"):
        st.image("occlusion_graph.png", caption="Occlusion Graph")

    if os.path.exists("occlusion_data.txt"):
        with open("occlusion_data.txt") as f:
            st.text(f.read())





























