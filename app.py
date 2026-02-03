import streamlit as st
import cv2
import numpy as np
import tempfile
import subprocess
import os
import io
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

# Original frame size (important for scaling back ROI)
orig_h, orig_w = frame_rgb.shape[:2]

pil_image = Image.fromarray(frame_rgb)

# ---------------- RESIZE FOR CANVAS STABILITY ----------------
MAX_CANVAS_WIDTH = 900  # adjust if needed
scale_x = 1.0
scale_y = 1.0

if pil_image.width > MAX_CANVAS_WIDTH:
    resize_ratio = MAX_CANVAS_WIDTH / pil_image.width
    new_w = int(pil_image.width * resize_ratio)
    new_h = int(pil_image.height * resize_ratio)

    pil_image = pil_image.resize((new_w, new_h))

    # scale factors to map canvas coords -> original frame coords
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

# Convert PIL image to bytes (fixes Streamlit Cloud canvas error)
img_bytes = io.BytesIO()
pil_image.save(img_bytes, format="PNG")
img_bytes.seek(0)
canvas_bg = Image.open(img_bytes)

# ---------------- LAYOUT ----------------
left_col, right_col = st.columns([1, 1])

# ---------------- LEFT: DRAW ROI ----------------
with left_col:
    st.subheader("Draw Bounding Box on Object")

    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=canvas_bg,
        update_streamlit=True,
        height=canvas_bg.height,
        width=canvas_bg.width,
        drawing_mode="rect",
        key="canvas",
    )

    if canvas.json_data and len(canvas.json_data["objects"]) > 0:
        rect = canvas.json_data["objects"][0]

        # Canvas coordinates (on resized image)
        x_canvas = int(rect["left"])
        y_canvas = int(rect["top"])
        w_canvas = int(rect["width"])
        h_canvas = int(rect["height"])

        # Convert to original frame coordinates
        x = int(x_canvas * scale_x)
        y = int(y_canvas * scale_y)
        w = int(w_canvas * scale_x)
        h = int(h_canvas * scale_y)

        # Clamp values to stay inside frame bounds
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        w = max(1, min(w, orig_w - x))
        h = max(1, min(h, orig_h - y))

        roi = frame_rgb[y:y + h, x:x + w]

        st.success("ROI Selected")
        st.image(roi, caption="Selected Object", use_column_width=True)

        if st.button("Run Occlusion Analysis"):
            st.info("Processing... Please wait")

            result = subprocess.run(
                ["python", "index.py", video_path, str(x), str(y), str(w), str(h)],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                st.error("Occlusion analysis failed!")
                st.text("Error details:")
                st.text(result.stderr)
            else:
                st.success("Occlusion analysis completed successfully!")

# ---------------- RIGHT: OUTPUT ----------------
with right_col:
    st.subheader("Output")

    if os.path.exists("occlusion_graph.png"):
        st.image("occlusion_graph.png", caption="Occlusion Graph")

    if os.path.exists("occlusion_data.txt"):
        with open("occlusion_data.txt") as f:
            st.text(f.read())































