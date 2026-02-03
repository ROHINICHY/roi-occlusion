import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="ROI Occlusion System", layout="wide")
st.title("🎯 ROI Occlusion System (Object Occlusion Analysis)")

uploaded_video = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is None:
    st.warning("Please upload a video first.")
    st.stop()

# Save uploaded video
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded_video.read())
video_path = tfile.name

st.subheader("🎬 Uploaded Video Preview")
st.video(video_path)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames <= 0:
    st.error("❌ Could not read video frames. Try another video.")
    st.stop()

st.success(f"✅ Video Loaded | Total Frames: {total_frames}")

frame_no = st.slider("🎞 Select Frame for ROI Selection", 0, total_frames - 1, 0)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ret, frame = cap.read()

if not ret:
    st.error("❌ Could not read selected frame.")
    st.stop()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(frame_rgb)

st.subheader("🟥 Draw Bounding Box on Object (ROI Selection)")

# IMPORTANT: background_image must be PIL Image (not numpy)
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.2)",
    stroke_width=2,
    stroke_color="red",
    background_image=img_pil,
    update_streamlit=True,
    height=img_pil.height,
    width=img_pil.width,
    drawing_mode="rect",
    key="canvas",
)

# ROI extraction
if canvas_result.json_data is None:
    st.info("✍️ Draw a rectangle on the object to select ROI.")
    st.stop()

objects = canvas_result.json_data.get("objects", [])
if len(objects) == 0:
    st.info("✍️ Draw a rectangle on the object to select ROI.")
    st.stop()

obj = objects[-1]

x = int(obj["left"])
y = int(obj["top"])
w = int(obj["width"])
h = int(obj["height"])

if w <= 5 or h <= 5:
    st.error("❌ ROI too small. Please draw a bigger rectangle.")
    st.stop()

st.success("✅ ROI Selected Successfully!")

if st.button("🚀 Run Occlusion Analysis"):
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, first_frame = cap.read()
    if not ret:
        st.error("❌ Failed to read selected frame again.")
        st.stop()

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    roi_template = first_gray[y:y+h, x:x+w]

    if roi_template.size == 0:
        st.error("❌ ROI extraction failed. Draw ROI fully inside the frame.")
        st.stop()

    frames_list = []
    occlusion_list = []

    f = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        roi_now = gray[y:y+h, x:x+w]

        if roi_now.size == 0:
            occlusion_percent = 100.0
        else:
            roi_now = cv2.resize(roi_now, (roi_template.shape[1], roi_template.shape[0]))
            diff = cv2.absdiff(roi_template, roi_now)
            occlusion_percent = (np.sum(diff > 30) / diff.size) * 100

        frames_list.append(f)
        occlusion_list.append(round(float(occlusion_percent), 2))
        f += 1

    cap.release()

    df = pd.DataFrame({
        "Frame": frames_list,
        "Occlusion (%)": occlusion_list
    })

    st.subheader("📈 Occlusion Graph (Frame vs Occlusion %)")
    st.line_chart(df.set_index("Frame"))

    st.subheader("📋 Occlusion Data Table")
    st.dataframe(df, use_container_width=True)

    st.success("✅ Analysis Completed Successfully!")


































