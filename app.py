import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd

st.set_page_config(page_title="ROI Occlusion System", layout="wide")

st.title("🎯 ROI Occlusion System")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is None:
    st.warning("Please upload a video first.")
    st.stop()

# Save uploaded video to temp file
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_video.read())
video_path = tfile.name

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

st.success(f"✅ Video Loaded Successfully | Total Frames: {total_frames} | FPS: {round(fps, 2)}")

frame_no = st.slider("Select Frame for ROI Selection", 0, total_frames - 1, 0)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ret, frame = cap.read()

if not ret:
    st.error("❌ Could not read selected frame from video.")
    st.stop()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

st.subheader(f"📌 Selected Frame Preview (Frame No: {frame_no})")

# Get image size
h_img, w_img, _ = frame_rgb.shape

# ROI Selection Inputs
st.subheader("🟥 Select ROI (Object)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    x = st.number_input("x (left)", min_value=0, max_value=w_img - 1, value=50)
with col2:
    y = st.number_input("y (top)", min_value=0, max_value=h_img - 1, value=50)
with col3:
    w = st.number_input("width", min_value=1, max_value=w_img - int(x), value=100)
with col4:
    h = st.number_input("height", min_value=1, max_value=h_img - int(y), value=100)

# Convert to int
x, y, w, h = int(x), int(y), int(w), int(h)

# Draw ROI on preview
preview = frame_rgb.copy()
cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 3)

st.image(preview, caption="ROI Preview (Rectangle on Frame)", use_container_width=True)

# ROI Crop Preview
roi_crop = frame_rgb[y:y+h, x:x+w]
if roi_crop.size > 0:
    st.subheader("📌 ROI Crop Preview (Selected Object)")
    st.image(roi_crop, use_container_width=False)

# Run Analysis Button
st.subheader("📊 Occlusion Analysis")

if st.button("Run Occlusion Analysis"):
    cap.release()
    cap = cv2.VideoCapture(video_path)

    # Read template ROI from selected frame again
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, first_frame = cap.read()

    if not ret:
        st.error("❌ Failed to read selected frame again.")
        st.stop()

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    roi_template = first_gray[y:y+h, x:x+w]

    if roi_template.size == 0:
        st.error("❌ ROI extraction failed. Please keep ROI inside the frame.")
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

    if len(occlusion_list) == 0:
        st.error("❌ No frames processed. Please upload another video.")
        st.stop()

    df = pd.DataFrame({
        "Frame": frames_list,
        "Occlusion (%)": occlusion_list
    })

    st.success("✅ Occlusion Analysis Completed!")

    st.subheader("📈 Occlusion Graph (Frame vs Occlusion %)")
    st.line_chart(df.set_index("Frame"))

    st.subheader("📋 Occlusion Data Table")
    st.dataframe(df, use_container_width=True)
































