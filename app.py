import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from PIL import Image

st.set_page_config(page_title="DRDO ROI Occlusion System", layout="wide")
st.title("DRDO ROI Occlusion System (Stable Version)")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is None:
    st.warning("Please upload a video first.")
    st.stop()

# Save uploaded video
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_video.read())
video_path = tfile.name

# Show uploaded video
st.subheader("Uploaded Video Preview")
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

# Convert frame to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Convert to PIL (Fix Streamlit TypeError)
frame_pil = Image.fromarray(frame_rgb)

st.subheader(f"Selected Frame Preview (Frame No: {frame_no})")
st.image(frame_pil, use_container_width=True)

h_img, w_img = frame_rgb.shape[:2]

st.subheader("Enter ROI Coordinates Manually (x, y, width, height)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    x = st.number_input("x (left)", min_value=0, max_value=w_img - 1, value=10)
with col2:
    y = st.number_input("y (top)", min_value=0, max_value=h_img - 1, value=10)
with col3:
    w = st.number_input("width (w)", min_value=1, max_value=w_img, value=100)
with col4:
    h = st.number_input("height (h)", min_value=1, max_value=h_img, value=100)

# Validate ROI bounds
if x + w > w_img:
    st.error("❌ ROI width goes outside frame. Reduce w or x.")
    st.stop()

if y + h > h_img:
    st.error("❌ ROI height goes outside frame. Reduce h or y.")
    st.stop()

# Show ROI preview
roi_preview = frame_rgb[y:y+h, x:x+w]

roi_pil = Image.fromarray(roi_preview)

st.subheader("ROI Preview (Selected Object)")
st.image(roi_pil, caption="This is your selected ROI object", use_container_width=False)

st.success(f"ROI Selected ✅ x={x}, y={y}, w={w}, h={h}")

# Run Analysis
if st.button("Run Occlusion Analysis"):
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, first_frame = cap.read()

    if not ret:
        st.error("Failed to read selected frame again.")
        st.stop()

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    roi_template = first_gray[y:y+h, x:x+w]

    if roi_template.size == 0:
        st.error("ROI extraction failed. Please correct ROI coordinates.")
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
            occlusion_percent = 100
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

    st.subheader("Occlusion Graph (Frame vs Occlusion %)")
    st.line_chart(df.set_index("Frame"))

    st.subheader("Occlusion Data Table")
    st.dataframe(df)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Occlusion Data (CSV)", csv, "occlusion_data.csv", "text/csv")

    st.success("Occlusion Analysis Completed ✅")




















