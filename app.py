import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from PIL import Image

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

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_pil = Image.fromarray(frame_rgb)

st.subheader(f"Selected Frame Preview (Frame No: {frame_no})")
st.image(frame_pil, width=900)

st.markdown("## Select ROI using Two Clicks (Top-Left and Bottom-Right)")
st.info("Step 1: Enter Top-Left (x1,y1)\n\nStep 2: Enter Bottom-Right (x2,y2)")

h_img, w_img = frame_rgb.shape[:2]

col1, col2 = st.columns(2)

with col1:
    x1 = st.number_input("Top-Left X (x1)", min_value=0, max_value=w_img-1, value=0)
    y1 = st.number_input("Top-Left Y (y1)", min_value=0, max_value=h_img-1, value=0)

with col2:
    x2 = st.number_input("Bottom-Right X (x2)", min_value=0, max_value=w_img-1, value=min(100, w_img-1))
    y2 = st.number_input("Bottom-Right Y (y2)", min_value=0, max_value=h_img-1, value=min(100, h_img-1))

# Validate ROI
if x2 <= x1 or y2 <= y1:
    st.error("❌ Invalid ROI: x2 must be > x1 and y2 must be > y1")
    st.stop()

roi_preview = frame_rgb[int(y1):int(y2), int(x1):int(x2)]

st.subheader("ROI Preview (Selected Object)")
st.image(Image.fromarray(roi_preview), width=400)

if st.button("Run Occlusion Analysis"):
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, first_frame = cap.read()
    if not ret:
        st.error("Failed to read selected frame again.")
        st.stop()

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    roi_template = first_gray[int(y1):int(y2), int(x1):int(x2)]

    if roi_template.size == 0:
        st.error("ROI template extraction failed.")
        st.stop()

    frames_list = []
    occlusion_list = []

    f = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        roi_now = gray[int(y1):int(y2), int(x1):int(x2)]

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

    df = pd.DataFrame({"Frame": frames_list, "Occlusion (%)": occlusion_list})

    st.subheader("Occlusion Graph (Frame vs Occlusion %)")
    st.line_chart(df.set_index("Frame"))

    st.subheader("Occlusion Data Table")
    st.dataframe(df)

    st.success("Occlusion Analysis Completed ✅")

























