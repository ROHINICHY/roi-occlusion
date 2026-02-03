import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="DRDO ROI Occlusion", layout="wide")

st.title("🎯 DRDO ROI Occlusion Detection")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:

    # Save video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    st.success(f"✅ Video Loaded Successfully | Total Frames: {total_frames}")

    frame_no = st.slider("Select Frame for ROI Selection", 0, total_frames - 1, 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()

    if not ret:
        st.error("❌ Could not read selected frame.")
        st.stop()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.subheader(f"Selected Frame Preview (Frame No: {frame_no})")

    # Image size
    h_img, w_img, _ = frame_rgb.shape

    # ROI Inputs
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

    # Draw ROI rectangle on frame for preview
    preview = frame_rgb.copy()
    x, y, w, h = int(x), int(y), int(w), int(h)

    cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 3)

    st.image(preview, caption="ROI Preview (Rectangle on Frame)", use_container_width=True)

    # ROI crop preview
    roi_crop = frame_rgb[y:y+h, x:x+w]
    if roi_crop.size > 0:
        st.subheader("📌 ROI Crop Preview (Selected Object)")
        st.image(roi_crop, use_container_width=False)

    # Generate output video
    if st.button("▶️ Generate Output Video with ROI Highlight"):
        cap.release()
        cap = cv2.VideoCapture(tfile.name)

        output_path = "output_roi_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frm = cap.read()
            if not ret:
                break

            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 0, 255), 2)
            out.write(frm)

        cap.release()
        out.release()

        st.success("✅ Output video generated successfully!")

        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ Download Output Video",
                data=f,
                file_name="roi_output.mp4",
                mime="video/mp4"
            )





















































