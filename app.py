import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile
import io

st.set_page_config(page_title="DRDO ROI Occlusion", layout="wide")

st.title("🎯 DRDO ROI Occlusion Detection (ROI Selection + Output Video)")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:

    # Save uploaded video to temp file
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
        st.error("❌ Could not read the selected frame!")
        st.stop()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    st.subheader(f"Selected Frame Preview (Frame No: {frame_no})")
    st.image(pil_img, use_container_width=True)

    st.subheader("🟥 Draw Bounding Box on Object (ROI Selection)")

    # Convert PIL image to bytes (IMPORTANT FIX FOR STREAMLIT CLOUD)
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    background_img = Image.open(img_bytes)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=3,
        stroke_color="red",
        background_image=background_img,   # ✅ FIXED
        update_streamlit=True,
        height=pil_img.size[1],
        width=pil_img.size[0],
        drawing_mode="rect",
        key="canvas",
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])

        if len(objects) > 0:
            rect = objects[-1]

            x = int(rect["left"])
            y = int(rect["top"])
            w = int(rect["width"])
            h = int(rect["height"])

            st.success(f"✅ ROI Selected: x={x}, y={y}, w={w}, h={h}")

            roi_crop = frame_rgb[y:y+h, x:x+w]
            if roi_crop.size > 0:
                st.subheader("📌 ROI Preview (Selected Object)")
                st.image(roi_crop, use_container_width=False)

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

        else:
            st.warning("⚠️ Please draw a rectangle on the frame to select ROI.")


































