import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="DRDO ROI Occlusion System", layout="wide")
st.title("DRDO ROI Occlusion System (Draw ROI Box)")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is None:
    st.warning("Please upload a video first.")
    st.stop()

# Save uploaded video
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_video.read())
video_path = tfile.name

# Show uploaded video preview
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
frame_pil = Image.fromarray(frame_rgb)

st.subheader(f"Selected Frame Preview (Frame No: {frame_no})")
st.image(frame_pil, width=900)

# Resize image for canvas (Cloud safe)
CANVAS_W = 900
scale = CANVAS_W / frame_pil.size[0]
CANVAS_H = int(frame_pil.size[1] * scale)

frame_pil_resized = frame_pil.resize((CANVAS_W, CANVAS_H))

st.subheader("Draw Bounding Box on Object (ROI Selection)")

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.2)",
    stroke_width=2,
    stroke_color="red",
    background_image=frame_pil_resized,
    update_streamlit=True,
    height=CANVAS_H,
    width=CANVAS_W,
    drawing_mode="rect",
    key="canvas_roi",
)

# Check if rectangle is drawn
if canvas_result.json_data is None or len(canvas_result.json_data["objects"]) == 0:
    st.info("✍️ Please draw a rectangle on the object to select ROI.")
    st.stop()

# Get last rectangle
obj = canvas_result.json_data["objects"][-1]

# ROI coordinates from resized canvas
x = int(obj["left"])
y = int(obj["top"])
w = int(obj["width"])
h = int(obj["height"])

# Convert back to original frame coordinates
orig_x = int(x / scale)
orig_y = int(y / scale)
orig_w = int(w / scale)
orig_h = int(h / scale)

st.success(f"ROI Selected ✅ x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}")

# Validate ROI size
if orig_w <= 5 or orig_h <= 5:
    st.error("ROI too small. Please draw a bigger box.")
    st.stop()

# ROI Preview
roi_preview = frame_rgb[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]

if roi_preview.size == 0:
    st.error("ROI extraction failed. Please draw ROI inside frame.")
    st.stop()

st.subheader("ROI Preview (Selected Object)")
st.image(Image.fromarray(roi_preview), width=400)

# Run analysis
if st.button("Run Occlusion Analysis"):
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, first_frame = cap.read()

    if not ret:
        st.error("Failed to read selected frame again.")
        st.stop()

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    roi_template = first_gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]

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
        roi_now = gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]

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

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Occlusion Data (CSV)", csv, "occlusion_data.csv", "text/csv")

    st.success("Occlusion Analysis Completed ✅")























