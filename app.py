import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile
import os

st.set_page_config(page_title="DRDO Occlusion Monitor", layout="wide")

st.title("🎯 DRDO ROI Occlusion Detector")
st.sidebar.header("Settings")

# 1. Upload Video
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 2. Select Reference Frame
    frame_no = st.sidebar.slider("Step 1: Select Reference Frame", 0, total_frames - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, ref_frame = cap.read()

    if ret:
        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        
        st.subheader("Step 2: Draw ROI around Target Object")
        # Canvas for ROI Selection
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=Image.fromarray(ref_rgb),
            update_streamlit=True,
            height=ref_rgb.shape[0],
            width=ref_rgb.shape[1],
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            # Extract coordinates
            obj = canvas_result.json_data["objects"][-1]
            x, y, w, h = int(obj["left"]), int(obj["top"]), int(obj["width"]), int(obj["height"])
            
            # Crop the "Golden Template" (The object we are looking for)
            template = ref_frame[y:y+h, x:x+w]
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            st.sidebar.image(cv2.cvtColor(template, cv2.COLOR_BGR2RGB), caption="Target Template")

            # 3. Process Video
            if st.button("🚀 Run Occlusion Analysis"):
                st.subheader("Analysis Stream")
                
                # Reset video to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Prepare Output
                output_path = "occlusion_output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (ref_frame.shape[1], ref_frame.shape[0]))
                
                # Placeholders for UI
                video_placeholder = st.empty()
                metrics_col1, metrics_col2 = st.columns(2)
                occ_metric = metrics_col1.empty()
                status_metric = metrics_col2.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    # Current ROI area
                    current_roi = frame[y:y+h, x:x+w]
                    current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)

                    # --- CALCULATE OCCLUSION ---
                    # Using Normalized Cross-Correlation to find similarity
                    res = cv2.matchTemplate(current_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    
                    # Convert similarity to occlusion percentage
                    # max_val = 1 means 0% occlusion; max_val = 0 means 100% occlusion
                    occlusion_pct = max(0, (1 - max_val) * 100)
                    
                    # Thresholding (e.g., if > 40% different, mark as occluded)
                    is_occluded = occlusion_pct > 40
                    box_color = (0, 0, 255) if is_occluded else (0, 255, 0)
                    label = "OCCLUDED" if is_occluded else "CLEAR"

                    # --- ANNOTATION ---
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                    cv2.putText(frame, f"{label} ({occlusion_pct:.1f}%)", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    
                    # Save Frame
                    out.write(frame)

                    # Update Live View
                    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 2 == 0:
                        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        occ_metric.metric("Occlusion Level", f"{occlusion_pct:.1f}%")
                        status_metric.metric("Status", label)

                cap.release()
                out.release()
                
                st.success("Analysis Complete!")
                with open(output_path, "rb") as f:
                    st.download_button("⬇️ Download Processed Video", f, "processed_video.mp4")

    cap.release()












































