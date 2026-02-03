import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile
import os

st.set_page_config(page_title="DRDO ROI Occlusion", layout="wide")

st.title("🎯 DRDO ROI Occlusion Detection")
st.markdown("Select an ROI to monitor for occlusions in the video stream.")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Frame Selection for ROI
    frame_no = st.sidebar.slider("Select Frame for ROI", 0, total_frames - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Canvas for ROI Selection
        st.subheader("1. Define Target ROI")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=3,
            stroke_color="red",
            background_image=pil_img,
            update_streamlit=True,
            height=pil_img.size[1],
            width=pil_img.size[0],
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            rect = canvas_result.json_data["objects"][-1]
            x, y, w, h = int(rect["left"]), int(rect["top"]), int(rect["width"]), int(rect["height"])
            
            # Save the reference ROI for comparison
            ref_roi = frame[y:y+h, x:x+w]
            ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY)

            st.success(f"ROI Locked at [{x}, {y}, {w}, {h}]")

            if st.button("▶️ Analyze Video for Occlusion"):
                st.subheader("2. Analysis Output")
                
                # Setup for processing
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                output_path = "processed_occlusion.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
                
                prog_bar = st.progress(0)
                status_text = st.empty()
                video_placeholder = st.empty()

                frame_count = 0
                while cap.isOpened():
                    ret, frm = cap.read()
                    if not ret: break

                    # Extract current frame ROI
                    current_roi = frm[y:y+h, x:x+w]
                    current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)

                    # --- SIMPLE OCCLUSION LOGIC ---
                    # Calculate absolute difference between reference and current
                    diff = cv2.absdiff(ref_gray, current_gray)
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    occlusion_pct = (np.sum(thresh == 255) / thresh.size) * 100
                    
                    # Determine Status
                    status = "OCCLUDED" if occlusion_pct > 25 else "CLEAR" # Threshold 25%
                    color = (0, 0, 255) if status == "OCCLUDED" else (0, 255, 0)

                    # Annotate Frame
                    cv2.rectangle(frm, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(frm, f"Status: {status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frm, f"Occlusion: {occlusion_pct:.1f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    out.write(frm)
                    
                    # Update UI intermittently to save resources
                    if frame_count % 5 == 0:
                        video_placeholder.image(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB), channels="RGB")
                        prog_bar.progress(frame_count / total_frames)
                        status_text.text(f"Processing Frame {frame_count}/{total_frames} | Occlusion: {occlusion_pct:.1f}%")
                    
                    frame_count += 1

                cap.release()
                out.release()
                prog_bar.progress(1.0)
                st.success("Analysis Complete!")

                with open(output_path, "rb") as f:
                    st.download_button("⬇️ Download Result", f, "occlusion_analysis.mp4")

    cap.release()











































