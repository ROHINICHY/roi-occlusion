import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile
import pandas as pd

st.set_page_config(page_title="DRDO Occlusion Monitor", layout="wide")

st.title("🎯 DRDO ROI Occlusion Detector with Real-time Graphing")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sidebar Settings
    st.sidebar.header("Configuration")
    frame_no = st.sidebar.slider("Step 1: Select Reference Frame", 0, total_frames - 1, 0)
    sensitivity = st.sidebar.slider("Occlusion Sensitivity (Threshold)", 0, 100, 40)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, ref_frame = cap.read()

    if ret:
        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        
        st.subheader("Step 2: Draw ROI around Target Object")
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
            obj = canvas_result.json_data["objects"][-1]
            x, y, w, h = int(obj["left"]), int(obj["top"]), int(obj["width"]), int(obj["height"])
            
            # Extract Template
            template = ref_frame[y:y+h, x:x+w]
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            if st.button("🚀 Run Full Analysis"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Layout for Live Updates
                col_vid, col_graph = st.columns([2, 1])
                
                with col_vid:
                    video_placeholder = st.empty()
                    status_placeholder = st.empty()
                
                with col_graph:
                    st.write("**Occlusion History (%)**")
                    chart_placeholder = st.empty()
                    metric_placeholder = st.empty()

                # Data storage for the graph
                occlusion_history = []
                
                output_path = "occlusion_output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (ref_frame.shape[1], ref_frame.shape[0]))

                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    # Match Template
                    current_roi = frame[y:y+h, x:x+w]
                    if current_roi.shape[0] == h and current_roi.shape[1] == w:
                        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
                        res = cv2.matchTemplate(current_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        
                        occ_pct = round(max(0, (1 - max_val) * 100), 2)
                    else:
                        occ_pct = 100.0  # Out of bounds/Error
                    
                    occlusion_history.append(occ_pct)
                    
                    # UI Indicators
                    is_occ = occ_pct > sensitivity
                    color = (0, 0, 255) if is_occ else (0, 255, 0)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(frame, f"OCC: {occ_pct}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    out.write(frame)

                    # Update UI every 3 frames for performance
                    if frame_idx % 3 == 0:
                        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        metric_placeholder.metric("Current Occlusion", f"{occ_pct}%", delta=f"{occ_pct - (occlusion_history[-2] if len(occlusion_history)>1 else 0):.1f}%", delta_color="inverse")
                        
                        # Update Chart
                        chart_placeholder.line_chart(occlusion_history)
                        
                        status_msg = "⚠️ OBJECT BLOCKED" if is_occ else "✅ PATH CLEAR"
                        status_placeholder.subheader(status_msg)

                    frame_idx += 1

                cap.release()
                out.release()
                
                st.success("Analysis Finished!")
                with open(output_path, "rb") as f:
                    st.download_button("⬇️ Download Video", f, "output.mp4")

    cap.release()













































