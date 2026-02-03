import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile
import pandas as pd

st.set_page_config(page_title="DRDO Occlusion Monitor", layout="wide")

st.title("🎯 DRDO ROI Occlusion Monitor")

uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sidebar: Frame Selection
    st.sidebar.header("🕹️ Controls")
    frame_no = st.sidebar.slider("Select Frame for ROI", 0, total_frames - 1, 0)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    
    if ret:
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Calculate scaling to make sure the image fits the screen
        max_display_width = 1000
        scale_ratio = max_display_width / pil_img.width
        display_width = int(pil_img.width * scale_ratio)
        display_height = int(pil_img.height * scale_ratio)
        
        # Resize image for the Canvas display
        preview_img = pil_img.resize((display_width, display_height))
        
        st.subheader("📍 Step 1: Draw ROI on the Object")
        st.caption(f"Original Resolution: {frame.shape[1]}x{frame.shape[0]} | Preview Scaling: {scale_ratio:.2f}")

        # --- DRAWING CANVAS ---
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=preview_img,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key="roi_canvas",
        )

        # Check if ROI is drawn
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][-1]
            
            # Re-scale coordinates back to original video size
            x = int(obj["left"] / scale_ratio)
            y = int(obj["top"] / scale_ratio)
            w = int(obj["width"] / scale_ratio)
            h = int(obj["height"] / scale_ratio)

            # Display ROI Metrics
            st.success(f"✅ ROI Captured: X={x}, Y={y}, Width={w}, Height={h}")
            
            # Template for matching
            template = frame[y:y+h, x:x+w]
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            if st.button("🚀 Start Live Analysis"):
                st.divider()
                
                # Layout for Live Video and Graph
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 🎥 Live Video Feed")
                    video_placeholder = st.empty()
                
                with col2:
                    st.markdown("### 📈 Occlusion Graph (%)")
                    chart_placeholder = st.empty()
                    metric_placeholder = st.empty()

                # Processing variables
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                occlusion_history = []
                
                # Process video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 1. Extract current ROI area
                    current_roi = frame[y:y+h, x:x+w]
                    
                    if current_roi.size > 0:
                        # 2. Compare template to current frame
                        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
                        res = cv2.matchTemplate(current_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        
                        # Similarity to Occlusion Percentage
                        occ_pct = round(max(0, (1 - max_val) * 100), 2)
                    else:
                        occ_pct = 100.0

                    occlusion_history.append(occ_pct)
                    
                    # 3. Draw on video frame
                    box_color = (0, 0, 255) if occ_pct > 30 else (0, 255, 0) # Red if blocked
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 4)
                    cv2.putText(frame, f"OCC: {occ_pct}%", (x, y-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

                    # 4. Update Streamlit UI (Live)
                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Update Graph with only the last 100 data points for better visibility
                    chart_placeholder.line_chart(occlusion_history)
                    
                    # Update Metric
                    status_text = "⚠️ OCCLUDED" if occ_pct > 30 else "✅ CLEAR"
                    metric_placeholder.metric("Current Occlusion", f"{occ_pct}%", status_text)

                cap.release()
                st.balloons()
                st.success("Video processing complete!")

    cap.release()

















































