import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile

st.set_page_config(page_title="DRDO ROI Occlusion Monitor", layout="wide")

st.title("🎯 DRDO ROI Occlusion Monitor (Manual & Canvas Selection)")

uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🕹️ ROI Configuration")
    frame_no = st.sidebar.slider("Select Frame for Reference", 0, total_frames - 1, 0)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔢 Manual Coordinate Input")
    # Manual Input Fields
    m_x = st.sidebar.number_input("X Coordinate", 0, width, 100)
    m_y = st.sidebar.number_input("Y Coordinate", 0, height, 100)
    m_w = st.sidebar.number_input("Width", 1, width - m_x, 200)
    m_h = st.sidebar.number_input("Height", 1, height - m_y, 200)
    
    use_manual = st.sidebar.checkbox("Use Manual Coordinates", value=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Scaling for the Display Canvas
        max_display_width = 800
        scale_ratio = max_display_width / width
        d_width = int(width * scale_ratio)
        d_height = int(height * scale_ratio)
        preview_img = pil_img.resize((d_width, d_height))
        
        st.subheader("📍 Step 1: Define Your ROI")
        col_canvas, col_preview = st.columns([2, 1])

        with col_canvas:
            st.caption("Draw on the image below OR use Sidebar to enter X, Y coordinates manually.")
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=preview_img,
                update_streamlit=True,
                height=d_height,
                width=d_width,
                drawing_mode="rect",
                key="roi_canvas",
            )

        # Logic to decide which coordinates to use
        if use_manual:
            final_x, final_y, final_w, final_h = m_x, m_y, m_w, m_h
            st.sidebar.info(f"Using Manual: {final_x}, {final_y}, {final_w}, {final_h}")
        elif canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][-1]
            final_x = int(obj["left"] / scale_ratio)
            final_y = int(obj["top"] / scale_ratio)
            final_w = int(obj["width"] / scale_ratio)
            final_h = int(obj["height"] / scale_ratio)
        else:
            final_x, final_y, final_w, final_h = 0, 0, 0, 0

        # Show selected ROI preview
        if final_w > 0 and final_h > 0:
            with col_preview:
                st.markdown("### 🔍 Target Preview")
                roi_crop = frame_rgb[final_y:final_y+final_h, final_x:final_x+final_w]
                if roi_crop.size > 0:
                    st.image(roi_crop, use_container_width=True)
                    st.metric("ROI Geometry", f"{final_w}x{final_h}", f"X:{final_x} Y:{final_y}")

            # --- LIVE ANALYSIS ---
            if st.button("🚀 Start Live Processing"):
                st.divider()
                c1, c2 = st.columns([1, 1])
                video_feed = c1.empty()
                chart_feed = c2.empty()
                
                # Setup Analysis
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                template_gray = cv2.cvtColor(frame[final_y:final_y+final_h, final_x:final_x+final_w], cv2.COLOR_BGR2GRAY)
                occlusion_history = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Core Occlusion Math
                    current_roi = frame[final_y:final_y+final_h, final_x:final_x+final_w]
                    if current_roi.size > 0:
                        gray_roi = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
                        res = cv2.matchTemplate(gray_roi, template_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        occ_pct = round(max(0, (1 - max_val) * 100), 2)
                    else:
                        occ_pct = 100.0

                    occlusion_history.append(occ_pct)
                    
                    # Visual Feedback
                    box_color = (0, 0, 255) if occ_pct > 35 else (0, 255, 0)
                    cv2.rectangle(frame, (final_x, final_y), (final_x+final_w, final_y+final_h), box_color, 4)
                    cv2.putText(frame, f"OCCLUSION: {occ_pct}%", (final_x, final_y-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3)

                    # Update Streamlit UI
                    video_feed.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    chart_feed.line_chart(occlusion_history)

                cap.release()
                st.success("Analysis Finished.")

    cap.release()














































