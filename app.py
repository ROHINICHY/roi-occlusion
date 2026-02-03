import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile
import time

st.set_page_config(page_title="DRDO Occlusion Lab", layout="wide")

st.title("🎯 ROI Occlusion Detection & Live Graphing")

# 1. Video Upload
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- SIDEBAR: MANUAL X, Y, W, H ---
    st.sidebar.header("🕹️ ROI Manual Control")
    use_manual = st.sidebar.toggle("Enable Manual Coordinates", value=False)
    
    # Manual Sliders/Inputs
    mx = st.sidebar.number_input("X (Left)", 0, W, 100)
    my = st.sidebar.number_input("Y (Top)", 0, H, 100)
    mw = st.sidebar.number_input("Width", 10, W-mx, 150)
    mh = st.sidebar.number_input("Height", 10, H-my, 150)

    # Reference Frame Selector
    ref_frame_idx = st.sidebar.slider("Select Reference Frame", 0, total_frames-1, 0)
    
    # Load Reference Frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, ref_frame = cap.read()
    
    if ret:
        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        
        st.subheader("📍 Step 1: Define Target ROI")
        
        # Display selection methods
        col_c, col_p = st.columns([2, 1])
        
        with col_c:
            if not use_manual:
                # Canvas Selection
                st.info("Drawing Mode: Use the rectangle tool on the frame.")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="red",
                    background_image=Image.fromarray(ref_rgb).resize((700, int(700*(H/W)))),
                    update_streamlit=True,
                    height=int(700*(H/W)),
                    width=700,
                    drawing_mode="rect",
                    key="canvas",
                )
                
                # Convert Canvas scale back to Video scale
                if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
                    obj = canvas_result.json_data["objects"][-1]
                    scale = W / 700
                    final_x, final_y = int(obj["left"] * scale), int(obj["top"] * scale)
                    final_w, final_h = int(obj["width"] * scale), int(obj["height"] * scale)
                else:
                    final_x, final_y, final_w, final_h = 0, 0, 0, 0
            else:
                # Manual Selection
                st.warning("Manual Mode: Using coordinates from the sidebar.")
                final_x, final_y, final_w, final_h = mx, my, mw, mh
                # Draw preview box on ref_rgb
                preview_box = ref_rgb.copy()
                cv2.rectangle(preview_box, (mx, my), (mx+mw, my+mh), (255, 0, 0), 5)
                st.image(preview_box, caption="Manual ROI Preview", use_container_width=True)

        with col_p:
            st.markdown("### 🔍 Target Template")
            if final_w > 0 and final_h > 0:
                template_img = ref_rgb[final_y:final_y+final_h, final_x:final_x+final_w]
                if template_img.size > 0:
                    st.image(template_img, caption=f"Tracking: {final_w}x{final_h}")
                    st.write(f"**X:** {final_x}, **Y:** {final_y}")
            else:
                st.write("No ROI selected yet.")

        # --- STEP 2: LIVE ANALYSIS ---
        if st.button("🚀 Start Analysis") and final_w > 0:
            st.divider()
            
            # Setup Layout
            vid_col, graph_col = st.columns([1, 1])
            vid_placeholder = vid_col.empty()
            graph_placeholder = graph_col.empty()
            metric_placeholder = st.sidebar.empty()

            # Initialize tracking
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            template_gray = cv2.cvtColor(ref_frame[final_y:final_y+final_h, final_x:final_x+final_w], cv2.COLOR_BGR2GRAY)
            
            occlusion_history = []
            
            # Processing Loop
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                
                # Extract current ROI
                current_roi = frame[final_y:final_y+final_h, final_x:final_x+final_w]
                
                if current_roi.size > 0:
                    current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
                    # Template Matching for Similarity
                    res = cv2.matchTemplate(current_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    
                    # Score to Percentage
                    occ_score = max(0, (1 - max_val) * 100)
                else:
                    occ_score = 100.0
                
                occlusion_history.append(occ_score)
                
                # Visuals
                color = (0, 0, 255) if occ_score > 30 else (0, 255, 0)
                cv2.rectangle(frame, (final_x, final_y), (final_x+final_w, final_y+final_h), color, 4)
                cv2.putText(frame, f"OCC: {occ_score:.1f}%", (final_x, final_y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                # Live Updates (Every 2 frames for speed)
                if i % 2 == 0:
                    vid_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    graph_placeholder.line_chart(occlusion_history)
                    metric_placeholder.metric("Current Occlusion", f"{occ_score:.1f}%")

            cap.release()
            st.success("Analysis Complete!")

    cap.release()















































