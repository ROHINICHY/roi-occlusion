import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile
import pandas as pd

st.set_page_config(page_title="DRDO Occlusion Lab", layout="wide")

st.title("🎯 ROI Occlusion Detection & Live Graphing")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- SIDEBAR: CONTROLS ---
    st.sidebar.header("🕹️ ROI Configuration")
    
    sensitivity = st.sidebar.slider("Occlusion Threshold (%)", 0, 100, 35)
    use_manual = st.sidebar.toggle("Enable Manual X, Y, W, H", value=False)
    
    # Initialize variables to avoid "undefined" errors
    final_x, final_y, final_w, final_h = 0, 0, 0, 0

    if use_manual:
        st.sidebar.subheader("🔢 Manual Inputs")
        # Ensure manual inputs stay within video resolution [0, W] and [0, H]
        mx = st.sidebar.number_input("X (Left Start)", 0, W-1, 100)
        my = st.sidebar.number_input("Y (Top Start)", 0, H-1, 100)
        mw = st.sidebar.number_input("Width", 1, W - mx, 150)
        mh = st.sidebar.number_input("Height", 1, H - my, 150)
        final_x, final_y, final_w, final_h = int(mx), int(my), int(mw), int(mh)
    
    ref_frame_idx = st.sidebar.slider("Select Reference Frame", 0, total_frames - 1, 0)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, ref_frame = cap.read()
    
    if ret:
        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        
        st.subheader("📍 Step 1: Define Target ROI")
        col_c, col_p = st.columns([2, 1])
        
        with col_c:
            if not use_manual:
                # CANVAS MODE
                canvas_width = 700
                scale = W / canvas_width
                canvas_height = int(H / scale)
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="red",
                    background_image=Image.fromarray(ref_rgb).resize((canvas_width, canvas_height)),
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",
                    key="canvas",
                )
                
                if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
                    obj = canvas_result.json_data["objects"][-1]
                    final_x = int(max(0, obj["left"] * scale))
                    final_y = int(max(0, obj["top"] * scale))
                    final_w = int(min(W - final_x, obj["width"] * scale))
                    final_h = int(min(H - final_y, obj["height"] * scale))
            else:
                # MANUAL PREVIEW (Safe slicing)
                preview_ref = ref_rgb.copy()
                # Draw the box on the full image so the user sees the position
                cv2.rectangle(preview_ref, (final_x, final_y), (final_x + final_w, final_y + final_h), (255, 0, 0), 10)
                st.image(preview_ref, caption="ROI Manual Position", use_container_width=True)

        with col_p:
            st.markdown("### 🔍 Target Template")
            # CRITICAL: Ensure slice is not empty before st.image
            if final_w > 5 and final_h > 5:
                template_img = ref_rgb[final_y : final_y + final_h, final_x : final_x + final_w]
                if template_img.size > 0:
                    st.image(template_img, caption=f"Tracking Area ({final_w}x{final_h})")
                else:
                    st.error("Invalid ROI Size")
            else:
                st.warning("ROI too small or not selected.")

        # --- STEP 2: LIVE ANALYSIS ---
        if st.button("🚀 Start Live Analysis") and final_w > 5:
            st.divider()
            vid_col, graph_col = st.columns([1, 1])
            vid_placeholder = vid_col.empty()
            graph_placeholder = graph_col.empty()
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Create a grayscale template for comparison
            roi_ref_gray = cv2.cvtColor(ref_frame[final_y:final_y+final_h, final_x:final_x+final_w], cv2.COLOR_BGR2GRAY)
            
            occ_data = []
            
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                
                # Extract current ROI from the frame
                curr_roi = frame[final_y:final_y+final_h, final_x:final_x+final_w]
                
                if curr_roi.size > 0:
                    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
                    # Compute similarity using Template Matching
                    res = cv2.matchTemplate(curr_gray, roi_ref_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    # Convert match score (0 to 1) to occlusion % (0 to 100)
                    score = max(0, (1 - max_val) * 100)
                else:
                    score = 100.0
                
                occ_data.append(score)
                
                # Annotate Frame
                box_color = (0, 0, 255) if score > sensitivity else (0, 255, 0)
                cv2.rectangle(frame, (final_x, final_y), (final_x + final_w, final_y + final_h), box_color, 4)
                cv2.putText(frame, f"OCC: {score:.1f}%", (final_x, final_y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

                # Update UI every 2 frames
                if i % 2 == 0:
                    vid_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    graph_placeholder.line_chart(occ_data)

            st.success("✅ Analysis Completed.")
            cap.release()

    cap.release()
















































