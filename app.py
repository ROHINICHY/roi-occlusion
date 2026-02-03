import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile
import pandas as pd

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

    # --- SIDEBAR: CONTROLS ---
    st.sidebar.header("🕹️ ROI Configuration")
    
    # Sensitivity Slider
    sensitivity = st.sidebar.slider("Occlusion Threshold (%)", 0, 100, 35, 
                                   help="At what percentage should the box turn RED?")
    
    # Manual Coordinate Toggle
    use_manual = st.sidebar.toggle("Enable Manual X, Y, W, H", value=False)
    
    if use_manual:
        st.sidebar.subheader("🔢 Manual Inputs")
        mx = st.sidebar.number_input("X (Left)", 0, W, 100)
        my = st.sidebar.number_input("Y (Top)", 0, H, 100)
        mw = st.sidebar.number_input("Width", 10, W - mx, 150)
        mh = st.sidebar.number_input("Height", 10, H - my, 150)
        final_x, final_y, final_w, final_h = mx, my, mw, mh
    
    ref_frame_idx = st.sidebar.slider("Select Reference Frame", 0, total_frames - 1, 0)
    
    # Load Reference Frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, ref_frame = cap.read()
    
    if ret:
        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        
        st.subheader("📍 Step 1: Define Target ROI")
        col_c, col_p = st.columns([2, 1])
        
        with col_c:
            if not use_manual:
                st.info("Drawing Mode: Draw a rectangle on the frame below.")
                # Canvas Logic
                canvas_width = 800
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
                    final_x = int(obj["left"] * scale)
                    final_y = int(obj["top"] * scale)
                    final_w = int(obj["width"] * scale)
                    final_h = int(obj["height"] * scale)
                else:
                    final_x, final_y, final_w, final_h = 0, 0, 0, 0
            else:
                st.warning("Manual Mode: Coordinates set in sidebar.")
                # Preview manual box
                preview_ref = ref_rgb.copy()
                cv2.rectangle(preview_ref, (final_x, final_y), (final_x+final_w, final_y+final_h), (255, 0, 0), 5)
                st.image(preview_ref, caption="Manual Selection Preview", use_container_width=True)

        with col_p:
            st.markdown("### 🔍 Target Template")
            if final_w > 0 and final_h > 0:
                template_img = ref_rgb[final_y:final_y+final_h, final_x:final_x+final_w]
                if template_img.size > 0:
                    st.image(template_img, caption=f"Tracking: {final_w}x{final_h}")
                    st.write(f"**Coordinates:** X:{final_x}, Y:{final_y}")
            else:
                st.write("Please select an ROI to begin.")

        # --- STEP 2: LIVE ANALYSIS ---
        if st.button("🚀 Start Live Analysis") and final_w > 0:
            st.divider()
            
            vid_col, graph_col = st.columns([1, 1])
            vid_placeholder = vid_col.empty()
            graph_placeholder = graph_col.empty()
            
            # Setup Analysis
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            template_gray = cv2.cvtColor(ref_frame[final_y:final_y+final_h, final_x:final_x+final_w], cv2.COLOR_BGR2GRAY)
            
            occ_data = []
            
            # Use a container for the chart to keep it smooth
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                
                # ROI Logic
                curr_roi = frame[final_y:final_y+final_h, final_x:final_x+final_w]
                
                if curr_roi.size > 0:
                    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(curr_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    score = max(0, (1 - max_val) * 100)
                else:
                    score = 100.0
                
                occ_data.append(score)
                
                # Visual Markers
                box_color = (0, 0, 255) if score > sensitivity else (0, 255, 0)
                cv2.rectangle(frame, (final_x, final_y), (final_x+final_w, final_y+final_h), box_color, 4)
                cv2.putText(frame, f"Occlusion: {score:.1f}%", (final_x, final_y-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3)

                # Live Stream (Update UI)
                if i % 2 == 0:
                    vid_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    graph_placeholder.line_chart(occ_data)

            st.success("✅ Analysis Completed.")
            
            # Option to download data
            df = pd.DataFrame(occ_data, columns=["Occlusion_Percentage"])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Occlusion Data (CSV)", csv, "occlusion_results.csv", "text/csv")

    cap.release()
















































