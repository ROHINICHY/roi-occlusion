import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile

st.set_page_config(page_title="DRDO Occlusion Monitor", layout="wide")

st.title("🎯 DRDO ROI Occlusion Detector")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- STEP 1: SELECT FRAME ---
    st.sidebar.header("1. Video Navigation")
    frame_no = st.sidebar.slider("Select Frame for ROI", 0, total_frames - 1, 0)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, ref_frame = cap.read()

    if ret:
        # Convert BGR to RGB
        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(ref_rgb)
        
        # Resize image for display to ensure it fits the UI nicely
        # We calculate the aspect ratio to maintain proportions
        max_width = 800
        aspect_ratio = pil_img.height / pil_img.width
        display_width = max_width
        display_height = int(max_width * aspect_ratio)
        
        preview_img = pil_img.resize((display_width, display_height))

        st.subheader("Step 2: Draw ROI around Target Object")
        st.info("💡 Use the rectangle tool below to select the object you want to monitor.")

        # --- STEP 2: DRAWING CANVAS (Background is now fixed) ---
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=3,
            stroke_color="#ff0000",
            background_image=preview_img,  # Fixed image passed here
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key="roi_selector",  # Unique key
        )

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            # Map canvas coordinates back to original video resolution
            scale_x = pil_img.width / display_width
            scale_y = pil_img.height / display_height
            
            obj = canvas_result.json_data["objects"][-1]
            x = int(obj["left"] * scale_x)
            y = int(obj["top"] * scale_y)
            w = int(obj["width"] * scale_x)
            h = int(obj["height"] * scale_y)
            
            # Show a crop of what the user selected
            template_crop = ref_rgb[y:y+h, x:x+w]
            if template_crop.size > 0:
                st.sidebar.image(template_crop, caption="Selected Target")

            # --- STEP 3: RUN ANALYSIS ---
            if st.button("🚀 Start Occlusion Analysis"):
                # Processing Logic
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # UI Layout
                col1, col2 = st.columns([2, 1])
                video_feed = col1.empty()
                graph_feed = col2.empty()
                
                occlusion_data = []
                template_gray = cv2.cvtColor(cv2.cvtColor(template_crop, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    # ROI Tracking Logic
                    roi_area = frame[y:y+h, x:x+w]
                    if roi_area.shape[0] == h and roi_area.shape[1] == w:
                        roi_gray = cv2.cvtColor(roi_area, cv2.COLOR_BGR2GRAY)
                        res = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        occ_pct = max(0, (1 - max_val) * 100)
                    else:
                        occ_pct = 100

                    occlusion_data.append(occ_pct)
                    
                    # Drawing on frame
                    color = (0, 0, 255) if occ_pct > 40 else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Update App
                    video_feed.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    graph_feed.line_chart(occlusion_data)

                st.success("Analysis Complete!")
    
    cap.release()














































