import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="DRDO ROI Occlusion System",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for DRDO theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --drdo-blue: #003366;
        --drdo-red: #cc0000;
        --drdo-gold: #ffcc00;
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--drdo-blue), #004488);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        border-left: 8px solid var(--drdo-red);
    }
    
    .drdo-title {
        color: var(--drdo-gold);
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .drdo-subtitle {
        color: white;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .section-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid var(--drdo-blue);
        margin: 1rem 0;
        font-weight: bold;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid var(--drdo-blue);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--drdo-blue), #004488);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #004488, var(--drdo-blue));
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .success-box {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom table styling */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background-color: var(--drdo-blue);
        color: white;
        padding: 12px;
        text-align: left;
    }
    
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    
    .dataframe tr:hover {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <div class="drdo-title">Defence Research and Development Organisation</div>
    <div class="drdo-subtitle">Jodhpur - ROI Occlusion Analysis System</div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'roi_coordinates' not in st.session_state:
    st.session_state.roi_coordinates = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'occlusion_data' not in st.session_state:
    st.session_state.occlusion_data = None

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header">📹 Upload Input Video</div>', unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Upload Video", 
        type=["mp4", "avi", "mov", "MP4", "AVI", "MOV"],
        label_visibility="collapsed",
        help="Drag and drop file here. Limit 200MB per file - MP4, AVI, MPEG4"
    )

if uploaded_video is None:
    with col1:
        st.markdown('<div class="warning-box">⚠️ Please upload a video to continue</div>', unsafe_allow_html=True)
    st.stop()

# Save uploaded video
with col1:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    st.markdown(f"""
    <div class="success-box">
        ✅ Video uploaded successfully: {uploaded_video.name} ({uploaded_video.size//1024}KB)
    </div>
    """, unsafe_allow_html=True)

# Get video properties
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps

with col2:
    st.markdown('<div class="section-header">📊 Video Information</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 1.5rem; color: var(--drdo-blue); font-weight: bold;">Video Loaded Successfully</div>
        <hr style="margin: 10px 0;">
        <div><strong>Total Frames:</strong> {total_frames}</div>
        <div><strong>FPS:</strong> {fps}</div>
        <div><strong>Resolution:</strong> {width}×{height}</div>
        <div><strong>Duration:</strong> {duration:.2f}s</div>
        <div><strong>File Size:</strong> {uploaded_video.size//1024}KB</div>
    </div>
    """, unsafe_allow_html=True)

# ROI Selection Section
st.markdown('<div class="section-header">🎯 Select Frame for ROI Selection</div>', unsafe_allow_html=True)

frame_col1, frame_col2 = st.columns([3, 1])

with frame_col1:
    frame_no = st.slider("", 0, total_frames - 1, 48, label_visibility="collapsed")

# Get the selected frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ret, frame = cap.read()

if not ret:
    st.error("Could not read frame from video.")
    st.stop()

# Convert frame to RGB for display
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_pil = Image.fromarray(frame_rgb)

# Display frame
st.markdown(f'<div style="text-align: center; font-weight: bold; margin: 10px 0;">Selected Frame Preview (Frame No: {frame_no})</div>', unsafe_allow_html=True)
st.image(frame_pil, width=900)

# ROI Selection Canvas
st.markdown('<div class="section-header">🖱️ Draw Bounding Box on Object</div>', unsafe_allow_html=True)

# Resize image for canvas
CANVAS_W = 900
scale = CANVAS_W / frame_pil.size[0]
CANVAS_H = int(frame_pil.size[1] * scale)
frame_pil_resized = frame_pil.resize((CANVAS_W, CANVAS_H))

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=3,
    stroke_color="red",
    background_image=frame_pil_resized,
    update_streamlit=True,
    height=CANVAS_H,
    width=CANVAS_W,
    drawing_mode="rect",
    key="canvas_roi",
)

# Manual ROI Coordinates Input
st.markdown('<div class="section-header">📐 Enter ROI Coordinates Manually (x, y, width, height)</div>', unsafe_allow_html=True)

coord_col1, coord_col2, coord_col3, coord_col4 = st.columns(4)

with coord_col1:
    manual_x = st.number_input("x (left)", min_value=0, max_value=width, value=66)
with coord_col2:
    manual_y = st.number_input("y (top)", min_value=0, max_value=height, value=42)
with coord_col3:
    manual_w = st.number_input("width (w)", min_value=1, max_value=width, value=200)
with coord_col4:
    manual_h = st.number_input("height (h)", min_value=1, max_value=height, value=150)

# Use either canvas or manual coordinates
if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
    # Get rectangle from canvas
    obj = canvas_result.json_data["objects"][-1]
    x = int(obj["left"])
    y = int(obj["top"])
    w = int(obj["width"])
    h = int(obj["height"])
    
    # Convert back to original coordinates
    orig_x = int(x / scale)
    orig_y = int(y / scale)
    orig_w = int(w / scale)
    orig_h = int(h / scale)
else:
    # Use manual coordinates
    orig_x = manual_x
    orig_y = manual_y
    orig_w = manual_w
    orig_h = manual_h

st.session_state.roi_coordinates = (orig_x, orig_y, orig_w, orig_h)

# ROI Preview
roi_preview = frame_rgb[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]

if roi_preview.size > 0:
    st.markdown('<div class="section-header">🔍 ROI Preview (Selected Object)</div>', unsafe_allow_html=True)
    st.image(Image.fromarray(roi_preview), width=400)
    
    st.markdown(f"""
    <div class="success-box">
        ✅ ROI Selected Successfully<br>
        Position: ({orig_x}, {orig_y}) | Size: {orig_w}×{orig_h} | Area: {orig_w*orig_h} pixels
    </div>
    """, unsafe_allow_html=True)

# Analysis Button
st.markdown('<div class="section-header">🚀 Run Occlusion Analysis</div>', unsafe_allow_html=True)

if st.button("▶ Run Analysis", type="primary", use_container_width=True):
    with st.spinner("Analyzing video frames..."):
        cap = cv2.VideoCapture(video_path)
        
        # Get template from first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        
        if ret:
            first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            roi_template = first_gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
            
            if roi_template.size == 0:
                st.error("ROI template extraction failed.")
                st.stop()
            
            frames_list = []
            occlusion_list = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze all frames
            for f in range(total_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Calculate occlusion
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_now = gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
                
                if roi_now.size > 0:
                    roi_now_resized = cv2.resize(roi_now, (roi_template.shape[1], roi_template.shape[0]))
                    diff = cv2.absdiff(roi_template, roi_now_resized)
                    occlusion_percent = (np.sum(diff > 30) / diff.size) * 100
                else:
                    occlusion_percent = 100
                
                frames_list.append(f)
                occlusion_list.append(round(float(occlusion_percent), 2))
                
                # Update progress
                progress = (f + 1) / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {f+1}/{total_frames}...")
            
            cap.release()
            
            # Create dataframe
            df = pd.DataFrame({
                "Frame": frames_list,
                "Occlusion (%)": occlusion_list
            })
            
            st.session_state.occlusion_data = df
            st.session_state.analysis_done = True
            
            st.markdown('<div class="success-box">✅ Occlusion Analysis Completed</div>', unsafe_allow_html=True)

# Display Results
if st.session_state.analysis_done and st.session_state.occlusion_data is not None:
    df = st.session_state.occlusion_data
    
    # Create two columns for graph and stats
    results_col1, results_col2 = st.columns([2, 1])
    
    with results_col1:
        st.markdown('<div class="section-header">📈 Occlusion Graph (Frame vs Occlusion %)</div>', unsafe_allow_html=True)
        
        # Create matplotlib figure for better control
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["Frame"], df["Occlusion (%)"], color='red', linewidth=2)
        ax.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Occlusion Percentage', fontsize=12, fontweight='bold')
        ax.set_title('Object Occlusion Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, total_frames)
        ax.set_ylim(0, 100)
        
        # Highlight high occlusion areas
        high_occlusion = df[df["Occlusion (%)"] > 50]
        if not high_occlusion.empty:
            ax.fill_between(high_occlusion["Frame"], high_occlusion["Occlusion (%)"], 
                           color='orange', alpha=0.3, label='High Occlusion (>50%)')
        
        # Add legend
        ax.legend()
        
        st.pyplot(fig)
    
    with results_col2:
        st.markdown('<div class="section-header">📊 Statistics</div>', unsafe_allow_html=True)
        
        avg_occlusion = df["Occlusion (%)"].mean()
        max_occlusion = df["Occlusion (%)"].max()
        min_occlusion = df["Occlusion (%)"].min()
        std_occlusion = df["Occlusion (%)"].std()
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="text-align: center; margin-bottom: 15px;">
                <div style="font-size: 0.9rem; color: #666;">Average Occlusion</div>
                <div style="font-size: 2rem; color: var(--drdo-blue); font-weight: bold;">{avg_occlusion:.1f}%</div>
            </div>
            <hr>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div style="font-size: 0.8rem; color: #666;">Max</div>
                    <div style="font-size: 1.2rem; font-weight: bold;">{max_occlusion:.1f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #666;">Min</div>
                    <div style="font-size: 1.2rem; font-weight: bold;">{min_occlusion:.1f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #666;">Std Dev</div>
                    <div style="font-size: 1.2rem; font-weight: bold;">{std_occlusion:.1f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate frames with significant occlusion
        significant_occlusion = len(df[df["Occlusion (%)"] > 30])
        percent_significant = (significant_occlusion / len(df)) * 100
        
        st.metric(
            label="Frames with >30% Occlusion",
            value=f"{significant_occlusion}",
            delta=f"{percent_significant:.1f}% of total"
        )
    
    # Data Table
    st.markdown('<div class="section-header">📋 Occlusion Data Table</div>', unsafe_allow_html=True)
    
    # Show first 10 rows by default
    show_all = st.checkbox("Show all data", value=False)
    
    if show_all:
        st.dataframe(df, use_container_width=True)
    else:
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 of {len(df)} rows. Check 'Show all data' to see complete dataset.")
    
    # Download Button
    csv = df.to_csv(index=False).encode("utf-8")
    
    download_col1, download_col2, download_col3 = st.columns([2, 1, 2])
    with download_col2:
        st.download_button(
            "⬇ Download Occlusion Data (CSV)",
            csv,
            "occlusion_data.csv",
            "text/csv",
            use_container_width=True,
            type="primary"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <strong>DRDO Jodhpur</strong> | ROI Occlusion Analysis System v1.0 | © 2024 Defence Research and Development Organisation
</div>
""", unsafe_allow_html=True)

# Cleanup
try:
    tfile.close()
except:
    pass









































