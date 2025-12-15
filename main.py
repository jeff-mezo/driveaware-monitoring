import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
from datetime import datetime

# --- IVP CONCEPTS: ENHANCEMENT & RESTORATION [cite: 14, 15] ---
def adjust_gamma(image, gamma=1.0):
    """
    Applies Gamma Correction to handle different lighting conditions.
    Satisfies: Image Enhancement technique.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(gray_image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Satisfies: Advanced Image Enhancement (better than standard eq).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def eye_aspect_ratio(eye):
    """
    Calculates the Eye Aspect Ratio (EAR).
    Satisfies: Feature Extraction & Classification geometry.
    """
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """
    Calculates Mouth Aspect Ratio (MAR) to detect yawning.
    Satisfies: Multi-factor Classification.
    """
    # Vertical points
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57
    # Horizontal distance
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# --- MAIN APP UI [cite: 26, 46] ---
st.set_page_config(page_title="DriveAware: IVP Final Project", layout="wide")

# Initialize session state early
if 'run' not in st.session_state:
    st.session_state['run'] = False

# Custom CSS for better styling
st.markdown("""
<style>
    /* Global surface styling */
    body, .block-container {
        background: radial-gradient(circle at 20% 20%, #f6f9ff, #eef2fb 45%, #e7ecf7 65%, #e3e8f5 100%);
    }
    .block-container { padding-top: 1.5rem; }

    /* Typography */
    .main-title {
        text-align: center;
        color: #1f4e79;
        font-size: 2.6em;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .subtitle {
        text-align: center;
        color: #4f5b6b;
        font-size: 1.05em;
        margin-bottom: 26px;
    }

    /* Panel styling */
    .panel {
        background: #ffffffdd;
        border-radius: 14px;
        padding: 18px 18px 10px 18px;
        box-shadow: 0 16px 40px rgba(31, 78, 121, 0.12);
        border: 1px solid #e3e8f5;
    }
    .panel h3 { margin-top: 0; }

    /* Metric cards */
    .metric-card, .alert-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e4e8f2;
        box-shadow: 0 12px 32px rgba(0,0,0,0.04);
    }
    .metric-card { background: linear-gradient(135deg, #f7fbff, #eef3ff); }
    .alert-card { background: linear-gradient(135deg, #ffecec, #ffdfe0); }

    .status-active {
        color: #0f9d58;
        font-weight: 700;
        font-size: 1.15em;
    }
    .status-alert {
        color: #d93025;
        font-weight: 800;
        font-size: 1.15em;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.75; }
    }

    /* Badges & pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: 999px;
        font-weight: 700;
        color: #0f1d40;
        background: #ffffff;
        border: 1px solid #d8e2f1;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }

    /* Live feed frames */
    .stImage img {
        border-radius: 12px;
        border: 2px solid #d5def0;
        box-shadow: 0 12px 32px rgba(31, 78, 121, 0.18);
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0f1d40, #1f3a93);
        color: #ffffff;
    }
    /* Force all sidebar text to white */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stCheckbox label {
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
        border: 1px solid #4ea1ff;
        color: #ffffff;
        font-weight: 700;
        background: linear-gradient(135deg, #7cc6ff, #4ea1ff);
        box-shadow: 0 8px 18px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="main-title">🚗 DriveAware</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Driver Drowsiness & Yawn Detection System | CMSC 162 Final Project</div>', unsafe_allow_html=True)

st.divider()

# Quick tips banner
st.markdown(
    """
    <div class="panel">
        <b>How to use:</b> Start the camera, tune thresholds in the sidebar, and toggle segmentation/debug views as needed. Night Mode helps in low light. Alerts trigger when EAR or MAR cross your set limits.
    </div>
    """,
    unsafe_allow_html=True,
)

# Control Panel for IVP Variables
st.sidebar.header("⚙️ Processing Parameters")
EAR_THRESHOLD = st.sidebar.slider("Eye Threshold (EAR)", 0.15, 0.4, 0.25, 0.01)
MAR_THRESHOLD = st.sidebar.slider("Yawn Threshold (MAR)", 0.3, 0.9, 0.6, 0.05)
CONSEC_FRAMES = st.sidebar.slider("Consecutive Frames for Alert", 10, 60, 20)

st.sidebar.divider()

st.sidebar.header("🎨 Image Enhancement Options")
use_night_mode = st.sidebar.checkbox("Enable Night Mode (CLAHE)", value=False)
gamma_value = st.sidebar.slider("Gamma Correction", 0.5, 3.0, 1.0, 0.1)
show_debug = st.sidebar.checkbox("Show Debug View (Thresholding)", value=True)
show_segmentation = st.sidebar.checkbox("Show Face Segmentation (Eyes & Mouth)", value=True)

st.sidebar.divider()

st.sidebar.header("Camera Control")
col_start, col_stop = st.sidebar.columns(2)
with col_start:
    start_button = st.button("▶️ Start Camera", use_container_width=True)
with col_stop:
    stop_button = st.button("⏹️ Stop Camera", use_container_width=True)

if start_button:
    st.session_state['run'] = True
if stop_button:
    st.session_state['run'] = False

# Setup Dlib (The Core Logic)
# Note: Ensure shape_predictor_68_face_landmarks.dat is in the same folder
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError:
    st.error("Missing 'shape_predictor_68_face_landmarks.dat'. Please download it.")
    st.stop()

# Grab indexes for eyes and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Layout
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📹 Live Feed & Classification")
    frame_window = st.image([])
with col2:
    st.subheader("🔬 Analysis View")
    debug_window = st.image([])
    stats_placeholder = st.empty()

# Status badge placeholder (updates every frame)
status_badge = st.empty()

# Counterssss
COUNTER = 0
ALARM_ON = False
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    st.error("Failed to open camera. Make sure a camera is connected and available.")
    st.stop()

# --- PROCESSING LOOP ---
while st.session_state['run']:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video.")
        break

    # 1. Resize for speed (Standard IVP optimization)
    frame = cv2.resize(frame, (640, 480))
    
    # 2. Pre-processing 
    # Gamma Correction
    frame = adjust_gamma(frame, gamma_value)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Enhancement (Night Mode)
    if use_night_mode:
        gray = apply_clahe(gray)
        # Apply slight median blur to remove noise enhanced by CLAHE
        gray = cv2.medianBlur(gray, 3) 

    # 4. Detection
    rects = detector(gray, 0)
    
    # Default status
    status_text = "Status: Active"
    status_color = (0, 255, 0)

    for rect in rects:
        # 5. Segmentation (Landmarks)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # 6. Visualization (Convex Hull)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        
        if show_segmentation:
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)

        # 7. Classification Logic 
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= CONSEC_FRAMES:
                status_text = "DROWSINESS ALERT!"
                status_color = (0, 0, 255)
                # Visual Alarm
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            
        if mar > MAR_THRESHOLD:
            cv2.putText(frame, "YAWNING DETECTED", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display Stats on Frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Live status badge above the feeds
    badge_border = "#d93025" if "ALERT" in status_text else "#0f9d58"
    badge_icon = "🚨" if "ALERT" in status_text else "✅"
    status_badge.markdown(
        f"""
        <div class="status-pill" style="border-color:{badge_border}; box-shadow: 0 10px 26px rgba(0,0,0,0.08);">
            <span>{badge_icon}</span>
            <span>{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 8. Create Debug View (Segmentation Visualization)
    # Shows the binary thresholded version of the image to prove "Segmentation"
    if show_debug:
        # Simple thresholding to show what the computer 'sees' as features
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        debug_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        debug_window.image(debug_frame, channels="BGR", caption="Binary Threshold (Debug)")
    else:
        # Show just the grayscale processed input
        debug_window.image(gray, clamp=True, channels="GRAY", caption="Processed Input (Gray/CLAHE)")

    # Update Main Feed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Draw a border based on status
    if "ALERT" in status_text:
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 0, 0])
    
    frame_window.image(frame)
    
    # Update stats with better formatting
    if "ALERT" in status_text:
        stats_placeholder.markdown(f"""
        <div class="alert-card">
            <div class="status-alert">🚨 {status_text}</div>
            <br>
            <b>👁️ EAR:</b> {ear:.2f} (Threshold: {EAR_THRESHOLD})<br>
            <b>😑 MAR:</b> {mar:.2f} (Threshold: {MAR_THRESHOLD})<br>
            <b>⏱️ Frames:</b> {COUNTER} / {CONSEC_FRAMES}
        </div>
        """, unsafe_allow_html=True)
    else:
        stats_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="status-active">✅ {status_text}</div>
            <br>
            <b>👁️ EAR:</b> {ear:.2f} (Threshold: {EAR_THRESHOLD})<br>
            <b>😑 MAR:</b> {mar:.2f} (Threshold: {MAR_THRESHOLD})<br>
            <b>⏱️ Frames:</b> {COUNTER} / {CONSEC_FRAMES}
        </div>
        """, unsafe_allow_html=True)

cap.release()