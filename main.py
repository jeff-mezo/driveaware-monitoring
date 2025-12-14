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

st.sidebar.title("DriveAware Control Panel")
st.sidebar.write("CMSC 162 Final Project")

# Control Panel for IVP Variables
st.sidebar.header("Processing Parameters")
EAR_THRESHOLD = st.sidebar.slider("Eye Threshold (EAR)", 0.15, 0.4, 0.25, 0.01)
MAR_THRESHOLD = st.sidebar.slider("Yawn Threshold (MAR)", 0.3, 0.9, 0.6, 0.05)
CONSEC_FRAMES = st.sidebar.slider("Consecutive Frames for Alert", 10, 60, 20)

st.sidebar.header("Image Enhancement Options")
use_night_mode = st.sidebar.checkbox("Enable Night Mode (CLAHE)", value=False)
gamma_value = st.sidebar.slider("Gamma Correction", 0.5, 3.0, 1.0, 0.1)
show_debug = st.sidebar.checkbox("Show Debug View (Thresholding)", value=True)

# Initialization
if 'run' not in st.session_state:
    st.session_state['run'] = False

start_button = st.sidebar.button("Start Camera")
stop_button = st.sidebar.button("Stop Camera")

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
    st.subheader("Live Feed & Classification")
    frame_window = st.image([])
with col2:
    st.subheader("IVP Debug View")
    debug_window = st.image([])
    stats_placeholder = st.empty()

# Counterssss
COUNTER = 0
ALARM_ON = False
cap = cv2.VideoCapture(1)

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
    
    # Update stats
    stats_placeholder.markdown(f"""
    **Current Metrics:**
    - **Status:** :{ "red" if "ALERT" in status_text else "green" }[{status_text}]
    - **Frame Counter:** {COUNTER} / {CONSEC_FRAMES}
    """)

cap.release()