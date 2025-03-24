import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import datetime

# Page configuration
st.set_page_config(
    page_title="AI Attendance Tracker",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #1A1F2B;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2C3A4F;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #b4c2dc;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #56647b;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FF4D4D;
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(255, 77, 77, 0.5);
    }
    
    /* Card-like containers */
    .card {
        background-color: #292e3b;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(180, 194, 220, 0.1);
    }
    
    /* Stats display */
    .stat-box {
        background-color: #414654;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FF4D4D;
    }
    
    .stat-label {
        color: #e0e0e0;
        font-size: 0.9rem;
    }
    
    /* Image container */
    .img-container {
        border: 2px solid #56647b;
        border-radius: 10px;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .img-container:hover {
        border-color: #FF4D4D;
        box-shadow: 0 0 20px rgba(255, 77, 77, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load YOLOv8 model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model_path = "best.pt"  # Change this if needed
model = load_model(model_path)

# App header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/fluency/96/000000/school-building.png", width=80)
with col2:
    st.title("Classroom Attendance Tracker 🏫")
    st.markdown("<p style='color: #b4c2dc; margin-top: -10px;'>Powered by YOLO Face Recognition</p>", unsafe_allow_html=True)

# Function to detect people in an image
def detect_people(frame):
    results = model(frame)  # Run YOLO detection
    detected_objects = results[0].boxes
    num_students = 0 

    for box in detected_objects:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())

        if cls == 0:  # Class 0 is 'person' in COCO dataset
            num_students += 1
            label = f'Person {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, num_students

# Sidebar for navigation
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
    
    # Current date and time
    current_time = datetime.datetime.now().strftime("%B %d, %Y | %H:%M")
    st.markdown(f"<p style='text-align: center; color: #e0e0e0;'>{current_time}</p>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    option = st.radio("Choose an option", ["Upload Image", "Live Camera"])
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # System status
    st.markdown("<h3 style='text-align: center;'>System Status</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='stat-box'>
        <div class='stat-value'>✓</div>
        <div class='stat-label'>Model Loaded</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Credits
    st.markdown("<p style='text-align: center; color: #56647b; font-size: 0.8rem;'>Developed for AI Workshop</p>", unsafe_allow_html=True)

# Main content area
if option == "Upload Image":
    st.markdown("<h2>Upload an Image for Attendance Detection</h2>", unsafe_allow_html=True)
    
    # Instructions card
    st.markdown("""
    <div class='card'>
        <h3>📋 Instructions</h3>
        <p>Upload a classroom image to detect and count students automatically.</p>
        <p>Supported formats: JPG, PNG, JPEG</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display progress
        with st.spinner("Processing image..."):
            # Add a small delay to simulate processing
            time.sleep(1)
            
            # Load and display original image
            image = Image.open(uploaded_file)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>📷 Original Image</h3>", unsafe_allow_html=True)
            st.markdown("<div class='img-container'>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Process image
            image_np = np.array(image)
            processed_image, count = detect_people(image_np)
            
            # Results section
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>🔍 Detection Results</h3>", unsafe_allow_html=True)
            
            # Display processed image
            st.markdown("<div class='img-container'>", unsafe_allow_html=True)
            st.image(processed_image, caption="Processed Image with Detections", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display count with animation
            st.markdown(f"""
            <div style='text-align: center; margin-top: 20px;'>
                <div class='stat-box' style='display: inline-block; min-width: 200px;'>
                    <div class='stat-value'>{count}</div>
                    <div class='stat-label'>Students Detected</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Timestamp
            detection_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"<p style='text-align: right; color: #56647b; font-size: 0.8rem;'>Detection completed at {detection_time}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Success message
            st.success(f"Successfully detected {count} students in the image!")

elif option == "Live Camera":
    st.markdown("<h2>Live Attendance Tracking 🎥</h2>", unsafe_allow_html=True)
    
    # Instructions card
    st.markdown("""
    <div class='card'>
        <h3>📋 Instructions</h3>
        <p>Use your webcam to track attendance in real-time.</p>
        <p>Click "Start Camera" to begin and "Stop Camera" when finished.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera controls
    col1, col2 = st.columns(2)
    with col1:
        start = st.button("▶️ Start Camera")
    with col2:
        stop = st.button("⏹️ Stop Camera")
    
    # Camera view container
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📹 Camera Feed</h3>", unsafe_allow_html=True)
    frame_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Stats container
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>📊 Live Statistics</h3>", unsafe_allow_html=True)
    attendance_text = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    if start:
        cap = cv2.VideoCapture(0)  # Open webcam
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame.")
                break
            
            # Process frame
            processed_frame, num_students = detect_people(frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
            
            # Update UI
            frame_placeholder.image(processed_frame, caption="Live Attendance Tracking", use_container_width=True)
            
            # Update stats with styled HTML
            attendance_text.markdown(f"""
            <div style='text-align: center;'>
                <div class='stat-box' style='display: inline-block; min-width: 200px;'>
                    <div class='stat-value'>{num_students}</div>
                    <div class='stat-label'>Current Attendance</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Stop the camera when button is pressed
            if stop:
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Footer
st.markdown("""
<div style='position: fixed; bottom: 0; left: 0; width: 100%; background-color: #2C3A4F; padding: 10px; text-align: center;'>
    <p style='margin: 0; color: #e0e0e0; font-size: 0.8rem;'>
        AI Attendance Tracking System | Face Recognition with YOLO | © 2023
    </p>
</div>
""", unsafe_allow_html=True)



