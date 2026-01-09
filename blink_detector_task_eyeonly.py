import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime

# Initialize MediaPipe Face Mesh (we'll use it just for eye detection)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Important for accurate eye landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Eye landmarks indices (MediaPipe Face Mesh)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Blink detection parameters
EAR_THRESHOLD = 0.21  # Lower threshold for eye-only videos
CONSECUTIVE_FRAMES = 2  # Fewer frames needed for eye-only
MIN_BLINK_DURATION = 0.08  # Shorter minimum duration

# Initialize variables
blink_counter = 0
frame_counter = 0
is_blinking = False
blink_start_time = 0
blink_data = []

# CSV file setup
csv_filename = f"eye_blink_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Blink Number", "Start Time", "End Time", "Duration (s)"])

def eye_aspect_ratio(eye_landmarks):
    # Calculate vertical distances
    vertical1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    vertical2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    
    # Calculate horizontal distance
    horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    
    # Calculate eye aspect ratio
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Open video file
video_path = 'MVI_4759.MP4'  # Your eye-only video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# For eye-only videos, we'll assume the eye region is centered
# You may need to adjust these based on your specific video
EYE_REGION_WIDTH = frame_width // 2
EYE_REGION_HEIGHT = frame_height // 2
EYE_REGION_X = (frame_width - EYE_REGION_WIDTH) // 2
EYE_REGION_Y = (frame_height - EYE_REGION_HEIGHT) // 2

print(f"Processing eye video: {frame_width}x{frame_height} at {fps:.2f} fps")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculate current time in video
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    current_time = frame_number / fps
    
    # Extract eye region (center of frame)
    eye_region = frame[EYE_REGION_Y:EYE_REGION_Y+EYE_REGION_HEIGHT, 
                      EYE_REGION_X:EYE_REGION_X+EYE_REGION_WIDTH]
    
    # Process the eye region with Face Mesh
    eye_region_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(eye_region_rgb)
    
    ear = None  # Initialize ear variable
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get left eye landmarks (we'll use these even if it's actually the right eye)
            eye = []
            for idx in LEFT_EYE_INDICES:  # Using left eye indices as reference
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * EYE_REGION_WIDTH) + EYE_REGION_X
                y = int(landmark.y * EYE_REGION_HEIGHT) + EYE_REGION_Y
                eye.append((x, y))
            
            # Calculate eye aspect ratio
            ear = eye_aspect_ratio(eye)
            
            # Blink detection logic
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if not is_blinking and frame_counter >= CONSECUTIVE_FRAMES:
                    is_blinking = True
                    blink_start_time = current_time
            else:
                if is_blinking:
                    is_blinking = False
                    blink_duration = current_time - blink_start_time
                    if blink_duration >= MIN_BLINK_DURATION:
                        blink_counter += 1
                        blink_data.append({
                            "Blink Number": blink_counter,
                            "Start Time": blink_start_time,
                            "End Time": current_time,
                            "Duration (s)": blink_duration
                        })
                        # Write to CSV
                        with open(csv_filename, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                blink_counter,
                                f"{blink_start_time:.3f}",
                                f"{current_time:.3f}",
                                f"{blink_duration:.3f}"
                            ])
                frame_counter = 0
            
            # Draw eye landmarks on original frame
            for (x, y) in eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    # Display information
    display_text = [
        f"EAR: {ear:.2f}" if ear is not None else "EAR: N/A",
        f"Blinks: {blink_counter}",
        f"Time: {current_time:.1f}s",
        f"Threshold: {EAR_THRESHOLD}",
        f"Frame: {frame_number}"
    ]
    
    for i, text in enumerate(display_text):
        cv2.putText(frame, text, (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show the frame with eye region highlighted
    cv2.rectangle(frame, (EYE_REGION_X, EYE_REGION_Y), 
                 (EYE_REGION_X+EYE_REGION_WIDTH, EYE_REGION_Y+EYE_REGION_HEIGHT),
                 (0, 255, 0), 2)
    
    cv2.imshow('Eye Blink Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

print(f"\nProcessing complete!")
print(f"Total blinks detected: {blink_counter}")
print(f"Data saved to: {csv_filename}")