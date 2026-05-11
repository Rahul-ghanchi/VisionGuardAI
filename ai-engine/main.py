from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import cv2
import time
import os

# =========================
# Load YOLOv8 Model
# =========================
model = YOLO("yolov8n.pt")

# =========================
# Initialize Tracker
# =========================
tracker = DeepSort(max_age=30)

# =========================
# Create Screenshot Folder
# =========================
if not os.path.exists("../screenshots"):
    os.makedirs("../screenshots")

# =========================
# Open Webcam
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

# =========================
# FPS Variables
# =========================
prev_time = 0

print("VisionGuard AI Tracking Started")

# =========================
# Main Loop
# =========================
while True:

    # Read Frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # =========================
    # Run YOLO Detection
    # =========================
    results = model(frame)

    detections = []

    person_count = 0

    # =========================
    # Process Detections
    # =========================
    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0]

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        # Only track persons
        if class_name == "person":

            person_count += 1

            detections.append(
                (
                    [x1, y1, x2 - x1, y2 - y1],
                    confidence,
                    class_name
                )
            )

    # =========================
    # Update Tracker
    # =========================
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    # =========================
    # Draw Tracking
    # =========================
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        # Draw Rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Draw ID
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # =========================
    # FPS Calculation
    # =========================
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # =========================
    # Display FPS
    # =========================
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # =========================
    # Display Person Count
    # =========================
    cv2.putText(
        frame,
        f"Persons: {person_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # =========================
    # Instructions
    # =========================
    cv2.putText(
        frame,
        "Press S to Save Screenshot",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =========================
    # Show Window
    # =========================
    cv2.imshow(
        "VisionGuard AI - Person Tracking",
        frame
    )

    # =========================
    # Keyboard Controls
    # =========================
    key = cv2.waitKey(1)

    # Save Screenshot
    if key & 0xFF == ord("s"):

        timestamp = int(time.time())

        filename = f"../screenshots/tracking_{timestamp}.jpg"

        cv2.imwrite(filename, frame)

        print(f"Screenshot Saved: {filename}")

    # Quit Program
    if key & 0xFF == ord("q"):

        print("Exiting VisionGuard AI...")

        break

# =========================
# Release Resources
# =========================
cap.release()

cv2.destroyAllWindows()