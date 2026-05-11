import cv2
import face_recognition
import os
import time
from playsound import playsound

# =========================
# Create Required Folders
# =========================

if not os.path.exists("../intruders"):
    os.makedirs("../intruders")

if not os.path.exists("../alerts"):
    os.makedirs("../alerts")

# =========================
# Load Known Face
# =========================

known_image = face_recognition.load_image_file(
    "../known_faces/rahul.jpg"
)

known_encoding = face_recognition.face_encodings(
    known_image
)[0]

known_face_encodings = [known_encoding]

known_face_names = ["Rahul"]

# =========================
# Open Webcam
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not found")
    exit()

print("VisionGuard AI Security Started")

# =========================
# FPS Variables
# =========================

prev_time = 0

# Alert Cooldown
last_alert_time = 0

# =========================
# Main Loop
# =========================

while True:

    # Read Frame
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame")
        break

    # Resize Frame
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=0.25,
        fy=0.25
    )

    # Convert to RGB
    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect Faces
    face_locations = face_recognition.face_locations(
        rgb_small_frame
    )

    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )

    # =========================
    # Process Faces
    # =========================

    for (top, right, bottom, left), face_encoding in zip(
        face_locations,
        face_encodings
    ):

        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding
        )

        name = "Unknown"

        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )

        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Scale Back Coordinates
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # =========================
        # Unknown Person Alert
        # =========================

        if name == "Unknown":

            color = (0, 0, 255)

            current_time = time.time()

            # Alert every 5 seconds
            if current_time - last_alert_time > 5:

                last_alert_time = current_time

                # Screenshot Filename
                timestamp = int(time.time())

                filename = (
                    f"../intruders/intruder_{timestamp}.jpg"
                )

                # Save Screenshot
                success = cv2.imwrite(filename, frame)

                if success:
                    print(f"Screenshot Saved: {filename}")
                else:
                    print("Screenshot Failed!")

                # Play Alert Sound
                try:
                    playsound("../alerts/alert.mp3")
                except:
                    print("Alert sound not found!")

                print("WARNING: Unknown Person Detected!")

        else:

            color = (0, 255, 0)

        # =========================
        # Draw Face Box
        # =========================

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            3
        )

        # =========================
        # Draw Label Box
        # =========================

        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            color,
            cv2.FILLED
        )

        # =========================
        # Draw Name Text
        # =========================

        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
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
    # Security Status
    # =========================

    cv2.putText(
        frame,
        "VisionGuard AI Security Active",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Green = Authorized",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Red = Intruder",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =========================
    # Show Window
    # =========================

    cv2.imshow(
        "VisionGuard AI - Security System",
        frame
    )

    # =========================
    # Exit Key
    # =========================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print("Exiting VisionGuard AI...")

        break

# =========================
# Release Resources
# =========================

cap.release()

cv2.destroyAllWindows()