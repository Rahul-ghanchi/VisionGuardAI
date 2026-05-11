from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO

import cv2
import os
import time
import threading
import yagmail
import pyttsx3
import face_recognition
import numpy as np
import pywhatkit
import matplotlib.pyplot as plt

from datetime import datetime

from reportlab.pdfgen import canvas

from database import SessionLocal
from database import IntruderLog

# =========================
# APP
# =========================

app = FastAPI()

# =========================
# FOLDERS
# =========================

folders = [
    "../intruders",
    "../reports",
    "../screenshots",
    "../alerts"
]

for folder in folders:

    if not os.path.exists(folder):
        os.makedirs(folder)

# =========================
# STATIC FILES
# =========================

app.mount(
    "/intruders",
    StaticFiles(directory="../intruders"),
    name="intruders"
)

app.mount(
    "/alerts",
    StaticFiles(directory="../alerts"),
    name="alerts"
)

# =========================
# LOGIN DETAILS
# =========================

USERNAME = "admin"
PASSWORD = "visionguard"

# =========================
# EMAIL SETTINGS
# =========================

EMAIL_SENDER = "24amtics559@gmail.com"

EMAIL_PASSWORD = "hzsz cofc nija ohrc"

EMAIL_RECEIVER = "24amtics559@gmail.com"

# =========================
# WHATSAPP NUMBER
# =========================

WHATSAPP_NUMBER = "+918320765444"

# =========================
# DATABASE
# =========================

db = SessionLocal()

# =========================
# AI MODELS
# =========================

model = YOLO("../models/yolov8n.pt")

weapon_model = YOLO("../models/yolov8n.pt")

# =========================
# CAMERA
# =========================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# VOICE ENGINE
# =========================

engine = pyttsx3.init()

# =========================
# FACE RECOGNITION
# =========================

known_face_encodings = []

known_face_names = []

rahul_image = face_recognition.load_image_file(
    "../known_faces/rahul.jpg"
)

rahul_encoding = face_recognition.face_encodings(
    rahul_image
)[0]

known_face_encodings.append(
    rahul_encoding
)

known_face_names.append(
    "Rahul Ghanchi"
)

# =========================
# =========================
# GLOBAL VARIABLES
# =========================

person_count = 0

fps_value = 0

intruder_count = 0

security_status = "SAFE"

weapon_detected = "NO"

recording = False

video_writer = None

latest_frame = None

last_save_time = 0

# =========================
# MOTION DETECTION
# =========================

previous_frame = None

motion_detected = False

# =========================
# CAMERA PROCESS
# =========================

def process_camera():

    global latest_frame
    global person_count
    global fps_value
    global intruder_count
    global security_status
    global weapon_detected
    global recording
    global video_writer
    global last_save_time
    global previous_frame
    global motion_detected

    while True:

        success, frame = camera.read()

        if not success:
            continue

        start = time.time()

        # =========================
        # MOTION DETECTION
        # =========================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (21, 21),
            0
        )

        if previous_frame is None:

            previous_frame = gray

            continue

        frame_diff = cv2.absdiff(
            previous_frame,
            gray
        )

        threshold = cv2.threshold(
            frame_diff,
            25,
            255,
            cv2.THRESH_BINARY
        )[1]

        threshold = cv2.dilate(
            threshold,
            None,
            iterations=2
        )

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False

        for contour in contours:

            if cv2.contourArea(contour) > 5000:

                motion_detected = True

                break

        previous_frame = gray

        # =========================
        # RUN AI ONLY IF MOTION
        # =========================

        if not motion_detected:

            latest_frame = frame

            continue

        # =========================
        # OPTIMIZED FRAME
        # =========================

        small_frame = cv2.resize(
            frame,
            (640, 480)
        )

        # =========================
        # YOLO DETECTION
        # =========================

        results = model(
            small_frame,
            imgsz=320,
            conf=0.45
        )

        weapon_results = weapon_model(
            small_frame,
            imgsz=320,
            conf=0.45
        )

        person_count = 0

        # =========================
        # FAST FACE RECOGNITION
        # =========================

        small_face_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.25,
            fy=0.25
        )

        rgb_frame = cv2.cvtColor(
            small_face_frame,
            cv2.COLOR_BGR2RGB
        )

        face_locations = face_recognition.face_locations(
            rgb_frame,
            model="hog"
        )

        face_encodings = face_recognition.face_encodings(
            rgb_frame,
            face_locations
        )

        detected_names = []

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
                known_face_encodings,
                face_encoding
            )

            name = "Unknown Person"

            face_distances = (
                face_recognition.face_distance(
                    known_face_encodings,
                    face_encoding
                )
            )

            best_match_index = np.argmin(
                face_distances
            )

            if matches[best_match_index]:

                name = known_face_names[
                    best_match_index
                ]

            detected_names.append(name)

        face_index = 0

        # =========================
        # PERSON DETECTION
        # =========================

        for result in results:

            boxes = result.boxes

            for box in boxes:

                cls = int(box.cls[0])

                class_name = model.names[cls]

                if class_name == "person":

                    person_count += 1

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    if face_index < len(detected_names):

                        detected_name = detected_names[
                            face_index
                        ]

                    else:

                        detected_name = "Unknown Person"

                    face_index += 1

                    if detected_name == "Unknown Person":

                        color = (0, 0, 255)

                    else:

                        color = (0, 255, 0)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        3
                    )

                    cv2.putText(
                        frame,
                        detected_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        3
                    )

                    # =========================
                    # UNKNOWN PERSON ALERT
                    # =========================

                    if detected_name == "Unknown Person":

                        security_status = (
                            "UNKNOWN PERSON DETECTED"
                        )

                        current_time = time.time()

                        if (
                            current_time - last_save_time
                            > 10
                        ):

                            last_save_time = current_time

                            intruder_count += 1

                            filename = (
                                "../intruders/"
                                f"intruder_"
                                f"{int(time.time())}.jpg"
                            )

                            cv2.imwrite(
                                filename,
                                frame
                            )

                            try:

                                yag = yagmail.SMTP(
                                    EMAIL_SENDER,
                                    EMAIL_PASSWORD
                                )

                                yag.send(
                                    to=EMAIL_RECEIVER,
                                    subject="VisionGuard Alert",
                                    contents="Unknown Person Detected",
                                    attachments=filename
                                )

                            except Exception as e:

                                print(e)

                            threading.Thread(
                                target=speak_alert
                            ).start()

                    else:

                        security_status = "SAFE"

        # =========================
        # WEAPON DETECTION
        # =========================

        weapon_detected = "NO"

        for result in weapon_results:

            boxes = result.boxes

            for box in boxes:

                cls = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = weapon_model.names[cls]

                dangerous_items = [
                    "knife",
                    "scissors",
                    "baseball bat"
                ]

                if (
                    class_name in dangerous_items
                    and confidence > 0.40
                ):

                    weapon_detected = "YES"

                    security_status = (
                        "WEAPON DETECTED"
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        4
                    )

                    cv2.putText(
                        frame,
                        f"WEAPON: {class_name}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    # =========================
                    # AUTO RECORDING
                    # =========================

                    if not recording:

                        recording = True

                        video_name = (
                            "../alerts/"
                            f"weapon_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                        )

                        fourcc = cv2.VideoWriter_fourcc(
                            *'XVID'
                        )

                        video_writer = cv2.VideoWriter(
                            video_name,
                            fourcc,
                            20.0,
                            (640, 480)
                        )

                    if recording and video_writer:

                        video_writer.write(frame)

                else:

                    if recording and video_writer:

                        video_writer.release()

                        recording = False

        # =========================
        # FPS
        # =========================

        end = time.time()

        fps_value = int(
            1 / (end - start + 0.001)
        )

        latest_frame = frame

# =========================
# THREAD START
# =========================

threading.Thread(
    target=process_camera,
    daemon=True
).start()

# =========================
# VIDEO STREAM
# =========================

def generate_frames():

    global latest_frame

    while True:

        if latest_frame is None:
            continue

        _, buffer = cv2.imencode(
            ".jpg",
            latest_frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )

# =========================
# LOGIN PAGE
# =========================

@app.get("/", response_class=HTMLResponse)
def login_page():

    return """

    <html>

    <body style="
        background:black;
        color:white;
        font-family:Arial;
        text-align:center;
    ">

        <h1 style="
            color:lime;
            margin-top:100px;
        ">

            VisionGuard AI Login

        </h1>

        <form action="/login">

            <input
                type="text"
                name="username"
                placeholder="Username"
                style="
                    padding:15px;
                    width:300px;
                "
            >

            <br><br>

            <input
                type="password"
                name="password"
                placeholder="Password"
                style="
                    padding:15px;
                    width:300px;
                "
            >

            <br><br>

            <button
                type="submit"
                style="
                    padding:15px;
                    width:200px;
                    background:lime;
                    font-size:20px;
                "
            >

                LOGIN

            </button>

        </form>

    </body>

    </html>

    """

# =========================
# LOGIN CHECK
# =========================

@app.get("/login")
def login(username: str, password: str):

    if (
        username == USERNAME and
        password == PASSWORD
    ):

        return RedirectResponse(
            url="/dashboard",
            status_code=302
        )

    return HTMLResponse(
        "<h1 style='color:red'>Invalid Login</h1>"
    )

# =========================
# DASHBOARD
# =========================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():

    gallery = ""

    files = os.listdir("../intruders")

    files = sorted(
        files,
        reverse=True
    )[:8]

    for file in files:

        gallery += f"""

        <img
            src='/intruders/{file}'
            width='250'
            style='
                border:4px solid red;
                border-radius:15px;
            '
        >

        """

    logs = db.query(
        IntruderLog
    ).order_by(
        IntruderLog.id.desc()
    ).limit(10).all()

    logs_html = ""

    for log in logs:

        logs_html += f"""

        <tr>

            <td>{log.person_name}</td>

            <td>{log.detection_time}</td>

            <td>{log.alert_status}</td>

        </tr>

        """

    return f"""

    <html>

    <body style="
        background:#020617;
        color:white;
        font-family:Arial;
    ">

        <h1 style="
            color:lime;
            text-align:center;
        ">

            VisionGuard AI Dashboard

        </h1>

        <h2 style="
            text-align:center;
        ">

            👑 Rahul Ghanchi

        </h2>

        <div style="
            display:flex;
            justify-content:center;
            gap:30px;
            flex-wrap:wrap;
        ">

            <div style="
                background:#111827;
                padding:20px;
                border-radius:15px;
                width:250px;
                text-align:center;
            ">

                <h1>{person_count}</h1>

                <p>Persons</p>

            </div>

            <div style="
                background:#111827;
                padding:20px;
                border-radius:15px;
                width:250px;
                text-align:center;
            ">

                <h1>{fps_value}</h1>

                <p>FPS</p>

            </div>

            <div style="
                background:#111827;
                padding:20px;
                border-radius:15px;
                width:250px;
                text-align:center;
            ">

                <h1>{intruder_count}</h1>

                <p>Alerts</p>

            </div>

            <div style="
                background:#111827;
                padding:20px;
                border-radius:15px;
                width:250px;
                text-align:center;
            ">

                <h1>{weapon_detected}</h1>

                <p>Weapon Detection</p>

            </div>

        </div>

        <h1 style="
            text-align:center;
            color:red;
            margin-top:30px;
        ">

            🚨 {security_status}

        </h1>

        <div style="text-align:center;">
        <div style="
    text-align:center;
    margin-top:30px;
">

    <a href="/report">

        <button
            style="
                padding:20px;
                background:lime;
                font-size:20px;
                border:none;
                border-radius:15px;
                cursor:pointer;
            "
        >

            DOWNLOAD REPORT

        </button>

    </a>

    <a href="/analytics-chart">

        <button
            style="
                padding:20px;
                background:cyan;
                font-size:20px;
                border:none;
                border-radius:15px;
                cursor:pointer;
                margin-left:20px;
            "
        >

            DOWNLOAD ANALYTICS

        </button>

    </a>

</div>

            <img
                src="/video"
                width="1000"
                style="
                    border:5px solid lime;
                    border-radius:20px;
                    margin-top:20px;
                "
            >

        </div>

        <h1 style="
            color:cyan;
            margin-left:20px;
            margin-top:50px;
        ">

            Live Security Logs

        </h1>

        <table
            border="1"
            cellpadding="15"
            style="
                width:95%;
                margin:20px;
                border-collapse:collapse;
                background:#111827;
                color:white;
                font-size:20px;
            "
        >

            <tr style="background:red;">

                <th>Person</th>

                <th>Detection Time</th>

                <th>Status</th>

            </tr>

            {logs_html}

        </table>

        <h1 style="
            color:lime;
            margin-left:20px;
        ">

            Intruder Gallery

        </h1>

        <div style="
            display:flex;
            gap:20px;
            flex-wrap:wrap;
            padding:20px;
        ">

            {gallery}

        </div>

    </body>

    </html>

    """

# =========================
# VIDEO ROUTE
# =========================

@app.get("/video")
def video():

    return StreamingResponse(
        generate_frames(),
        media_type=(
            "multipart/x-mixed-replace; boundary=frame"
        )
    )

# =========================
# PDF REPORT
# =========================

@app.get("/report")
def report():

    report_file = "../reports/security_report.pdf"

    pdf = canvas.Canvas(report_file)

    pdf.setFont(
        "Helvetica-Bold",
        25
    )

    pdf.drawString(
        120,
        800,
        "VisionGuard AI Report"
    )

    pdf.setFont(
        "Helvetica",
        18
    )

    pdf.drawString(
        100,
        700,
        f"Persons Detected: {person_count}"
    )

    pdf.drawString(
        100,
        650,
        f"Intruder Alerts: {intruder_count}"
    )

    pdf.drawString(
        100,
        600,
        f"FPS: {fps_value}"
    )

    pdf.drawString(
        100,
        550,
        f"Security Status: {security_status}"
    )

    pdf.save()

    return FileResponse(
        report_file,
        media_type='application/pdf',
        filename='VisionGuard_Report.pdf'
    )

# =========================
# ANALYTICS CHART
# =========================

@app.get("/analytics-chart")
def analytics_chart():

    labels = [
        "Persons",
        "Alerts"
    ]

    values = [
        person_count,
        intruder_count
    ]

    plt.figure(figsize=(6, 5))

    plt.bar(
        labels,
        values
    )

    plt.title(
        "VisionGuard AI Analytics"
    )

    plt.xlabel(
        "Category"
    )

    plt.ylabel(
        "Count"
    )

    chart_path = (
        "../reports/analytics_chart.png"
    )

    plt.savefig(chart_path)

    plt.close()

    return FileResponse(
        chart_path,
        media_type="image/png",
        filename="analytics_chart.png"
    )