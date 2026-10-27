from flask import Flask, Response
import cv2
import numpy as np
import os
from datetime import datetime
import time

app = Flask(__name__)

# ============================================================
# EXTERNAL USB WEBCAM
# ============================================================

camera = cv2.VideoCapture(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# FACE RECOGNITION MODEL
# ============================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")


# ============================================================
# REGISTERED PEOPLE
# ============================================================

names = np.load(
    "names.npy",
    allow_pickle=True
).item()


# ============================================================
# UNKNOWN FACE STORAGE
# ============================================================

os.makedirs("unknown_faces", exist_ok=True)

last_unknown_capture = 0
CAPTURE_COOLDOWN = 10


# ============================================================
# RECOGNITION SETTINGS
# ============================================================

# Lower LBPH value = better match
RECOGNITION_THRESHOLD = 100


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    global last_unknown_capture

    while True:

        success, frame = camera.read()

        if not success:
            continue

        # Convert camera image to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ====================================================
        # FACE DETECTION
        # ====================================================

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100)
        )


        # ====================================================
        # NO FACE
        # ====================================================

        if len(faces) == 0:

            cv2.putText(
                frame,
                "No face detected",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


        # ====================================================
        # PROCESS EACH FACE
        # ====================================================

        for (x, y, w, h) in faces:

            face = gray[
                y:y+h,
                x:x+w
            ]


            # Ignore very small faces
            if w < 100 or h < 100:
                continue


            # =================================================
            # BLUR / QUALITY CHECK
            # =================================================

            sharpness = cv2.Laplacian(
                face,
                cv2.CV_64F
            ).var()


            if sharpness < 40:

                box_color = (255, 255, 0)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    box_color,
                    3
                )

                cv2.putText(
                    frame,
                    "Face unclear",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

                continue


            # =================================================
            # FACE RECOGNITION
            # =================================================

            label, confidence = recognizer.predict(face)


            # =================================================
            # REGISTERED PERSON
            # =================================================

            if confidence < RECOGNITION_THRESHOLD:

                name = names.get(
                    label,
                    "Unknown"
                )

                text = name

                box_color = (
                    0,
                    255,
                    0
                )


            # =================================================
            # UNKNOWN PERSON
            # =================================================

            else:

                text = "UNKNOWN PERSON"

                box_color = (
                    0,
                    0,
                    255
                )


                # =============================================
                # CAPTURE UNKNOWN FACE
                # =============================================

                current_time = time.time()


                if (
                    current_time -
                    last_unknown_capture
                    >= CAPTURE_COOLDOWN
                ):

                    timestamp = datetime.now().strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )


                    filename = os.path.join(
                        "unknown_faces",
                        f"unknown_{timestamp}.jpg"
                    )


                    # Save only the detected face
                    face_crop = frame[
                        y:y+h,
                        x:x+w
                    ]


                    cv2.imwrite(
                        filename,
                        face_crop
                    )


                    last_unknown_capture = current_time


                    print(
                        f"Unknown face captured: {filename}"
                    )


            # =================================================
            # FACE RECTANGLE
            # =================================================

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                box_color,
                3
            )


            # =================================================
            # PERSON NAME
            # =================================================

            cv2.putText(
                frame,
                text,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2
            )


            # =================================================
            # CONFIDENCE
            # =================================================

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}",
                (x, y+h+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )


        # ====================================================
        # INSPECTIQ TITLE
        # ====================================================

        cv2.putText(
            frame,
            "InspectIQ",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )


        # ====================================================
        # CONVERT FRAME TO JPEG
        # ====================================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )


        if not ret:
            continue


        frame_bytes = buffer.tobytes()


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>InspectIQ</title>

        <style>

            body {
                margin: 0;
                padding: 0;
                background: black;
                text-align: center;
            }

            img {
                width: 100%;
                max-width: 1280px;
                display: block;
                margin: auto;
            }

        </style>

    </head>


    <body>

        <img src="/video">

    </body>

    </html>
    """


# ============================================================
# VIDEO ROUTE
# ============================================================

@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )