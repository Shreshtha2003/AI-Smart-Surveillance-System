import cv2
import face_recognition
import pickle
import numpy as np
import simpleaudio as sa
import smtplib
from email.message import EmailMessage
import os
from datetime import datetime
import threading

# Paths
encoding_file = "../encodings/encodings.pkl"
alert_sound_path = "../alerts/alert.wav"
unknown_dir = "../unknown"

# Create unknown directory if not exists
os.makedirs(unknown_dir, exist_ok=True)

# Load face encodings
with open(encoding_file, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Track unknown face encodings (hashed)
already_alerted = set()

# Sound playback control
sound_active = False
sound_obj = None

def play_warning_sound_1_minute():
    global sound_active, sound_obj
    if sound_active:
        return  # Avoid overlapping

    try:
        wave_obj = sa.WaveObject.from_wave_file(alert_sound_path)
        sound_obj = wave_obj.play()
        sound_active = True

        # Stop the sound after 60 seconds
        threading.Timer(60, stop_warning_sound).start()
    except Exception as e:
        print(f"[SOUND ERROR] {e}")

def stop_warning_sound():
    global sound_active, sound_obj
    if sound_obj:
        sound_obj.stop()
    sound_active = False

# Email alert
def send_email_alert(image_path):
    try:
        EMAIL_ADDRESS = "your_email@gmail.com"
        EMAIL_PASSWORD = "your_app_password"
        RECEIVER_EMAIL = "receiver_email@gmail.com"

        msg = EmailMessage()
        msg['Subject'] = '🔴 ALERT: Unknown Face Detected'
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECEIVER_EMAIL
        msg.set_content("Alert: Unknown face detected by AI Surveillance System.")

        with open(image_path, "rb") as img:
            msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print("[EMAIL SENT] Alert email with image sent.")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

# Initialize camera
print("[INFO] Starting camera...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    names = []

    for encoding, box in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
        name = "Unknown"

        if True in matches:
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matched_idxs:
                counts[known_names[i]] = counts.get(known_names[i], 0) + 1

            name = max(counts, key=counts.get)

        # Handle unknown
        if name == "Unknown":
            top, right, bottom, left = [v * 4 for v in box]
            face_image = frame[top:bottom, left:right]
            face_encoding_hash = hash(encoding.tobytes())

            if face_encoding_hash not in already_alerted:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(unknown_dir, f"unknown_{timestamp}.jpg")
                cv2.imwrite(image_path, face_image)
                play_warning_sound_1_minute()
                send_email_alert(image_path)
                already_alerted.add(face_encoding_hash)

        names.append(name)

    # Draw boxes
    for ((top, right, bottom, left), name) in zip(face_locations, names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("AI Smart Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()