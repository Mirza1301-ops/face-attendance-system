import cv2
import numpy as np
import os
from datetime import datetime
import mysql.connector

# ================= LABEL MAP =================
label_map = {
    0: ("2229190", "Ayush Swain"),
    1: ("2129063", "Ayush Mohanty"),
    2: ("2229122", "Kushal Mishra"),
    3: ("2229007", "Akhilesh Das"),
    4: ("2229191", "Mirza Baig")
}

# ================= LOAD MODEL =================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# ================= FACE CASCADE =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= CSV =================
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("ID,Name,Date,Time\n")

marked_today = set()

# ================= MYSQL =================
db = mysql.connector.connect(
    host="localhost",
    user="face_user",
    password="face123",
    database="face_attendance_system_2026"
)
cursor = db.cursor()
print("✅ MySQL connected")

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("🎥 Camera started (Press Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        # 🔥 IMPORTANT FIX: Resize face
        face_img = cv2.resize(face_img, (200, 200))

        label, confidence = recognizer.predict(face_img)
        print(f"DEBUG → label={label}, confidence={confidence:.2f}")

        name = "Unknown"

        # 🔥 FINAL CONFIDENCE THRESHOLD
        if confidence < 70 and label in label_map:
            emp_id, name = label_map[label]
            today = datetime.now().date()
            time_now = datetime.now().time()
            unique_key = f"{emp_id}_{today}"

            if unique_key not in marked_today:
                marked_today.add(unique_key)

                # CSV
                with open(attendance_file, "a") as f:
                    f.write(f"{emp_id},{name},{today},{time_now}\n")

                # MYSQL (safe insert)
                cursor.execute(
                    """
                    INSERT INTO attendance_system (ID, Name, attend_date, attend_time)
                    VALUES (%s,%s,%s,%s)
                    """,
                    (emp_id, name, today, time_now)
                )
                db.commit()

                print(f"🟢 Attendance marked: {name}")

        # DRAW
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Live Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()
print("🛑 Camera stopped")
