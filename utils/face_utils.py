import cv2
import mysql.connector
from deepface import DeepFace
import os
import base64
import numpy as np

def connect_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="bittu",
        password="bittu",
        database="campus_security"
    )
    return conn

def fetch_users(cursor):
    cursor.execute("SELECT name, face, department FROM registered_users")
    return [{'name': row[0], 'face': row[1], 'department': row[2]} for row in cursor.fetchall()]

def base64_to_image(base64_str):
    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def detect_and_match_faces(image_path, users):
    frame = cv2.imread(image_path)

    try:
        detections = DeepFace.extract_faces(frame, enforce_detection=False)
        for det in detections:
            face_img = det['face']
            matched_name = "Unknown"

            for user in users:
                db_face_img = base64_to_image(user['face'])
                result = DeepFace.verify(face_img, db_face_img, enforce_detection=False, model_name='VGG-Face')
                if result['verified']:
                    matched_name = user['name']
                    break

            facial_area = det['facial_area']
            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, matched_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if matched_name != "Unknown":
                processed_image_path = os.path.splitext(image_path)[0] + "_processed.jpg"
                cv2.imwrite(processed_image_path, frame)
                return matched_name, processed_image_path  # Return first matched name

    except Exception as e:
        print("Error:", e)

    processed_image_path = os.path.splitext(image_path)[0] + "_processed.jpg"
    cv2.imwrite(processed_image_path, frame)
    return "Unknown", processed_image_path
