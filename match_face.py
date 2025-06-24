import mysql.connector
import base64
import re
import uuid
import os
from face_utils import detect_and_match_faces

def match_face_with_db(image_data: str):
    # Extract base64 part from data URL
    match = re.match(r"data:image\/\w+;base64,(.+)", image_data)
    if not match:
        return {"name": "Invalid image data"}
    image_b64 = match.group(1)
    image_bytes = base64.b64decode(image_b64)
    
    # Save to a temporary file
    captured_uuid = uuid.uuid4().hex
    temp_filename = f"tmp/{captured_uuid}.jpg"
    with open(temp_filename, "wb") as f:
        f.write(image_bytes)
    
    # Connect to DB and fetch users
    conn = mysql.connector.connect(
        host="localhost",
        user="bittu",
        password="bittu",
        database="deepface_schema"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT name, face, department FROM registered_users")
    users = cursor.fetchall()
    
    # Detect and match faces
    matched_name, processed_image_path = detect_and_match_faces(temp_filename, users)
    
    # Clean up temp file
    os.remove(temp_filename)
    
    return {
        "name": matched_name,
        "department": next((user['department'] for user in users if user['name'] == matched_name), "Unknown"),
        "captured_uuid": captured_uuid,
    }
