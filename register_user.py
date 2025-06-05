import mysql.connector
import base64

# Read and encode the image in base64
with open("faces/rohit.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# Connect to the database
conn = mysql.connector.connect(
    host="localhost",
    user="bittu",
    password="bittu",
    database="deepface_schema"
)

cursor = conn.cursor()

# Insert the name and base64-encoded image string
cursor.execute(
    "INSERT INTO registered_users (name, face, department) VALUES (%s, %s, %s)",
    ("Rohit", encoded_string, "BCA")
)

conn.commit()
conn.close()
