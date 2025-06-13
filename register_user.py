import mysql.connector
import base64

# Read and encode the image in base64
# with open("faces/rohit.jpg", "rb") as image_file:
#     encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

def register_user_base64encoded(name: str, image_data: str, department: str):
    try:
        # Strip data URL header if present
        _header, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
        
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
            (name, encoded, department)
        )

        conn.commit()
        conn.close()
        return {
            "success": True,
            "message": f"User {name} registered successfully in {department} department."
        }
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return {
            "success": False,
            "message": f"Failed to register user {name}: {err}"
        }
