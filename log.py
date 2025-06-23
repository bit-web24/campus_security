import mysql.connector
from mysql.connector import Error

def connect_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="bittu",
            password="bittu",
            database="deepface_schema"
        )
        return conn
    except Error as e:
        raise Exception(f"Database connection failed: {e}")


def save_log_to_database(data):
    try:
        conn = connect_db()
        cursor = conn.cursor()

        # Ensure the `logs` table exists or change the table name
        sql = """
            INSERT INTO logs_data (name, face_dept, uniform_dept, status)
            VALUES (%s, %s, %s, %s)
        """
        values = (
            data["name"],
            data["faceDept"],
            data["uniformDept"],
            data["status"]
        )

        cursor.execute(sql, values)
        conn.commit()

    except Error as e:
        raise Exception(f"Failed to insert log: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
