from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from match_face import match_face_with_db
from uniform import detect_department_from_uniform
from register_user import register_user_base64encoded
from log import save_log_to_database
import mysql.connector
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the 'tmp/' folder as static
app.mount("/tmp", StaticFiles(directory="tmp"), name="tmp")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Intelligent Surveillance and Identification for Secure Campus",
    })

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {
        "request": request,
        "title": "Register User",
    })

@app.post("/match")
async def match_face(request: Request):
    data = await request.json()
    image_data = data.get("image")
    if not image_data:
        return {"error": "No image data provided"}
    result = match_face_with_db(image_data)
    return {
        "name": result.get("name", "Unknown"),
        "department": result.get("department", "Unknown"),
        "captured_uuid": result.get("captured_uuid"),
    }

@app.post("/detect_department")
async def detect_department(request: Request):
    data = await request.json()
    image_data = data.get("image")
    if not image_data:
        return {"error": "No image data provided"}
    
    department, bbox = detect_department_from_uniform(image_data)
    return {
        "department": department,
        "bbox": bbox,
    }

@app.post("/register_user")
async def register_user(request: Request):
    data = await request.json()
    name = data.get("name")
    image_data = data.get("image")
    department = data.get("department")

    if not name or not image_data or not department:
        return {"error": "Name, image, and department are required"}
    
    result = register_user_base64encoded(name, image_data, department)
    
    return result

@app.post("/store_log")
async def store_log(request: Request):
    try:
        data = await request.json()
        name = data.get("name")
        faceDept = data.get("face_dept")
        uniformDept = data.get("uniform_dept")
        status = data.get("reason")
        captured_uuid = data.get("captured_uuid")

        # Save to database
        save_log_to_database({
            "name": name,
            "faceDept": faceDept,
            "uniformDept": uniformDept,
            "status": status,
            "captured_uuid": captured_uuid,
        })

        return {"message": "Log stored successfully"}

    except Exception as e:
        return {"error": f"Failed to store log: {str(e)}"}


@app.get("/logs", response_class=HTMLResponse)
async def view_logs(request: Request, date: str = Query(default="")):
    conn = mysql.connector.connect(
        host="localhost",
        user="bittu",
        password="bittu",
        database="deepface_schema"
    )
    cursor = conn.cursor(dictionary=True)

    # Fetch distinct dates for dropdown
    cursor.execute("SELECT DISTINCT DATE(created_at) as log_date FROM logs_data ORDER BY log_date DESC")
    rows = cursor.fetchall()
    available_dates = [row["log_date"].strftime("%Y-%m-%d") for row in rows]

    # Fetch filtered logs
    if date:
        cursor.execute("""
            SELECT name, face_dept, uniform_dept, status AS reason, created_at AS timestamp, captured_uuid
            FROM logs_data
            WHERE DATE(created_at) = %s
            ORDER BY created_at DESC
        """, (date,))
    else:
        cursor.execute("""
            SELECT name, face_dept, uniform_dept, status AS reason, created_at AS timestamp, captured_uuid
            FROM logs_data
            ORDER BY created_at DESC
            LIMIT 100
        """)

    logs = cursor.fetchall()
    cursor.close()
    conn.close()

    return templates.TemplateResponse("logs.html", {
        "request": request,
        "logs": logs,
        "available_dates": available_dates,
        "selected_date": date
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

