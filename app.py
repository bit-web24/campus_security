from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from match_face import match_face_with_db
from uniform import detect_department_from_uniform

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Intelligent Surveillance and Identification for Secure Campus",
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
    }

@app.post("/detect_department")
async def detect_department(request: Request):
    data = await request.json()
    image_data = data.get("image")
    if not image_data:
        return {"error": "No image data provided"}
    
    department, bbox = detect_department_from_uniform(image_data)
    if department is None:
        return {"error": "Department could not be detected"}
    return {
        "department": department,
        "bbox": bbox,
    }
