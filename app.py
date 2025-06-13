from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from match_face import match_face_with_db
from uniform import detect_department_from_uniform
from register_user import register_user_base64encoded

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

