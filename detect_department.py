import base64
from io import BytesIO
from PIL import Image

from dominant_color import detect_department_from_uniform_colorcode

def detect_department_from_uniform(image_data: str) -> str:
    try:
        # Decode base64 image
        header, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        # Detect objects (shirt) and extract department
        department = detect_department_from_uniform_colorcode(image)
        return department
    except Exception as e:
        print(f"Error in detect_department_from_uniform: {e}")
        return "B.TECH"
