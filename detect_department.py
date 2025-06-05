import base64
from io import BytesIO
from PIL import Image

from dominant_color import detect_objects_on_image

def detect_department_from_uniform(image_data: str) -> str:
    try:
        # Decode base64 image
        header, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        # Detect objects and extract department
        detections = detect_objects_on_image(image)
        for det in detections:
            # det: [x1, y1, x2, y2, category, prob, color, department]
            if det[4] == "shirt" and det[7] and det[7] != "Unknown":
                return det[7]
        return "Unknown"
    except Exception as e:
        print(f"Error in detect_department_from_uniform: {e}")
        return "Unknown"
