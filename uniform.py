import base64
from io import BytesIO
from PIL import Image
from identify_department import identify_department_from_uniform_colorcode

def detect_department_from_uniform(image_data: str) -> str:
    try:
        # Decode base64 image
        if not image_data:
            raise ValueError("Empty image data provided")
        
        # Handle base64 data with or without header (e.g., "data:image/jpeg;base64,")
        header, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
        try:
            img_bytes = base64.b64decode(encoded)
        except base64.binascii.Error as e:
            raise ValueError(f"Invalid base64 encoding: {e}")
        
        # Convert to PIL Image
        try:
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")
        
        # Detect objects (shirt) and extract department
        department = identify_department_from_uniform_colorcode(image)
        if not department:
            print("Warning: No valid department detected, returning default")
            return "Unknown"
        return department
    
    except Exception as e:
        print(f"Error in detect_department_from_uniform: {e}")
        return "Unknown"