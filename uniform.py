import base64
import numpy as np
import cv2
from identify_department import identify_department_from_uniform_colorcode

def detect_department_from_uniform(image_data: str) -> str:
    try:
        # Validate input
        if not image_data:
            raise ValueError("Empty image data provided")
        
        # Strip data URL header if present
        header, encoded = image_data.split(",", 1) if "," in image_data else ("", image_data)
        
        # Decode base64
        try:
            img_bytes = base64.b64decode(encoded)
        except base64.binascii.Error as e:
            raise ValueError(f"Invalid base64 encoding: {e}")
        
        # Convert bytes to NumPy array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        
        # Decode NumPy array to OpenCV image (BGR)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from bytes")

        # Detect shirt & department
        results = identify_department_from_uniform_colorcode(image)
        if not results["department"]:
            print("Warning: No valid department detected, returning default")
            return "Unknown", [0, 50, 0, 50]
        
        return results["department"], results["bounding_box"]
    
    except Exception as e:
        print(f"Error in detect_department_from_uniform: {e}")
        return "Unknown", [0, 50, 0, 50]
