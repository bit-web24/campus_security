from ultralytics import YOLO
from PIL import Image, ImageColor
import numpy as np
from sklearn.cluster import KMeans

model = YOLO("models/best-cloth.pt")

# Class groups
SHIRT_CLASSES = {
    "blazer", "cardigan", "coat", "hoodies", "jacket",
    "longsleeve", "mtm", "padding", "shirt", "shortsleeve",
    "sweater", "zipup"
}

PANT_CLASSES = {
    "cottonpants", "denimpants", "shortpants", "skirt", "slacks", "trainingpants"
}

def get_dominant_color(image):
    """Extract dominant color from PIL image using K-means clustering"""
    try:
        # Resize image to reduce computation time
        image = image.resize((50, 50))
        data = np.array(image)
        
        # Handle grayscale images
        if len(data.shape) == 2:
            return '#808080'  # Return gray for grayscale images
        
        data = data.reshape((-1, 3))  # Flatten pixels to rows
        
        # Remove any invalid pixel values
        data = data[~np.any(data < 0, axis=1)]
        data = data[~np.any(data > 255, axis=1)]
        
        if len(data) == 0:
            return '#808080'  # Default gray if no valid pixels
        
        # Use K-means to find dominant color
        kmeans = KMeans(n_clusters=min(3, len(data)), random_state=0, n_init=10).fit(data)
        counts = np.bincount(kmeans.labels_)
        dominant = kmeans.cluster_centers_[np.argmax(counts)]
        dominant = tuple(map(int, dominant))
        
        return '#{:02x}{:02x}{:02x}'.format(*dominant)
    except Exception as e:
        print(f"Error extracting dominant color: {e}")
        return '#808080'

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    try:
        return ImageColor.getrgb(hex_color)
    except:
        return (128, 128, 128)  # Default gray

def color_distance(c1, c2):
    """Calculate Euclidean distance between two RGB colors"""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def get_department_from_color(color_hex, threshold=180):
    """Map color to department based on predefined color scheme"""
    if not color_hex:
        return "Unknown"
    
    try:
        color_rgb = hex_to_rgb(color_hex)
        
        # Department color mapping
        color_map = {
            "BCA": hex_to_rgb("#ffadbb"),      # Light pink
            "B.Tech": hex_to_rgb("#d7c18b"),   # Wheat
            "B.Pharma": hex_to_rgb("#62b2d1"), # Sky blue
            "BMLT": hex_to_rgb("#c0588c"),     # Dark pink
            "MBA": hex_to_rgb("#ffff00"),      # Yellow (fixed from white)
        }
        
        closest_dept = "Unknown"
        min_distance = float("inf")
        
        for dept, ref_rgb in color_map.items():
            dist = color_distance(color_rgb, ref_rgb)
            if dist < min_distance:
                min_distance = dist
                closest_dept = dept
        
        return closest_dept if min_distance < threshold else "Unknown"
    except Exception as e:
        print(f"Error mapping color to department: {e}")
        return "Unknown"

def detect_objects_on_image(image):
    """Detect objects in image and return detected departments from shirts only"""
    try:
        results = model.predict(image, conf=0.3, verbose=False)
        if not results:
            return []
        result = results[0]
        departments = []
        for box in result.boxes:
            try:
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                class_id = int(box.cls[0].item())
                label = result.names[class_id]
                if label in SHIRT_CLASSES:
                    # Ensure coordinates are within image bounds
                    img_width, img_height = image.size
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))
                    if x2 > x1 and y2 > y1:
                        cropped = image.crop((x1, y1, x2, y2))
                        color = get_dominant_color(cropped)
                        department = get_department_from_color(color)
                        if department and department != "Unknown":
                            departments.append(department)
            except Exception as e:
                print(f"Error processing detection box: {e}")
                continue
        return departments[0] if departments else "Unknown"
    except Exception as e:
        print(f"Error in detect_objects_on_image: {e}")
        return "Unknown"
