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

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV color space"""
    r, g, b = r/255.0, g/255.0, b/255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Hue calculation
    h = 0
    if diff == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif max_val == b:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # Saturation calculation
    s = 0 if max_val == 0 else (diff / max_val) * 100
    
    # Value calculation
    v = max_val * 100
    
    return h, s, v

def is_color_in_range(color_rgb, target_ranges):
    """Check if color falls within any of the target ranges"""
    r, g, b = color_rgb
    h, s, v = rgb_to_hsv(r, g, b)
    
    for range_data in target_ranges:
        h_min, h_max = range_data['hue']
        s_min, s_max = range_data['saturation']
        v_min, v_max = range_data['value']
        
        # Handle hue wraparound (e.g., red spans 350-360 and 0-10)
        if h_min > h_max:  # Wraparound case
            h_match = h >= h_min or h <= h_max
        else:
            h_match = h_min <= h <= h_max
        
        s_match = s_min <= s <= s_max
        v_match = v_min <= v <= v_max
        
        if h_match and s_match and v_match:
            return True
    
    return False

def get_department_from_color(color_hex):
    """Map color to department based on predefined color ranges"""
    if not color_hex:
        return "Unknown"
    
    try:
        color_rgb = hex_to_rgb(color_hex)
        
        # Department color range mapping (HSV ranges)
        department_ranges = {
            "BCA": [
                # Light pink ranges
                {'hue': (330, 360), 'saturation': (20, 60), 'value': (80, 100)},  # Light pink
                {'hue': (0, 30), 'saturation': (20, 60), 'value': (80, 100)},    # Light pink wraparound
            ],
            "B.Tech": [
                # Gray/wheat ranges
                {'hue': (0, 360), 'saturation': (0, 20), 'value': (50, 80)},     # Gray tones
                {'hue': (40, 60), 'saturation': (10, 40), 'value': (60, 85)},    # Wheat/beige
            ],
            "B.Pharma": [
                # Sky blue ranges
                {'hue': (180, 220), 'saturation': (40, 80), 'value': (60, 90)},  # Sky blue
                {'hue': (200, 240), 'saturation': (30, 70), 'value': (70, 95)},  # Light blue
            ],
            "BMLT": [
                # Dark pink ranges
                {'hue': (320, 350), 'saturation': (40, 80), 'value': (40, 70)},  # Dark pink
                {'hue': (300, 330), 'saturation': (50, 90), 'value': (45, 75)},  # Magenta-pink
            ],
            "MBA": [
                # Yellow ranges
                {'hue': (50, 70), 'saturation': (80, 100), 'value': (90, 100)},  # Bright yellow
                {'hue': (45, 75), 'saturation': (70, 100), 'value': (85, 100)},  # Yellow variations
            ],
        }
        
        # Check each department's color ranges
        for dept, ranges in department_ranges.items():
            if is_color_in_range(color_rgb, ranges):
                return dept
        
        return "Unknown"
        
    except Exception as e:
        print(f"Error mapping color to department: {e}")
        return "Unknown"

def detect_department_from_uniform_colorcode(image):
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
