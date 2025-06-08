import colorsys
import numpy as np

DEPARTMENT_COLORS = {
    "BTECH": ["#a4a5a1", "#777775", "#4d4e4f", "#656260", "#847b7c"],
    "BCA": ["#c6bdcd", "#6f687c", "#948ba6", "#878083", "#a494b2"],
    "BMLT": ["#876389", "#413a58", "#5e4a6b", "#55415b", "#573b64"],
}

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV"""
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    return h * 360, s * 100, v * 100

def calculate_color_distance(color1_rgb, color2_rgb):
    """Calculate Euclidean distance between two RGB colors"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1_rgb, color2_rgb)))

def calculate_hsv_distance(hsv1, hsv2):
    """Calculate weighted HSV distance considering hue wraparound"""
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2
    
    # Handle hue wraparound (circular distance)
    hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
    
    # Weighted distance (hue is most important, then saturation, then value)
    distance = np.sqrt(
        (hue_diff * 2) ** 2 +  # Hue weight: 2
        (abs(s1 - s2) * 1.5) ** 2 +  # Saturation weight: 1.5
        (abs(v1 - v2) * 1) ** 2   # Value weight: 1
    )
    
    return distance