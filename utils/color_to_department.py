from utils.colors import DEPARTMENT_COLORS, hex_to_rgb, rgb_to_hsv, calculate_color_distance, calculate_hsv_distance
from utils.dominant_color import get_dominant_color

def build_department_ranges():
    """Build HSV ranges for each department based on actual color samples"""
    department_ranges = {}
    
    for dept, hex_colors in DEPARTMENT_COLORS.items():
        hsv_colors = []
        
        # Convert all hex colors to HSV
        for hex_color in hex_colors:
            rgb = hex_to_rgb(hex_color)
            hsv = rgb_to_hsv(*rgb)
            hsv_colors.append(hsv)
        
        # Calculate ranges with some tolerance
        h_values = [hsv[0] for hsv in hsv_colors]
        s_values = [hsv[1] for hsv in hsv_colors]
        v_values = [hsv[2] for hsv in hsv_colors]
        
        # Add tolerance to ranges (±15 for hue, ±20 for saturation, ±25 for value)
        hue_range = (max(0, min(h_values) - 15), min(360, max(h_values) + 15))
        sat_range = (max(0, min(s_values) - 20), min(100, max(s_values) + 20))
        val_range = (max(0, min(v_values) - 25), min(100, max(v_values) + 25))
        
        department_ranges[dept] = {
            'hue': hue_range,
            'saturation': sat_range,
            'value': val_range,
            'reference_colors': hsv_colors  # Keep reference colors for distance calculation
        }
    
    return department_ranges



def get_department_from_color(color_hex, method='hybrid'):
    """
    Map color to department using multiple methods
    
    Args:
        color_hex: Hex color string
        method: 'range', 'distance', or 'hybrid' (default)
    """
    if not color_hex:
        return None
    
    try:
        color_rgb = hex_to_rgb(color_hex)
        color_hsv = rgb_to_hsv(*color_rgb)
        
        if method == 'range':
            return _get_department_by_range(color_hsv)
        elif method == 'distance':
            return _get_department_by_distance(color_rgb, color_hsv)
        else:  # hybrid
            return _get_department_hybrid(color_rgb, color_hsv)
            
    except Exception as e:
        print(f"Error mapping color to department: {e}")
        return None

def _get_department_by_range(color_hsv):
    """Get department using HSV range matching"""
    department_ranges = build_department_ranges()
    
    h, s, v = color_hsv
    
    for dept, ranges in department_ranges.items():
        h_min, h_max = ranges['hue']
        s_min, s_max = ranges['saturation']
        v_min, v_max = ranges['value']
        
        # Handle hue wraparound
        if h_min > h_max:
            h_match = h >= h_min or h <= h_max
        else:
            h_match = h_min <= h <= h_max
        
        s_match = s_min <= s <= s_max
        v_match = v_min <= v <= v_max
        
        if h_match and s_match and v_match:
            return dept
    
    return None

def _get_department_by_distance(color_rgb, color_hsv):
    """Get department using color distance to reference colors"""
    best_match = None
    min_distance = float('inf')
    
    for dept, hex_colors in DEPARTMENT_COLORS.items():
        dept_distances = []
        
        for hex_color in hex_colors:
            ref_rgb = hex_to_rgb(hex_color)
            ref_hsv = rgb_to_hsv(*ref_rgb)
            
            # Calculate both RGB and HSV distances
            rgb_dist = calculate_color_distance(color_rgb, ref_rgb)
            hsv_dist = calculate_hsv_distance(color_hsv, ref_hsv)
            
            # Combine distances (HSV is more perceptually accurate)
            combined_dist = hsv_dist * 0.7 + rgb_dist * 0.3
            dept_distances.append(combined_dist)
        
        # Use the minimum distance to any color in the department
        min_dept_distance = min(dept_distances)
        
        if min_dept_distance < min_distance:
            min_distance = min_dept_distance
            best_match = dept
    
    # Return match only if distance is reasonable (threshold can be tuned)
    return best_match if min_distance < 100 else None

def _get_department_hybrid(color_rgb, color_hsv):
    """Hybrid approach: try range first, then distance"""
    # First try range-based matching
    range_result = _get_department_by_range(color_hsv)
    if range_result:
        return range_result
    
    # If range matching fails, try distance-based matching
    distance_result = _get_department_by_distance(color_rgb, color_hsv)
    return distance_result

def analyze_image_for_department(image, method='hybrid', debug=False):
    """
    Complete pipeline to analyze image and determine department
    
    Args:
        image: PIL Image object
        method: Detection method ('range', 'distance', 'hybrid')
        debug: Print debug information
    """
    # Extract dominant color
    dominant_color = get_dominant_color(image)
    
    if debug:
        print(f"Dominant color extracted: {dominant_color}")
        rgb = hex_to_rgb(dominant_color)
        hsv = rgb_to_hsv(*rgb)
        print(f"RGB: {rgb}, HSV: {hsv}")
    
    # Determine department
    department = get_department_from_color(dominant_color, method=method)
    
    if debug:
        print(f"Detected department: {department}")
        
        # Show distances to all reference colors
        if department is None:
            print("Distance analysis:")
            color_rgb = hex_to_rgb(dominant_color)
            color_hsv = rgb_to_hsv(*color_rgb)
            
            for dept, hex_colors in DEPARTMENT_COLORS.items():
                distances = []
                for hex_color in hex_colors:
                    ref_rgb = hex_to_rgb(hex_color)
                    ref_hsv = rgb_to_hsv(*ref_rgb)
                    dist = calculate_hsv_distance(color_hsv, ref_hsv)
                    distances.append(dist)
                min_dist = min(distances)
                print(f"  {dept}: min distance = {min_dist:.2f}")
    
    return {
        'department': department,
        'dominant_color': dominant_color,
        'confidence': 'high' if department else 'low'
    }

# Example usage and testing
# if __name__ == "__main__":
#     # Test with known colors
#     test_colors = [
#         "#a4a5a1",  # BTECH
#         "#ffd3ee",  # BCA  
#         "#B87BB7",  # BMLT
#     ]
    
#     print("Testing color detection:")
#     for color in test_colors:
#         dept = get_department_from_color(color, method='hybrid')
#         print(f"Color {color} -> {dept}")
    
#     # Print department color ranges for reference
#     print("\nDepartment color ranges (HSV):")
#     ranges = build_department_ranges()
#     for dept, range_data in ranges.items():
#         print(f"{dept}:")
#         print(f"  Hue: {range_data['hue']}")
#         print(f"  Saturation: {range_data['saturation']}")
#         print(f"  Value: {range_data['value']}")
