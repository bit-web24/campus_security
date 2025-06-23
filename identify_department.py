from ultralytics import YOLO
import numpy as np
import cv2

# custom trained YOLO model for shirt detection
model = YOLO("runs/content/runs/detect/train/weights/best.pt")

def identify_department_from_uniform_colorcode(image):
    try:
        # Run YOLO inference
        results = model(image)
        departments = []

        # Process the results
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = model.names[class_id].lower()
                confidence = float(box.conf[0])

                # Detect only shirts
                if class_name != "shirt":
                    continue

                # Crop shirt region (BGR format)
                cropped_shirt = image[y1:y2, x1:x2]

                if cropped_shirt.size == 0:
                    continue
                
                crop_area = (x2 - x1) * (y2 - y1)
                image_area = image.shape[0] * image.shape[1]
                
                # Area thresholds to filter out large/small false detections
                min_shirt_area = 0.01 * image_area  # 1%
                max_shirt_area = 0.3 * image_area   # 50%
                
                if crop_area < min_shirt_area or crop_area > max_shirt_area:
                    continue  # Skip invalid shirt sizes
                
                department = get_department(cropped_shirt, DEPARTMENT_COLORS)
                
                overall_confidence = confidence * (0.8 if department else 0.3)
                color_confidence = "high" if department else "low"
                crop_area = (x2 - x1) * (y2 - y1)

                result_dict = {
                    'department': department,
                    # 'dominant_color': [int(x) for x in dominant_color],
                    'shirt_detection_confidence': confidence,
                    'overall_confidence': overall_confidence,
                    'color_confidence': color_confidence,
                    'bounding_box': [x1, y1, x2, y2],
                    'crop_area': crop_area,
                }
                departments.append(result_dict)
        
        if not departments:
            print("No shirts detected in the image.")
            return None
        return departments[0]
        
    except Exception as e:
        print(f"Error in identify_department_from_uniform_colorcode: {e}")
        return None



def find_max_contour(image, hsv_lower, hsv_upper, area_threshold=0.4):
    """
    Detects contours in an image based on HSV range, finds the contour with the maximum area,
    and checks if its area exceeds the specified threshold.

    Parameters:
        image (numpy.ndarray): Input image in BGR format.
        hsv_lower (tuple): Lower bound of HSV values (H, S, V).
        hsv_upper (tuple): Upper bound of HSV values (H, S, V).
        area_threshold (float): Minimum area threshold for the contour.

    Returns:
        tuple: (max_contour, max_area, is_above_threshold)
            - max_contour: Contour with the maximum area (None if no contours found).
            - max_area: Area of the maximum contour (0 if no contours found).
            - is_above_threshold: Boolean indicating if max_area exceeds area_threshold.
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask based on the HSV range
    mask = cv2.inRange(hsv_image, np.array(hsv_lower), np.array(hsv_upper))

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for maximum contour and area
    max_contour = None
    max_area = 0

    # Iterate through contours to find the one with maximum area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Check if the maximum area exceeds the threshold
    is_above_threshold = max_area > area_threshold

    return max_contour, max_area, is_above_threshold


def get_department(image, department_colors):
    """
    Identify the department based on the dominant color of the shirt in the image.

    Parameters:
    - image: Input image in BGR format.
    - department_colors: Dictionary containing department color ranges.

    Returns:
    - str: Detected department name or "No Match".
    """
    # Convert the image to HSV color space
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    department = {}

    # Find contours for each department color
    for dept, ranges in department_colors.items():
        lower = ranges['lower']
        upper = ranges['upper']

        max_contour, max_area, is_above_threshold = find_max_contour(image, lower, upper)

        if is_above_threshold:
            if not department or max_area > department['max_area']:
                department = {
                    'name': dept,
                    'max_contour': max_contour,
                    'max_area': max_area,
                    'hsv_lower': lower,
                    'hsv_upper': upper
                }

    return department['name'] if department else None


# defined DEPARTMENT_COLORS with lower and upper HSV ranges for each department
DEPARTMENT_COLORS = {
    "Unknown": {'lower': [58, 20, 16], 'upper': [144, 210, 96]},
    "B.TECH": {'lower': [0, 0, 0], 'upper': [40, 255, 168]},
    "B.PHARMA": {'lower': [105, 48, 0], 'upper': [179, 255, 255]},
    "BCA": {'lower': [87, 19, 156], 'upper': [179, 255, 255]},
}


# Example usage:
from PIL import Image

if __name__ == "__main__":
    try:
        test_img = "tests/images/btech.jpeg"
        
        # Load the PIL image
        pil_image = Image.open(test_img)
        
        # Convert PIL Image to OpenCV BGR format
        img_array = np.array(pil_image)              # RGB format
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        result = identify_department_from_uniform_colorcode(img_cv)

        if result and result.get('bounding_box'):
            print(f"\nShirt Detection Confidence: {result['shirt_detection_confidence']}")
            print(f"Bounding Box: {result['bounding_box']}")
            # print(f"Dominant Color: {result['dominant_color']}")
            print(f"Color Confidence: {result['color_confidence']}")
            print(f"Overall Confidence: {result['overall_confidence']}")
            print(f"Detected Department: {result['department']}")

            # Convert PIL image to OpenCV format (RGB to BGR)
            img_array = np.array(pil_image)
            if img_array.shape[-1] == 3:  # Ensure it's RGB
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                print("Image format not supported for display")
                exit(1)

            # Extract bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = result['bounding_box']

            # Draw bounding box on the image
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))
            color = (0, 255, 0)  # Green color in BGR
            thickness = 2
            img_cv = cv2.rectangle(img_cv, start_point, end_point, color, thickness)
            
            # Add department text above the bounding box
            department_text = result['department']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5  # Larger scale for big text
            text_color = (0, 0, 255)  # Black color in BGR
            text_thickness = 3  # Increased thickness for bold text
            
            # Calculate text position (slightly above the top-left corner of the bounding box)
            text_position = (int(x1), int(y1) - 10)  # Move 10 pixels above y1
            img_cv = cv2.putText(img_cv, department_text, text_position, font, font_scale, text_color, text_thickness, cv2.LINE_AA)

            # Display the image with bounding box
            cv2.imshow("Image with Bounding Box", img_cv)
            cv2.waitKey(0)  # Wait for key press to close
            cv2.destroyAllWindows()

        else:
            print("No department or bounding box detected")

    except FileNotFoundError:
        print(f"Test image not found - replace '{test_img}' with your image path")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
