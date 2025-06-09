from ultralytics import YOLO
from utils.dominant_color import get_dominant_color
import numpy as np
import cv2
from PIL import Image


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

                # Crop shirt region
                cropped_shirt = image[y1:y2, x1:x2]

                if cropped_shirt.size == 0:
                    continue
                
                # Convert to RGB for color extraction
                cropped_rgb = cv2.cvtColor(cropped_shirt, cv2.COLOR_BGR2RGB)
                dominant_color = get_dominant_color(cropped_rgb)
                department = detect_department_from_cropped_image(dominant_color, DEPARTMENT_COLORS)

                overall_confidence = confidence * (0.8 if department else 0.3)
                color_confidence = "high" if department else "low"
                crop_area = (x2 - x1) * (y2 - y1)

                result_dict = {
                    'department': department,
                    'dominant_color': [int(x) for x in dominant_color],
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

# defined DEPARTMENT_COLORS with lower and upper HSV ranges for each department
DEPARTMENT_COLORS = {
    "B.PHARMA_x": {'lower': [100, 0, 149], 'upper': [116, 84, 255]}, # > 0.7
    "B.TECH_x": {'lower': [0, 0, 0], 'upper': [40, 255, 168]}, # > 0.7 if lighting is good else < 0.4
    "BMLT_p": {'lower': [136, 0, 0], 'upper': [179, 255, 218]},
    "BMLT_x": {'lower': [131, 48, 0], 'upper': [147, 87, 255]},
    "BCA_x": {'lower': [87, 19, 156], 'upper': [179, 255, 255]},
}

def detect_department_from_cropped_image(dominant_color, department_colors):
    """
    Detect department based on the dominant HSV color.

    Parameters:
    - dominant_color: Tuple of HSV values (H, S, V).
    - department_colors: Dict of departments with HSV 'lower' and 'upper' bounds.

    Returns:
    - str: Department name if matched, else "No Match".
    """
    h, s, v = dominant_color

    for dept, ranges in department_colors.items():
        lower = ranges['lower']
        upper = ranges['upper']

        if (lower[0] <= h <= upper[0] and
            lower[1] <= s <= upper[1] and
            lower[2] <= v <= upper[2]):
            return dept

    return None



if __name__ == "__main__":
    try:
        test_img = "./tests/images/bmlt.jpg"
        
        # Load the PIL image
        pil_image = Image.open(test_img)
        
        # Convert PIL Image to OpenCV BGR format
        img_array = np.array(pil_image)              # RGB format
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        result = identify_department_from_uniform_colorcode(img_cv)

        if result and result.get('bounding_box'):
            print(f"\nShirt Detection Confidence: {result['shirt_detection_confidence']}")
            print(f"Bounding Box: {result['bounding_box']}")
            print(f"Dominant Color: {result['dominant_color']}")
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
