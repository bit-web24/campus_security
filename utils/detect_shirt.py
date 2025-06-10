import cv2
import sys
import numpy as np
from ultralytics import YOLO
from collections import deque

# Load the YOLO model
model_path = "../runs/content/runs/detect/train/weights/best.pt"
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error: Could not load model. {e}")
    sys.exit(1)

camera_index = 0
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available")
        camera_index = i
        cap.release()
        break

# Initialize video capture from the default webcam
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# Set a lower resolution to reduce camera lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Define the reference color for the B.TECH department shirt (based on previous logs)
reference_color_rgb = np.array([148, 148, 146])

# Variables for frame skipping and temporal consistency
frame_counter = 0
FRAME_SKIP = 5  # Perform color detection every 5 frames
history = deque(maxlen=3)  # Store the last 3 detection results for temporal consistency
last_detected_color = (0, 0, 0)  # Store the last detected color for skipped frames
last_is_btech = False  # Store the last detection result

# Function to calculate Euclidean distance between two RGB colors
def euclidean_distance(color1, color2):
    return np.sqrt(np.sum((color1 - color2) ** 2))

# Function to analyze shirt color using lightweight K-Means clustering
def analyze_shirt_color(frame, x1, y1, x2, y2, is_person_detection=False):
    # If the model detects "person" objects, focus on the upper half of the bounding box (shirt area)
    if is_person_detection:
        shirt_y1 = y1
        shirt_y2 = y1 + (y2 - y1) // 2
        shirt_x1 = x1
        shirt_x2 = x2
    else:
        shirt_y1 = y1
        shirt_y2 = y2
        shirt_x1 = x1
        shirt_x2 = x2
    
    # Ensure the region is valid
    shirt_y1 = max(shirt_y1, 0)
    shirt_y2 = min(shirt_y2, frame.shape[0])
    shirt_x1 = max(shirt_x1, 0)
    shirt_x2 = min(shirt_x2, frame.shape[1])
    
    if shirt_y2 <= shirt_y1 or shirt_x2 <= shirt_x1:
        return False, (0, 0, 0)
    
    # Extract the region of interest (ROI)
    roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    if roi.size == 0:
        return False, (0, 0, 0)
    
    # Downsample the ROI to reduce computation (resize to 32x32 pixels)
    roi_resized = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
    
    # Reshape the ROI into a 2D array of pixels for K-Means
    pixels = roi_resized.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    # Define criteria and apply K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    k = 3
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to integer
    centers = np.uint8(centers)
    
    # Find the dominant cluster (the one with the most pixels)
    labels = labels.flatten()
    dominant_cluster = np.argmax(np.bincount(labels))
    dominant_color = centers[dominant_cluster]
    
    # Convert dominant color to RGB for comparison
    dominant_color_rgb = np.array([dominant_color[2], dominant_color[1], dominant_color[0]])
    
    # Convert dominant color to HSV to check saturation
    dominant_color_bgr = np.uint8([[dominant_color]])
    hsv_color = cv2.cvtColor(dominant_color_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv_color[0][0][1]
    
    # Calculate Euclidean distance between the dominant color and the reference color
    distance = euclidean_distance(dominant_color_rgb, reference_color_rgb)
    
    # Check if the color is "greyish" (low saturation) and close to the reference color
    # Saturation threshold: 20 (about 8% in OpenCV's 0-255 scale)
    # Distance threshold: 12 (stricter for better accuracy)
    is_btech = (distance < 12) and (saturation < 20)
    
    # Return the dominant color in BGR for display
    avg_color = tuple(int(c) for c in dominant_color)
    
    return is_btech, avg_color

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Initialize a default color for the square
        detected_color = last_detected_color
        is_btech = last_is_btech

        # Perform inference using the YOLO model
        results = model(frame)

        # Increment frame counter
        frame_counter += 1

        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = model.names[class_id]

                # Determine if the detection is a person or a shirt
                is_person_detection = class_name.lower() == "person"

                # Skip if the class is neither "shirt" nor "person"
                if not (class_name.lower() in ["shirt", "person"]):
                    continue

                # Perform color detection only every FRAME_SKIP frames
                if frame_counter % FRAME_SKIP == 0:
                    # Analyze the shirt color using lightweight K-Means
                    is_btech_temp, detected_color_temp = analyze_shirt_color(frame, x1, y1, x2, y2, is_person_detection)
                    
                    # Update the last detected values
                    last_detected_color = detected_color_temp
                    detected_color = detected_color_temp
                    
                    # Add the detection result to history for temporal consistency
                    history.append(is_btech_temp)
                    
                    # Check if the shirt is consistently detected as B.TECH over the last few frames
                    if len(history) == history.maxlen:
                        is_btech = all(history)  # True only if all recent detections are B.TECH
                    else:
                        is_btech = False  # Default to False until history is full
                    
                    last_is_btech = is_btech
                    
                    # Print the detected color in RGB format
                    print(f"Detected shirt color (RGB): ({detected_color[2]}, {detected_color[1]}, {detected_color[0]})")
                
                # Draw rectangle (red, thickness=2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Determine the text to display based on shirt color
                text = "B.TECH" if is_btech else "Unknown"
                
                # Draw a background rectangle for better text visibility
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                rect_x1, rect_y1 = x1, y1 - text_height - 5
                rect_x2, rect_y2 = x1 + text_width, y1 - 5
                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)
                cv2.putText(frame, text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Draw a filled square in the bottom-right corner to show the detected color
        square_size = 50
        margin = 10
        frame_height, frame_width = frame.shape[:2]
        square_x1 = frame_width - square_size - margin
        square_y1 = frame_height - square_size - margin
        square_x2 = frame_width - margin
        square_y2 = frame_height - margin
        
        # Ensure detected_color is a valid tuple of integers
        if not isinstance(detected_color, tuple) or len(detected_color) != 3 or not all(isinstance(c, (int, np.integer)) for c in detected_color):
            detected_color = (0, 0, 0)
        
        # Draw the filled square with the detected color
        cv2.rectangle(frame, (square_x1, square_y1), (square_x2, square_y2), detected_color, -1)
        cv2.rectangle(frame, (square_x1, square_y1), (square_x2, square_y2), (0, 0, 0), 1)

        # Display the frame
        cv2.imshow('Shirt Detection', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
