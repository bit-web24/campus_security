from ultralytics import YOLO
from utils.color_to_department import get_department_from_color
from utils.dominant_color import get_dominant_color
import numpy as np

# Load your custom trained model for shirt detection
model = YOLO("./models/yolov8n.pt")  # Replace with your custom model path

def identify_department_from_uniform_colorcode(image, debug=False):
    """
    Detect shirt in image and return identified departments from the shirt color
    
    Args:
        image: PIL Image object
        debug: Whether to print debug information
        
    Returns:
        dict: Contains department, confidence, bounding_box, and dominant_color
    """
    try:
        # Predict with optimized parameters for shirt detection
        results = model.predict(
            image, 
            conf=0.4,           # Balanced confidence threshold
            iou=0.45,           # Standard IoU threshold
            verbose=False, 
            max_det=10,         # Reasonable max detections for efficiency
            imgsz=640           # Standard YOLO input size
        )
        
        if not results or len(results) == 0:
            if debug:
                print("No detection results returned")
            return None
            
        result = results[0]
        
        # Check if any detections were made
        if result.boxes is None or len(result.boxes) == 0:
            if debug:
                print("No boxes detected")
            return None
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Process all detections and find the best one
        best_detection = None
        best_score = 0
        
        for i, box in enumerate(result.boxes):
            try:
                # Get bounding box coordinates
                x1, y1, x2, y2 = [int(round(x)) for x in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                
                if debug:
                    print(f"Detection {i}: bbox=({x1},{y1},{x2},{y2}), conf={confidence:.3f}")
                
                # Validate and clamp coordinates
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(x1 + 1, min(x2, img_width))
                y2 = max(y1 + 1, min(y2, img_height))
                
                # Calculate box dimensions
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                # Minimum size requirements (adjust based on your needs)
                min_box_size = 30
                min_area = 900  # 30x30 minimum
                
                if box_width < min_box_size or box_height < min_box_size or box_area < min_area:
                    if debug:
                        print(f"  Skipping small detection: {box_width}x{box_height} (area: {box_area})")
                    continue
                
                # Calculate detection quality score
                # Factors: confidence, size (larger is usually better for color detection), aspect ratio
                size_score = min(box_area / (img_width * img_height), 0.5)  # Cap at 50% of image
                aspect_ratio = box_width / box_height
                aspect_score = 1.0 / (1.0 + abs(aspect_ratio - 1.0))  # Prefer square-ish detections
                
                quality_score = confidence * 0.6 + size_score * 0.3 + aspect_score * 0.1
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'area': box_area,
                        'quality_score': quality_score,
                        'index': i
                    }
                    
            except Exception as e:
                if debug:
                    print(f"Error processing detection {i}: {e}")
                continue
        
        if best_detection is None:
            if debug:
                print("No valid detections found")
            return None
        
        # Extract the best detection
        x1, y1, x2, y2 = best_detection['bbox']
        
        if debug:
            print(f"Best detection: bbox=({x1},{y1},{x2},{y2}), "
                  f"conf={best_detection['confidence']:.3f}, "
                  f"quality={best_detection['quality_score']:.3f}")
        
        # Crop the detected shirt region
        cropped = image.crop((x1, y1, x2, y2))
        
        # Optional: Add some padding to the crop for better color detection
        # This can help if the bounding box is too tight
        padding_ratio = 0.05  # 5% padding
        pad_x = int(cropped.width * padding_ratio)
        pad_y = int(cropped.height * padding_ratio)
        
        # Expand crop with padding (if possible)
        padded_x1 = max(0, x1 - pad_x)
        padded_y1 = max(0, y1 - pad_y)
        padded_x2 = min(img_width, x2 + pad_x)
        padded_y2 = min(img_height, y2 + pad_y)
        
        if (padded_x2 - padded_x1) > (x2 - x1) * 1.1:  # Only use padding if it adds significant area
            cropped = image.crop((padded_x1, padded_y1, padded_x2, padded_y2))
            if debug:
                print(f"Using padded crop: ({padded_x1},{padded_y1},{padded_x2},{padded_y2})")
        
        # Extract dominant color from the cropped shirt
        dominant_color = get_dominant_color(cropped)
        
        if debug:
            print(f"Dominant color: {dominant_color}")
        
        # Map color to department
        department = get_department_from_color(dominant_color)
        
        # Calculate overall confidence
        color_confidence = "high" if department else "low"
        overall_confidence = best_detection['confidence'] * (0.8 if department else 0.3)
        
        result_dict = {
            'department': department,
            'dominant_color': dominant_color,
            'detection_confidence': best_detection['confidence'],
            'overall_confidence': overall_confidence,
            'color_confidence': color_confidence,
            'bounding_box': best_detection['bbox'],
            'crop_area': best_detection['area']
        }
        
        if debug:
            print(f"Final result: {result_dict}")
        
        return result_dict
        
    except Exception as e:
        print(f"Error in identify_department_from_uniform_colorcode: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return None

def batch_process_images(image_list, debug=False):
    """
    Process multiple images and return results
    
    Args:
        image_list: List of PIL Image objects
        debug: Whether to print debug information
        
    Returns:
        list: Results for each image
    """
    results = []
    
    for i, image in enumerate(image_list):
        if debug:
            print(f"\n--- Processing image {i+1}/{len(image_list)} ---")
        
        result = identify_department_from_uniform_colorcode(image, debug=debug)
        results.append({
            'image_index': i,
            'result': result
        })
    
    return results

def get_detection_statistics(results):
    """
    Analyze batch processing results and return statistics
    
    Args:
        results: Results from batch_process_images
        
    Returns:
        dict: Statistics about the detection performance
    """
    total_images = len(results)
    successful_detections = sum(1 for r in results if r['result'] is not None)
    department_counts = {}
    
    for r in results:
        if r['result'] and r['result']['department']:
            dept = r['result']['department']
            department_counts[dept] = department_counts.get(dept, 0) + 1
    
    avg_confidence = 0
    if successful_detections > 0:
        confidences = [r['result']['overall_confidence'] for r in results 
                      if r['result'] is not None]
        avg_confidence = sum(confidences) / len(confidences)
    
    return {
        'total_images': total_images,
        'successful_detections': successful_detections,
        'detection_rate': successful_detections / total_images if total_images > 0 else 0,
        'department_distribution': department_counts,
        'average_confidence': avg_confidence
    }

# Example usage
# if __name__ == "__main__":
#     from PIL import Image
    
#     # Test with a single image
#     try:
#         image = Image.open("./tests/images/9.jpg")
#         result = identify_department_from_uniform_colorcode(image, debug=True)
        
#         if result:
#             print(f"\nDetected Department: {result['department']}")
#             print(f"Dominant Color: {result['dominant_color']}")
#             print(f"Confidence: {result['overall_confidence']:.3f}")
#         else:
#             print("No department detected")
            
#     except FileNotFoundError:
#         print("Test image not found - replace 'test_uniform.jpg' with your image path")