import numpy as np
import cv2
from sklearn.cluster import KMeans

def get_dominant_color(pil_image, k=3):
    # Convert PIL image to NumPy array (RGB)
    image = np.array(pil_image)

    # Reshape to a list of pixels
    pixels = image.reshape((-1, 3))

    # Run KMeans to find dominant RGB color
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(pixels)
    dominant_rgb = kmeans.cluster_centers_[np.bincount(kmeans.labels_).argmax()].astype(int)

    # Convert to HSV using OpenCV
    rgb_pixel = np.uint8([[dominant_rgb]])  # shape (1, 1, 3)
    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)

    return tuple(hsv_pixel[0][0])  # Return (H, S, V)
