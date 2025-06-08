import numpy as np
from sklearn.cluster import KMeans
from utils.colors import rgb_to_hsv

def get_dominant_color(image):
    """Extract dominant color from PIL image using K-means clustering"""
    try:
        # Resize image to reduce computation time
        image = image.resize((150, 150))
        data = np.array(image)
        
        # Handle grayscale images
        if len(data.shape) == 2:
            return '#808080'
        
        # Convert to RGB if RGBA
        if data.shape[2] == 4:
            data = data[:, :, :3]
        
        data = data.reshape((-1, 3))
        
        # Remove invalid pixels and very dark/bright pixels that might be noise
        data = data[~np.any(data < 10, axis=1)]  # Remove very dark pixels
        data = data[~np.any(data > 245, axis=1)]  # Remove very bright pixels
        
        if len(data) == 0:
            return '#808080'
        
        # Use more clusters for better color separation
        n_clusters = min(8, len(data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(data)
        
        # Get cluster centers and their frequencies
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Calculate cluster frequencies
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort clusters by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        # Try the most frequent clusters first, but avoid very gray colors
        for idx in sorted_indices:
            color = cluster_centers[unique_labels[idx]]
            rgb = tuple(map(int, color))
            
            # Skip very gray colors (low saturation)
            hsv = rgb_to_hsv(*rgb)
            if hsv[1] > 10:  # Minimum saturation threshold
                return '#{:02x}{:02x}{:02x}'.format(*rgb)
        
        # If all colors are gray, return the most frequent one
        dominant = cluster_centers[unique_labels[sorted_indices[0]]]
        dominant = tuple(map(int, dominant))
        return '#{:02x}{:02x}{:02x}'.format(*dominant)
        
    except Exception as e:
        print(f"Error extracting dominant color: {e}")
        return '#808080'