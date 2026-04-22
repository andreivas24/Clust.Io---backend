import numpy as np
from PIL import Image

def analyze_image_complexity(image_file):
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image)

    height, width, _ = image_np.shape
    total_pixels = width * height

    flat_pixels = image_np.reshape(-1, 3)

    # sample for performance if image is too large
    sample_size = min(50000, len(flat_pixels))
    if len(flat_pixels) > sample_size:
        indices = np.random.choice(len(flat_pixels), sample_size, replace=False)
        sampled_pixels = flat_pixels[indices]
    else:
        sampled_pixels = flat_pixels

    estimated_unique_colors = len(np.unique(sampled_pixels, axis=0))
    color_variance = float(np.var(sampled_pixels))

    if estimated_unique_colors < 1000 and color_variance < 800:
        complexity_level = "low"
    elif estimated_unique_colors < 5000 and color_variance < 2500:
        complexity_level = "medium"
    else:
        complexity_level = "high"

    return {
        "width": width,
        "height": height,
        "total_pixels": int(total_pixels),
        "estimated_unique_colors": int(estimated_unique_colors),
        "color_variance": round(color_variance, 4),
        "complexity_level": complexity_level,
    }


def suggest_parameters_from_analysis(analysis):
    width = analysis["width"]
    height = analysis["height"]
    total_pixels = analysis["total_pixels"]
    complexity_level = analysis["complexity_level"]

    # downsampling
    if total_pixels > 1500 * 1500:
        downsample_enabled = True
        downsample_size = 256
    elif total_pixels > 800 * 800:
        downsample_enabled = True
        downsample_size = 256
    elif total_pixels > 400 * 400:
        downsample_enabled = True
        downsample_size = 128
    else:
        downsample_enabled = False
        downsample_size = min(width, height)

    # clusters
    if complexity_level == "low":
        n_clusters = 4
    elif complexity_level == "medium":
        n_clusters = 6
    else:
        n_clusters = 8

    # patch size
    if complexity_level == "low":
        patch_size = 32
    elif complexity_level == "medium":
        patch_size = 16
    else:
        patch_size = 8

    # dbscan params
    if complexity_level == "low":
        dbscan_eps = 15
        dbscan_min_samples = 5
    elif complexity_level == "medium":
        dbscan_eps = 10
        dbscan_min_samples = 5
    else:
        dbscan_eps = 7
        dbscan_min_samples = 5

    # dec epochs
    if complexity_level == "low":
        max_epochs = 20
    elif complexity_level == "medium":
        max_epochs = 30
    else:
        max_epochs = 50

    notes = []

    if complexity_level == "high":
        notes.append("High image complexity detected.")
        notes.append("A smaller patch size may preserve more local detail.")
        notes.append("A higher number of clusters may better separate visual regions.")

    if downsample_enabled:
        notes.append("Downsampling is recommended for large or memory-intensive workloads.")

    if complexity_level == "low":
        notes.append("The image appears visually simple, so fewer clusters may be sufficient.")

    return {
        "n_clusters": n_clusters,
        "downsample_enabled": downsample_enabled,
        "downsample_size": downsample_size,
        "patch_size": patch_size,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "max_epochs": max_epochs,
        "notes": notes,
    }