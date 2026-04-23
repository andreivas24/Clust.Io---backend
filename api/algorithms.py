import base64
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances


def load_image_as_array(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


def downsample_image(image: Image.Image, target_size: int):
    image = image.copy()
    image.thumbnail((target_size, target_size))
    return image


def image_to_pixels(image: Image.Image):
    image_np = np.array(image)
    height, width, channels = image_np.shape
    pixels = image_np.reshape((-1, 3))
    return pixels, (width, height)


def pixels_to_image(clustered_pixels: np.ndarray, original_size):
    width, height = original_size
    image_array = clustered_pixels.reshape((height, width, 3))
    return Image.fromarray(image_array)


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def labels_to_segmentation_image(labels: np.ndarray, original_size):
    width, height = original_size

    palette = np.array([
        [66, 135, 245],    # blue
        [147, 51, 234],    # purple
        [34, 197, 94],     # green
        [249, 115, 22],    # orange
        [236, 72, 153],    # pink
        [14, 165, 233],    # cyan
        [234, 179, 8],     # yellow
        [99, 102, 241],    # indigo
        [168, 85, 247],    # violet
        [20, 184, 166],    # teal
        [244, 63, 94],     # rose
        [132, 204, 22],    # lime
    ], dtype=np.uint8)

    mapped_colors = []
    for label in labels:
        if label == -1:
            mapped_colors.append([0, 0, 0])  # noise -> black
        else:
            mapped_colors.append(palette[label % len(palette)])

    image_array = np.array(mapped_colors, dtype=np.uint8).reshape((height, width, 3))
    return Image.fromarray(image_array)


def compute_dunn_index(pixels: np.ndarray, labels: np.ndarray):
    """
    Dunn Index = min inter-cluster distance / max intra-cluster distance

    Higher is better.
    """
    unique_labels = [label for label in np.unique(labels) if label != -1]

    if len(unique_labels) < 2:
        return None

    clusters = [pixels[labels == label] for label in unique_labels]

    # max intra-cluster distance
    max_intra = 0.0
    for cluster in clusters:
        if len(cluster) < 2:
            continue

        intra_distances = pairwise_distances(cluster)
        cluster_max = np.max(intra_distances)

        if cluster_max > max_intra:
            max_intra = cluster_max

    if max_intra == 0:
        return None

    # min inter-cluster distance
    min_inter = float("inf")
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            inter_distances = pairwise_distances(clusters[i], clusters[j])
            cluster_min = np.min(inter_distances)

            if cluster_min < min_inter:
                min_inter = cluster_min

    if min_inter == float("inf"):
        return None

    return round(float(min_inter / max_intra), 4)

def compute_clustering_metrics(pixels: np.ndarray, labels: np.ndarray, model=None):
    metrics = {}

    unique_labels = np.unique(labels)
    valid_labels = [label for label in unique_labels if label != -1]

    if len(valid_labels) > 1:
        try:
            sample_size = min(5000, len(pixels))
            if len(pixels) > sample_size:
                indices = np.random.choice(len(pixels), sample_size, replace=False)
                sampled_pixels = pixels[indices]
                sampled_labels = labels[indices]
            else:
                sampled_pixels = pixels
                sampled_labels = labels

            metrics["silhouette_score"] = round(
                float(silhouette_score(sampled_pixels, sampled_labels)), 4
            )
        except Exception:
            metrics["silhouette_score"] = None

        try:
            metrics["davies_bouldin_score"] = round(
                float(davies_bouldin_score(pixels, labels)), 4
            )
        except Exception:
            metrics["davies_bouldin_score"] = None

        try:
            metrics["calinski_harabasz_score"] = round(
                float(calinski_harabasz_score(pixels, labels)), 4
            )
        except Exception:
            metrics["calinski_harabasz_score"] = None

        try:
            dunn_sample_size = min(2000, len(pixels))
            if len(pixels) > dunn_sample_size:
                dunn_indices = np.random.choice(len(pixels), dunn_sample_size, replace=False)
                dunn_pixels = pixels[dunn_indices]
                dunn_labels = labels[dunn_indices]
            else:
                dunn_pixels = pixels
                dunn_labels = labels

            metrics["dunn_index"] = compute_dunn_index(dunn_pixels, dunn_labels)
        except Exception:
            metrics["dunn_index"] = None

        try:
            metrics["davies_bouldin_score"] = round(
                float(davies_bouldin_score(pixels, labels)), 4
            )
        except Exception:
            metrics["davies_bouldin_score"] = None
    else:
        metrics["silhouette_score"] = None
        metrics["davies_bouldin_score"] = None
        metrics["calinski_harabasz_score"] = None
        metrics["dunn_index"] = None

    if model is not None and hasattr(model, "inertia_"):
        metrics["inertia"] = round(float(model.inertia_), 4)
    else:
        metrics["inertia"] = None

    return metrics


def build_cluster_stats(labels, centers):
    unique_labels, counts = np.unique(labels, return_counts=True)

    cluster_distribution = []
    for label, count in zip(unique_labels, counts):
        cluster_distribution.append({
            "cluster_index": int(label),
            "pixel_count": int(count),
        })

    cluster_centers = []
    for idx, center in enumerate(centers):
        cluster_centers.append({
            "cluster_index": int(idx),
            "rgb": [int(center[0]), int(center[1]), int(center[2])]
        })

    return cluster_distribution, cluster_centers


def run_kmeans(pixels: np.ndarray, n_clusters: int = 5, max_iter: int = 300, init: str = "k-means++"):
    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        init=init,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(pixels)
    centers = np.clip(model.cluster_centers_, 0, 255).astype(np.uint8)
    clustered_pixels = centers[labels]

    cluster_distribution, cluster_centers = build_cluster_stats(labels, centers)

    return clustered_pixels, cluster_distribution, cluster_centers, labels, model


def run_mini_batch_kmeans(
    pixels: np.ndarray,
    n_clusters: int = 5,
    batch_size: int = 256,
    max_iter: int = 100
):
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(pixels)
    centers = np.clip(model.cluster_centers_, 0, 255).astype(np.uint8)
    clustered_pixels = centers[labels]

    cluster_distribution, cluster_centers = build_cluster_stats(labels, centers)

    return clustered_pixels, cluster_distribution, cluster_centers, labels, model

def run_gmm(
    pixels: np.ndarray,
    n_components: int = 5,
    covariance_type: str = "full",
    max_iter: int = 100
):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=42,
    )

    labels = model.fit_predict(pixels)
    centers = np.clip(model.means_, 0, 255).astype(np.uint8)
    clustered_pixels = centers[labels]

    cluster_distribution, cluster_centers = build_cluster_stats(labels, centers)

    return clustered_pixels, cluster_distribution, cluster_centers, labels, model

def run_agglomerative(
    pixels: np.ndarray,
    n_clusters: int = 5,
    linkage: str = "ward",
    metric: str = "euclidean"
):
    # sklearn rule: ward linkage only supports euclidean
    if linkage == "ward":
        metric = "euclidean"

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric
    )

    labels = model.fit_predict(pixels)

    centers = []
    for i in range(n_clusters):
        cluster_pixels = pixels[labels == i]
        if len(cluster_pixels) == 0:
            centers.append(np.array([0, 0, 0]))
        else:
            centers.append(cluster_pixels.mean(axis=0))

    centers = np.clip(np.array(centers), 0, 255).astype(np.uint8)
    clustered_pixels = centers[labels]

    cluster_distribution, cluster_centers = build_cluster_stats(labels, centers)

    return clustered_pixels, cluster_distribution, cluster_centers, labels, model

def run_birch(
    pixels: np.ndarray,
    threshold: float = 0.5,
    branching_factor: int = 50,
    n_clusters: int = 5
):
    model = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters
    )

    labels = model.fit_predict(pixels)

    centers = []
    for i in range(n_clusters):
        cluster_pixels = pixels[labels == i]
        if len(cluster_pixels) == 0:
            centers.append(np.array([0, 0, 0]))
        else:
            centers.append(cluster_pixels.mean(axis=0))

    centers = np.clip(np.array(centers), 0, 255).astype(np.uint8)
    clustered_pixels = centers[labels]

    cluster_distribution, cluster_centers = build_cluster_stats(labels, centers)

    return clustered_pixels, cluster_distribution, cluster_centers, labels, model

def run_dbscan(
    pixels: np.ndarray,
    eps: float = 5.0,
    min_samples: int = 5,
    metric: str = "euclidean"
):
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric
    )

    labels = model.fit_predict(pixels)

    unique_labels = np.unique(labels)
    valid_labels = [label for label in unique_labels if label != -1]

    centers = []
    label_to_center_index = {}

    for idx, label in enumerate(valid_labels):
        cluster_pixels = pixels[labels == label]
        if len(cluster_pixels) == 0:
            center = np.array([0, 0, 0])
        else:
            center = cluster_pixels.mean(axis=0)

        centers.append(center)
        label_to_center_index[label] = idx

    centers = np.clip(np.array(centers), 0, 255).astype(np.uint8) if len(centers) > 0 else np.array([[0, 0, 0]], dtype=np.uint8)

    clustered_pixels = np.zeros_like(pixels, dtype=np.uint8)

    for i, label in enumerate(labels):
        if label == -1:
            clustered_pixels[i] = np.array([0, 0, 0], dtype=np.uint8)  # noise = black
        else:
            clustered_pixels[i] = centers[label_to_center_index[label]]

    cluster_distribution = []
    for label in unique_labels:
        count = int(np.sum(labels == label))
        cluster_distribution.append({
            "cluster_index": int(label),
            "pixel_count": count,
        })

    cluster_centers = []
    for original_label in unique_labels:
        if original_label == -1:
            cluster_centers.append({
                "cluster_index": -1,
                "rgb": [0, 0, 0]
            })
        else:
            mapped_idx = label_to_center_index[original_label]
            center = centers[mapped_idx]
            cluster_centers.append({
                "cluster_index": int(original_label),
                "rgb": [int(center[0]), int(center[1]), int(center[2])]
            })

    return clustered_pixels, cluster_distribution, cluster_centers, labels, model

def run_bgmm(
    pixels: np.ndarray,
    n_components: int = 10,
    covariance_type: str = "diag",
    weight_concentration_prior_type: str = "dirichlet_process",
    max_iter: int = 200
):
    model = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type=weight_concentration_prior_type,
        max_iter=max_iter,
        random_state=42,
    )

    labels = model.fit_predict(pixels)
    centers = np.clip(model.means_, 0, 255).astype(np.uint8)
    clustered_pixels = centers[labels]

    unique_labels, counts = np.unique(labels, return_counts=True)

    cluster_distribution = []
    for label, count in zip(unique_labels, counts):
        cluster_distribution.append({
            "cluster_index": int(label),
            "pixel_count": int(count),
        })

    cluster_centers = []
    for idx, center in enumerate(centers):
        cluster_centers.append({
            "cluster_index": int(idx),
            "rgb": [int(center[0]), int(center[1]), int(center[2])]
        })

    active_components = int(np.sum(model.weights_ > 0.01))

    return clustered_pixels, cluster_distribution, cluster_centers, labels, model, active_components

def process_kmeans(
    image_file,
    n_clusters=5,
    max_iter=300,
    init="k-means++",
    downsample_enabled=False,
    downsample_size=256
):
    start_time = time.time()

    image = load_image_as_array(image_file)
    original_width, original_height = image.size

    if downsample_enabled:
        image = downsample_image(image, downsample_size)

    pixels, original_size = image_to_pixels(image)

    clustered_pixels, cluster_distribution, cluster_centers, labels, model = run_kmeans(
        pixels=pixels,
        n_clusters=n_clusters,
        max_iter=max_iter,
        init=init,
    )

    metrics = compute_clustering_metrics(pixels, labels, model)

    pixel_scatter_sample = sample_pixel_space(pixels, labels)

    clustered_image = pixels_to_image(clustered_pixels, original_size)
    clustered_image_b64 = image_to_base64(clustered_image)

    segmentation_image = labels_to_segmentation_image(labels, original_size)
    segmentation_image_b64 = image_to_base64(segmentation_image)

    processing_time = round(time.time() - start_time, 4)
    total_pixels = int(len(pixels))

    return {
        "clustered_image": clustered_image_b64,
        "segmentation_map_image": segmentation_image_b64,
        "processing_time": processing_time,
        "width": original_size[0],
        "height": original_size[1],
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": original_size[0],
        "processed_height": original_size[1],
        "total_pixels": total_pixels,
        "cluster_distribution": cluster_distribution,
        "cluster_centers": cluster_centers,
        "parameters_used": {
            "n_clusters": n_clusters,
            "max_iter": max_iter,
            "init": init,
        },
        "metrics": metrics,
        "pixel_scatter_sample": pixel_scatter_sample,
    }


def process_mini_batch_kmeans(
    image_file,
    n_clusters=5,
    batch_size=256,
    max_iter=100,
    downsample_enabled=False,
    downsample_size=256
):
    start_time = time.time()

    image = load_image_as_array(image_file)
    original_width, original_height = image.size

    if downsample_enabled:
        image = downsample_image(image, downsample_size)

    pixels, original_size = image_to_pixels(image)

    clustered_pixels, cluster_distribution, cluster_centers, labels, model = run_mini_batch_kmeans(
        pixels=pixels,
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
    )

    metrics = compute_clustering_metrics(pixels, labels, model)

    pixel_scatter_sample = sample_pixel_space(pixels, labels)

    clustered_image = pixels_to_image(clustered_pixels, original_size)
    clustered_image_b64 = image_to_base64(clustered_image)

    segmentation_image = labels_to_segmentation_image(labels, original_size)
    segmentation_image_b64 = image_to_base64(segmentation_image)

    processing_time = round(time.time() - start_time, 4)
    total_pixels = int(len(pixels))

    return {
        "clustered_image": clustered_image_b64,
        "segmentation_map_image": segmentation_image_b64,
        "processing_time": processing_time,
        "width": original_size[0],
        "height": original_size[1],
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": original_size[0],
        "processed_height": original_size[1],
        "total_pixels": total_pixels,
        "cluster_distribution": cluster_distribution,
        "cluster_centers": cluster_centers,
        "parameters_used": {
            "n_clusters": n_clusters,
            "batch_size": batch_size,
            "max_iter": max_iter,
        },
        "metrics": metrics,
        "pixel_scatter_sample": pixel_scatter_sample,
    }

def process_gmm(
    image_file,
    n_components=5,
    covariance_type="full",
    max_iter=100,
    downsample_enabled=False,
    downsample_size=256
):
    start_time = time.time()

    image = load_image_as_array(image_file)
    original_width, original_height = image.size

    if downsample_enabled:
        image = downsample_image(image, downsample_size)

    pixels, original_size = image_to_pixels(image)

    clustered_pixels, cluster_distribution, cluster_centers, labels, model = run_gmm(
        pixels=pixels,
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
    )

    metrics = compute_clustering_metrics(pixels, labels, model)
    pixel_scatter_sample = sample_pixel_space(pixels, labels)

    clustered_image = pixels_to_image(clustered_pixels, original_size)
    clustered_image_b64 = image_to_base64(clustered_image)

    segmentation_image = labels_to_segmentation_image(labels, original_size)
    segmentation_image_b64 = image_to_base64(segmentation_image)

    processing_time = round(time.time() - start_time, 4)
    total_pixels = int(len(pixels))

    return {
        "clustered_image": clustered_image_b64,
        "segmentation_map_image": segmentation_image_b64,
        "processing_time": processing_time,
        "width": original_size[0],
        "height": original_size[1],
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": original_size[0],
        "processed_height": original_size[1],
        "total_pixels": total_pixels,
        "cluster_distribution": cluster_distribution,
        "cluster_centers": cluster_centers,
        "parameters_used": {
            "n_components": n_components,
            "covariance_type": covariance_type,
            "max_iter": max_iter,
        },
        "metrics": metrics,
        "pixel_scatter_sample": pixel_scatter_sample,
    }

def process_agglomerative(
    image_file,
    n_clusters=5,
    linkage="ward",
    metric="euclidean",
    downsample_enabled=False,
    downsample_size=256
):
    start_time = time.time()

    image = load_image_as_array(image_file)
    original_width, original_height = image.size

    if downsample_enabled:
        image = downsample_image(image, downsample_size)

    pixels, original_size = image_to_pixels(image)

    clustered_pixels, cluster_distribution, cluster_centers, labels, model = run_agglomerative(
        pixels=pixels,
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
    )

    metrics = compute_clustering_metrics(pixels, labels, model)
    pixel_scatter_sample = sample_pixel_space(pixels, labels)

    clustered_image = pixels_to_image(clustered_pixels, original_size)
    clustered_image_b64 = image_to_base64(clustered_image)

    segmentation_image = labels_to_segmentation_image(labels, original_size)
    segmentation_image_b64 = image_to_base64(segmentation_image)

    processing_time = round(time.time() - start_time, 4)
    total_pixels = int(len(pixels))

    return {
        "clustered_image": clustered_image_b64,
        "segmentation_map_image": segmentation_image_b64,
        "processing_time": processing_time,
        "width": original_size[0],
        "height": original_size[1],
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": original_size[0],
        "processed_height": original_size[1],
        "total_pixels": total_pixels,
        "cluster_distribution": cluster_distribution,
        "cluster_centers": cluster_centers,
        "parameters_used": {
            "n_clusters": n_clusters,
            "linkage": linkage,
            "metric": metric,
        },
        "metrics": metrics,
        "pixel_scatter_sample": pixel_scatter_sample,
    }

def process_birch(
    image_file,
    threshold=0.5,
    branching_factor=50,
    n_clusters=5,
    downsample_enabled=False,
    downsample_size=256
):
    start_time = time.time()

    image = load_image_as_array(image_file)
    original_width, original_height = image.size

    if downsample_enabled:
        image = downsample_image(image, downsample_size)

    pixels, original_size = image_to_pixels(image)

    clustered_pixels, cluster_distribution, cluster_centers, labels, model = run_birch(
        pixels=pixels,
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters,
    )

    metrics = compute_clustering_metrics(pixels, labels, model)
    pixel_scatter_sample = sample_pixel_space(pixels, labels)

    clustered_image = pixels_to_image(clustered_pixels, original_size)
    clustered_image_b64 = image_to_base64(clustered_image)

    segmentation_image = labels_to_segmentation_image(labels, original_size)
    segmentation_image_b64 = image_to_base64(segmentation_image)

    processing_time = round(time.time() - start_time, 4)
    total_pixels = int(len(pixels))

    return {
        "clustered_image": clustered_image_b64,
        "segmentation_map_image": segmentation_image_b64,
        "processing_time": processing_time,
        "width": original_size[0],
        "height": original_size[1],
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": original_size[0],
        "processed_height": original_size[1],
        "total_pixels": total_pixels,
        "cluster_distribution": cluster_distribution,
        "cluster_centers": cluster_centers,
        "parameters_used": {
            "threshold": threshold,
            "branching_factor": branching_factor,
            "n_clusters": n_clusters,
        },
        "metrics": metrics,
        "pixel_scatter_sample": pixel_scatter_sample,
    }

def process_dbscan(
    image_file,
    eps=10.0,
    min_samples=5,
    metric="euclidean",
    downsample_enabled=False,
    downsample_size=256
):
    start_time = time.time()

    image = load_image_as_array(image_file)
    original_width, original_height = image.size
    safe_size = 128

    if downsample_enabled:
        image = downsample_image(image, min(int(downsample_size), safe_size))
    else:
        image = downsample_image(image, safe_size)

    pixels, original_size = image_to_pixels(image)

    clustered_pixels, cluster_distribution, cluster_centers, labels, model = run_dbscan(
        pixels=pixels,
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )

    metrics = compute_clustering_metrics(pixels, labels, model)
    pixel_scatter_sample = sample_pixel_space(pixels, labels)
    dbscan_analysis = analyze_dbscan_result(labels)

    clustered_image = pixels_to_image(clustered_pixels, original_size)
    clustered_image_b64 = image_to_base64(clustered_image)

    segmentation_image = labels_to_segmentation_image(labels, original_size)
    segmentation_image_b64 = image_to_base64(segmentation_image)

    processing_time = round(time.time() - start_time, 4)
    total_pixels = int(len(pixels))

    return {
        "clustered_image": clustered_image_b64,
        "segmentation_map_image": segmentation_image_b64,
        "processing_time": processing_time,
        "width": original_size[0],
        "height": original_size[1],
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": original_size[0],
        "processed_height": original_size[1],
        "total_pixels": total_pixels,
        "cluster_distribution": cluster_distribution,
        "cluster_centers": cluster_centers,
        "parameters_used": {
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
        },
        "metrics": metrics,
        "pixel_scatter_sample": pixel_scatter_sample,
        "dbscan_analysis": dbscan_analysis,
    }

def process_bgmm(
    image_file,
    n_components=10,
    covariance_type="diag",
    weight_concentration_prior_type="dirichlet_process",
    max_iter=200,
    downsample_enabled=False,
    downsample_size=256
):
    start_time = time.time()

    image = load_image_as_array(image_file)
    original_width, original_height = image.size

    if downsample_enabled:
        image = downsample_image(image, downsample_size)

    pixels, processed_size = image_to_pixels(image)

    clustered_pixels, cluster_distribution, cluster_centers, labels, model, active_components = run_bgmm(
        pixels=pixels,
        n_components=n_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type=weight_concentration_prior_type,
        max_iter=max_iter,
    )

    metrics = compute_clustering_metrics(pixels, labels, model)

    clustered_image = pixels_to_image(clustered_pixels, processed_size)
    clustered_image_b64 = image_to_base64(clustered_image)

    segmentation_map_image = labels_to_segmentation_image(labels, processed_size)
    segmentation_map_image_b64 = image_to_base64(segmentation_map_image)

    # scatter sample pentru charts / 3D preview
    sample_size = min(2000, len(pixels))
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sampled_pixels = pixels[indices]
        sampled_labels = labels[indices]
    else:
        sampled_pixels = pixels
        sampled_labels = labels

    pixel_scatter_sample = []
    for pixel, label in zip(sampled_pixels, sampled_labels):
        pixel_scatter_sample.append({
            "r": int(pixel[0]),
            "g": int(pixel[1]),
            "b": int(pixel[2]),
            "label": int(label),
        })

    processing_time = round(time.time() - start_time, 4)
    total_pixels = int(len(pixels))

    return {
        "clustered_image": clustered_image_b64,
        "segmentation_map_image": segmentation_map_image_b64,
        "processing_time": processing_time,
        "width": processed_size[0],
        "height": processed_size[1],
        "total_pixels": total_pixels,
        "cluster_distribution": cluster_distribution,
        "cluster_centers": cluster_centers,
        "parameters_used": {
            "n_components": n_components,
            "covariance_type": covariance_type,
            "weight_concentration_prior_type": weight_concentration_prior_type,
            "max_iter": max_iter,
        },
        "metrics": metrics,
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": processed_size[0],
        "processed_height": processed_size[1],
        "effective_components": active_components,
        "pixel_scatter_sample": pixel_scatter_sample,
    }

def analyze_dbscan_result(labels: np.ndarray):
    unique_labels = np.unique(labels)

    noise_count = int(np.sum(labels == -1))
    total_count = int(len(labels))
    noise_ratio = noise_count / total_count if total_count > 0 else 0

    valid_labels = [label for label in unique_labels if label != -1]
    n_clusters = len(valid_labels)

    recommendations = []
    summary = {
        "noise_count": noise_count,
        "noise_ratio": round(noise_ratio, 4),
        "n_clusters": n_clusters,
        "has_noise": noise_count > 0,
        "collapsed_to_one_cluster": False,
        "too_much_noise": False,
        "recommendations": [],
    }

    # Case 1: all or almost all points collapsed into one cluster
    if n_clusters == 1 and noise_ratio < 0.1:
        summary["collapsed_to_one_cluster"] = True
        recommendations.append("All pixels collapsed into one dominant cluster.")
        recommendations.append("Recommended: decrease eps to create more separation between clusters.")

    # Case 2: too much noise
    if noise_ratio > 0.5:
        summary["too_much_noise"] = True
        recommendations.append("Too much noise detected in the image.")
        recommendations.append("Recommended: increase eps or decrease min_samples.")

    # Case 3: no clusters found
    if n_clusters == 0:
        recommendations.append("No dense clusters were found.")
        recommendations.append("Recommended: increase eps or reduce min_samples.")

    # Case 4: too many tiny clusters
    if n_clusters > 20:
        recommendations.append("A very large number of small clusters was detected.")
        recommendations.append("Recommended: increase eps to merge nearby dense regions.")

    if not recommendations:
        recommendations.append("DBSCAN found a reasonable density-based clustering for the current parameters.")

    summary["recommendations"] = recommendations
    return summary

def process_resnet_kmeans(
    image_file,
    n_clusters=5,
    patch_size=16,
    downsample_enabled=False,
    downsample_size=256,
    backbone_model="resnet18",
    feature_layer="avgpool"
):
    start_time = time.time()

    image = load_image_as_array(image_file)
    original_width, original_height = image.size

    # Auto-protection for high-resolution images
    max_side = max(image.size)
    auto_downsample_applied = False

    if downsample_enabled:
        image = downsample_image(image, downsample_size)
    elif max_side > 768:
        image = downsample_image(image, 256)
        auto_downsample_applied = True

    processed_width, processed_height = image.size

    patches, patch_positions, image_width, image_height = image_to_patches(
        image, patch_size=patch_size
    )

    if len(patches) == 0:
        raise ValueError("No valid patches could be extracted from the image.")

    # Limit patches for CPU safety
    max_patches = 1500
    if len(patches) > max_patches:
        selected_indices = np.linspace(0, len(patches) - 1, max_patches, dtype=int)
        patches = [patches[i] for i in selected_indices]
        patch_positions = [patch_positions[i] for i in selected_indices]

    resnet_input_size = 160
    feature_batch_size = 32

    features = extract_resnet_features_from_patches(
        patches=patches,
        backbone_model=backbone_model,
        batch_size=feature_batch_size,
        input_size=resnet_input_size,
    )

    feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
    feature_norms[feature_norms == 0] = 1.0
    features = features / feature_norms

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(features)

    patch_arrays = np.array(
        [np.array(patch).mean(axis=(0, 1)) for patch in patches],
        dtype=np.float32
    )

    centers_rgb = []
    for cluster_id in range(n_clusters):
        cluster_patch_colors = patch_arrays[labels == cluster_id]
        if len(cluster_patch_colors) == 0:
            centers_rgb.append(np.array([0, 0, 0], dtype=np.float32))
        else:
            centers_rgb.append(cluster_patch_colors.mean(axis=0))

    centers_rgb = np.clip(np.array(centers_rgb), 0, 255).astype(np.uint8)

    clustered_image = reconstruct_patch_clustered_image(
        patch_positions=patch_positions,
        labels=labels,
        centers_rgb=centers_rgb,
        image_width=image_width,
        image_height=image_height,
        patch_size=patch_size,
    )

    clustered_image_b64 = image_to_base64(clustered_image)

    seg_w = image_width // patch_size
    seg_h = image_height // patch_size

    segmentation_image = labels_to_segmentation_image(labels, (seg_w, seg_h))
    segmentation_image = segmentation_image.resize((image_width, image_height), Image.NEAREST)
    segmentation_image_b64 = image_to_base64(segmentation_image)

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_distribution = [
        {
            "cluster_index": int(label),
            "pixel_count": int(count),
        }
        for label, count in zip(unique_labels, counts)
    ]

    cluster_centers = [
        {
            "cluster_index": int(idx),
            "rgb": [int(center[0]), int(center[1]), int(center[2])]
        }
        for idx, center in enumerate(centers_rgb)
    ]

    metrics = {
        "inertia": round(float(kmeans.inertia_), 4),
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
        "dunn_index": None,
    }

    try:
        if len(np.unique(labels)) > 1:
            sample_size = min(2000, len(features))
            if len(features) > sample_size:
                indices = np.random.choice(len(features), sample_size, replace=False)
                sampled_features = features[indices]
                sampled_labels = labels[indices]
            else:
                sampled_features = features
                sampled_labels = labels

            metrics["silhouette_score"] = round(
                float(silhouette_score(sampled_features, sampled_labels)), 4
            )
            metrics["davies_bouldin_score"] = round(
                float(davies_bouldin_score(features, labels)), 4
            )
            metrics["calinski_harabasz_score"] = round(
                float(calinski_harabasz_score(features, labels)), 4
            )

            dunn_sample_size = min(1000, len(features))
            if len(features) > dunn_sample_size:
                dunn_indices = np.random.choice(len(features), dunn_sample_size, replace=False)
                dunn_features = features[dunn_indices]
                dunn_labels = labels[dunn_indices]
            else:
                dunn_features = features
                dunn_labels = labels

            metrics["dunn_index"] = compute_dunn_index(dunn_features, dunn_labels)

    except Exception:
        pass

    pixel_scatter_sample = []
    scatter_limit = min(2000, len(patch_arrays))
    for rgb, label in zip(patch_arrays[:scatter_limit], labels[:scatter_limit]):
        pixel_scatter_sample.append({
            "r": int(rgb[0]),
            "g": int(rgb[1]),
            "b": int(rgb[2]),
            "label": int(label),
        })

    processing_time = round(time.time() - start_time, 4)

    return {
        "clustered_image": clustered_image_b64,
        "segmentation_map_image": segmentation_image_b64,
        "processing_time": processing_time,
        "width": image_width,
        "height": image_height,
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "total_pixels": len(patches),
        "cluster_distribution": cluster_distribution,
        "cluster_centers": cluster_centers,
        "parameters_used": {
            "backbone_model": backbone_model,
            "feature_layer": feature_layer,
            "patch_size": patch_size,
            "n_clusters": n_clusters,
            "feature_batch_size": feature_batch_size,
            "resnet_input_size": resnet_input_size,
            "auto_downsample_applied": auto_downsample_applied,
            "max_patches_limit": max_patches,
        },
        "metrics": metrics,
        "pixel_scatter_sample": pixel_scatter_sample,
    }

def process_resnet_gmm(
    image_file,
    n_components=5,
    covariance_type="diag",
    patch_size=16,
    downsample_enabled=False,
    downsample_size=256,
    backbone_model="resnet18",
    feature_layer="avgpool"
):
    start_time = time.time()

    image = load_image_as_array(image_file)
    original_width, original_height = image.size

    max_side = max(image.size)
    auto_downsample_applied = False

    if downsample_enabled:
        image = downsample_image(image, downsample_size)
    elif max_side > 768:
        image = downsample_image(image, 256)
        auto_downsample_applied = True

    processed_width, processed_height = image.size

    patches, patch_positions, image_width, image_height = image_to_patches(
        image, patch_size=patch_size
    )

    if len(patches) == 0:
        raise ValueError("No valid patches could be extracted from the image.")

    max_patches = 1500
    if len(patches) > max_patches:
        selected_indices = np.linspace(0, len(patches) - 1, max_patches, dtype=int)
        patches = [patches[i] for i in selected_indices]
        patch_positions = [patch_positions[i] for i in selected_indices]

    resnet_input_size = 160
    feature_batch_size = 32

    features = extract_resnet_features_from_patches(
        patches=patches,
        backbone_model=backbone_model,
        batch_size=feature_batch_size,
        input_size=resnet_input_size,
    )

    feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
    feature_norms[feature_norms == 0] = 1.0
    features = features / feature_norms

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        reg_covar=1e-5,
    )

    labels = gmm.fit_predict(features)

    patch_arrays = np.array(
        [np.array(patch).mean(axis=(0, 1)) for patch in patches],
        dtype=np.float32
    )

    centers_rgb = []
    for cluster_id in range(n_components):
        cluster_patch_colors = patch_arrays[labels == cluster_id]
        if len(cluster_patch_colors) == 0:
            centers_rgb.append(np.array([0, 0, 0], dtype=np.float32))
        else:
            centers_rgb.append(cluster_patch_colors.mean(axis=0))

    centers_rgb = np.clip(np.array(centers_rgb), 0, 255).astype(np.uint8)

    clustered_image = reconstruct_patch_clustered_image(
        patch_positions=patch_positions,
        labels=labels,
        centers_rgb=centers_rgb,
        image_width=image_width,
        image_height=image_height,
        patch_size=patch_size,
    )

    clustered_image_b64 = image_to_base64(clustered_image)

    seg_w = image_width // patch_size
    seg_h = image_height // patch_size

    segmentation_image = labels_to_segmentation_image(labels, (seg_w, seg_h))
    segmentation_image = segmentation_image.resize((image_width, image_height), Image.NEAREST)
    segmentation_image_b64 = image_to_base64(segmentation_image)

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_distribution = [
        {
            "cluster_index": int(label),
            "pixel_count": int(count),
        }
        for label, count in zip(unique_labels, counts)
    ]

    cluster_centers = [
        {
            "cluster_index": int(idx),
            "rgb": [int(center[0]), int(center[1]), int(center[2])]
        }
        for idx, center in enumerate(centers_rgb)
    ]

    metrics = {
        "inertia": None,
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
        "dunn_index": None,
    }

    try:
        if len(np.unique(labels)) > 1:
            sample_size = min(2000, len(features))
            if len(features) > sample_size:
                indices = np.random.choice(len(features), sample_size, replace=False)
                sampled_features = features[indices]
                sampled_labels = labels[indices]
            else:
                sampled_features = features
                sampled_labels = labels

            metrics["silhouette_score"] = round(
                float(silhouette_score(sampled_features, sampled_labels)), 4
            )
            metrics["davies_bouldin_score"] = round(
                float(davies_bouldin_score(features, labels)), 4
            )
            metrics["calinski_harabasz_score"] = round(
                float(calinski_harabasz_score(features, labels)), 4
            )

            dunn_sample_size = min(1000, len(features))
            if len(features) > dunn_sample_size:
                dunn_indices = np.random.choice(len(features), dunn_sample_size, replace=False)
                dunn_features = features[dunn_indices]
                dunn_labels = labels[dunn_indices]
            else:
                dunn_features = features
                dunn_labels = labels

            metrics["dunn_index"] = compute_dunn_index(dunn_features, dunn_labels)

    except Exception:
        pass

    pixel_scatter_sample = []
    scatter_limit = min(2000, len(patch_arrays))
    for rgb, label in zip(patch_arrays[:scatter_limit], labels[:scatter_limit]):
        pixel_scatter_sample.append({
            "r": int(rgb[0]),
            "g": int(rgb[1]),
            "b": int(rgb[2]),
            "label": int(label),
        })

    processing_time = round(time.time() - start_time, 4)

    return {
        "clustered_image": clustered_image_b64,
        "segmentation_map_image": segmentation_image_b64,
        "processing_time": processing_time,
        "width": image_width,
        "height": image_height,
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "total_pixels": len(patches),
        "cluster_distribution": cluster_distribution,
        "cluster_centers": cluster_centers,
        "parameters_used": {
            "backbone_model": backbone_model,
            "feature_layer": feature_layer,
            "patch_size": patch_size,
            "n_components": n_components,
            "covariance_type": covariance_type,
            "feature_batch_size": feature_batch_size,
            "resnet_input_size": resnet_input_size,
            "auto_downsample_applied": auto_downsample_applied,
            "max_patches_limit": max_patches,
        },
        "metrics": metrics,
        "pixel_scatter_sample": pixel_scatter_sample,
    }

def process_dec(
    image_file,
    n_clusters=5,
    latent_dim=32,
    patch_size=16,
    max_epochs=10,
    downsample_enabled=False,
    downsample_size=256
):
    start_time = time.time()

    image = load_image_as_array(image_file)
    original_width, original_height = image.size

    max_side = max(image.size)
    auto_downsample_applied = False

    if downsample_enabled:
        image = downsample_image(image, downsample_size)
    elif max_side > 768:
        image = downsample_image(image, 256)
        auto_downsample_applied = True

    processed_width, processed_height = image.size

    patches, patch_positions, image_width, image_height = image_to_patches(
        image, patch_size=patch_size
    )

    max_patches = 1200
    if len(patches) > max_patches:
        selected_indices = np.linspace(0, len(patches) - 1, max_patches, dtype=int)
        patches = [patches[i] for i in selected_indices]
        patch_positions = [patch_positions[i] for i in selected_indices]

    if len(patches) == 0:
        raise ValueError("No valid patches could be extracted from the image.")

    patch_vectors, patch_mean_colors = patches_to_flat_vectors(patches)

    x_np = patch_vectors.astype(np.float32)
    x = torch.tensor(x_np, dtype=torch.float32)

    input_dim = x.shape[1]

    # CPU-friendly settings
    batch_size = min(256, max(32, len(x_np) // 8)) if len(x_np) > 0 else 64
    pretrain_epochs = min(8, max(4, max_epochs // 2))
    dec_epochs = min(15, max(6, max_epochs))
    patience = 3
    min_delta = 1e-4

    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    autoencoder = DECAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    # Step 1: pretrain autoencoder with mini-batches
    autoencoder.train()
    best_pretrain_loss = float("inf")
    no_improve_count = 0

    for _ in range(pretrain_epochs):
        epoch_loss = 0.0
        sample_count = 0

        for (batch_x,) in dataloader:
            z, x_recon = autoencoder(batch_x)
            recon_loss = mse_loss(x_recon, batch_x)

            ae_optimizer.zero_grad()
            recon_loss.backward()
            ae_optimizer.step()

            batch_size_actual = batch_x.size(0)
            epoch_loss += recon_loss.item() * batch_size_actual
            sample_count += batch_size_actual

        epoch_loss = epoch_loss / sample_count if sample_count > 0 else epoch_loss

        if best_pretrain_loss - epoch_loss > min_delta:
            best_pretrain_loss = epoch_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

    # Step 2: initialize latent space with KMeans
    autoencoder.eval()
    latent_batches = []

    with torch.no_grad():
        for (batch_x,) in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            z_batch, _ = autoencoder(batch_x)
            latent_batches.append(z_batch.cpu().numpy())

    z_init_np = np.concatenate(latent_batches, axis=0)

    # Normalize latent features for more stable initialization
    z_norms = np.linalg.norm(z_init_np, axis=1, keepdims=True)
    z_norms[z_norms == 0] = 1.0
    z_init_np = z_init_np / z_norms

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    _ = kmeans.fit_predict(z_init_np)

    dec_model = DECModel(
        autoencoder=autoencoder,
        n_clusters=n_clusters,
        latent_dim=latent_dim,
        alpha=1.0
    )

    dec_model.cluster_centers.data = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float32
    )

    optimizer = torch.optim.Adam(dec_model.parameters(), lr=1e-3)

    # Step 3: DEC fine-tuning with mini-batches + early stopping
    dec_model.train()
    best_dec_loss = float("inf")
    no_improve_count = 0

    for _ in range(dec_epochs):
        epoch_loss = 0.0
        sample_count = 0

        for (batch_x,) in dataloader:
            z, q, x_recon = dec_model(batch_x)
            p = target_distribution(q).detach()

            kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
            recon_loss = mse_loss(x_recon, batch_x)

            loss = kl_loss + 0.1 * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_x.size(0)
            epoch_loss += loss.item() * batch_size_actual
            sample_count += batch_size_actual

        epoch_loss = epoch_loss / sample_count if sample_count > 0 else epoch_loss

        if best_dec_loss - epoch_loss > min_delta:
            best_dec_loss = epoch_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

    # Step 4: final inference
    dec_model.eval()
    latent_final_batches = []
    q_final_batches = []

    with torch.no_grad():
        for (batch_x,) in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            z_batch, q_batch, _ = dec_model(batch_x)
            latent_final_batches.append(z_batch.cpu().numpy())
            q_final_batches.append(q_batch.cpu().numpy())

    latent_features = np.concatenate(latent_final_batches, axis=0)
    q_final_np = np.concatenate(q_final_batches, axis=0)
    labels = np.argmax(q_final_np, axis=1)

    centers_rgb = []
    for cluster_id in range(n_clusters):
        cluster_patch_colors = patch_mean_colors[labels == cluster_id]
        if len(cluster_patch_colors) == 0:
            centers_rgb.append(np.array([0, 0, 0], dtype=np.float32))
        else:
            centers_rgb.append(cluster_patch_colors.mean(axis=0))

    centers_rgb = np.clip(np.array(centers_rgb), 0, 255).astype(np.uint8)

    clustered_image = reconstruct_patch_clustered_image(
        patch_positions=patch_positions,
        labels=labels,
        centers_rgb=centers_rgb,
        image_width=image_width,
        image_height=image_height,
        patch_size=patch_size,
    )
    clustered_image_b64 = image_to_base64(clustered_image)

    seg_w = image_width // patch_size
    seg_h = image_height // patch_size

    segmentation_image = labels_to_segmentation_image(
        labels,
        (seg_w, seg_h)
    )
    segmentation_image = segmentation_image.resize((image_width, image_height), Image.NEAREST)
    segmentation_image_b64 = image_to_base64(segmentation_image)

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_distribution = []
    for label, count in zip(unique_labels, counts):
        cluster_distribution.append({
            "cluster_index": int(label),
            "pixel_count": int(count),
        })

    cluster_centers = []
    for idx, center in enumerate(centers_rgb):
        cluster_centers.append({
            "cluster_index": int(idx),
            "rgb": [int(center[0]), int(center[1]), int(center[2])]
        })

    metrics = {
        "inertia": None,
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
        "dunn_index": None,
    }

    try:
        if len(np.unique(labels)) > 1:
            # normalize latent features for metric stability
            latent_norms = np.linalg.norm(latent_features, axis=1, keepdims=True)
            latent_norms[latent_norms == 0] = 1.0
            latent_eval = latent_features / latent_norms

            sample_size = min(2000, len(latent_eval))
            if len(latent_eval) > sample_size:
                indices = np.random.choice(len(latent_eval), sample_size, replace=False)
                sampled_features = latent_eval[indices]
                sampled_labels = labels[indices]
            else:
                sampled_features = latent_eval
                sampled_labels = labels

            metrics["silhouette_score"] = round(
                float(silhouette_score(sampled_features, sampled_labels)), 4
            )
            metrics["davies_bouldin_score"] = round(
                float(davies_bouldin_score(latent_eval, labels)), 4
            )
            metrics["calinski_harabasz_score"] = round(
                float(calinski_harabasz_score(latent_eval, labels)), 4
            )

            dunn_sample_size = min(1000, len(latent_eval))
            if len(latent_eval) > dunn_sample_size:
                dunn_indices = np.random.choice(len(latent_eval), dunn_sample_size, replace=False)
                dunn_features = latent_eval[dunn_indices]
                dunn_labels = labels[dunn_indices]
            else:
                dunn_features = latent_eval
                dunn_labels = labels

            metrics["dunn_index"] = compute_dunn_index(dunn_features, dunn_labels)

    except Exception:
        pass

    pixel_scatter_sample = []
    scatter_limit = min(2000, len(patch_mean_colors))
    for rgb, label in zip(patch_mean_colors[:scatter_limit], labels[:scatter_limit]):
        pixel_scatter_sample.append({
            "r": int(rgb[0]),
            "g": int(rgb[1]),
            "b": int(rgb[2]),
            "label": int(label),
        })

    processing_time = round(time.time() - start_time, 4)

    return {
        "clustered_image": clustered_image_b64,
        "segmentation_map_image": segmentation_image_b64,
        "processing_time": processing_time,
        "width": image_width,
        "height": image_height,
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "total_pixels": len(patches),
        "cluster_distribution": cluster_distribution,
        "cluster_centers": cluster_centers,
        "parameters_used": {
            "n_clusters": n_clusters,
            "latent_dim": latent_dim,
            "patch_size": patch_size,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "pretrain_epochs": pretrain_epochs,
            "dec_epochs": dec_epochs,
            "auto_downsample_applied": auto_downsample_applied,
            "max_patches_limit": max_patches,
        },
        "metrics": metrics,
        "pixel_scatter_sample": pixel_scatter_sample,
    }

def labels_to_segmentation_image(labels: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
    width, height = original_size

    palette = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 128, 0],
        [128, 0, 255],
        [0, 128, 255],
        [128, 128, 128],
        [255, 105, 180],
        [0, 200, 100],
    ], dtype=np.uint8)

    mapped_colors = palette[labels % len(palette)]
    image_array = mapped_colors.reshape((height, width, 3))
    return Image.fromarray(image_array)

def downsample_image(image: Image.Image, target_size: int):
    image = image.copy()
    image.thumbnail((target_size, target_size))
    return image

def sample_pixel_space(pixels: np.ndarray, labels: np.ndarray, sample_size=2000):
    total = len(pixels)
    if total > sample_size:
        indices = np.random.choice(total, sample_size, replace=False)
        sampled_pixels = pixels[indices]
        sampled_labels = labels[indices]
    else:
        sampled_pixels = pixels
        sampled_labels = labels

    sampled_data = []
    for pixel, label in zip(sampled_pixels, sampled_labels):
        sampled_data.append({
            "r": int(pixel[0]),
            "g": int(pixel[1]),
            "b": int(pixel[2]),
            "label": int(label),
        })

    return sampled_data

#RESNET
_resnet_feature_extractors = {}

def get_resnet_feature_extractor(backbone_model="resnet50"):
    global _resnet_feature_extractors

    if backbone_model in _resnet_feature_extractors:
        return _resnet_feature_extractors[backbone_model]

    if backbone_model == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        backbone_model = "resnet50"

    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    _resnet_feature_extractors[backbone_model] = feature_extractor
    return feature_extractor

def get_resnet_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def image_to_patches(image: Image.Image, patch_size: int = 16):
    image_np = np.array(image)
    height, width, _ = image_np.shape

    patches = []
    patch_positions = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image_np[y:y + patch_size, x:x + patch_size]

            # skip incomplete edge patches for simplicity
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            patches.append(Image.fromarray(patch))
            patch_positions.append((x, y))

    return patches, patch_positions, width, height

def extract_resnet_features_from_patches(
    patches,
    backbone_model="resnet50",
    batch_size=32,
    input_size=224
):
    model = get_resnet_feature_extractor(backbone_model=backbone_model)
    transform = get_resnet_transform(input_size=input_size)

    if len(patches) == 0:
        return np.empty((0, 512 if backbone_model == "resnet18" else 2048), dtype=np.float32)

    tensors = [transform(patch) for patch in patches]
    features = []

    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i + batch_size], dim=0)  # [B, 3, H, W]
            output = model(batch)  # [B, C, 1, 1]
            output = output.view(output.size(0), -1)  # [B, C]
            features.append(output.cpu().numpy())

    return np.concatenate(features, axis=0).astype(np.float32)

def reconstruct_patch_clustered_image(
    patch_positions,
    labels,
    centers_rgb,
    image_width,
    image_height,
    patch_size=16
):
    output = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for (x, y), label in zip(patch_positions, labels):
        color = centers_rgb[label]
        output[y:y + patch_size, x:x + patch_size] = color

    return Image.fromarray(output)

# DEC
def patches_to_flat_vectors(patches):
    vectors = []
    patch_mean_colors = []

    for patch in patches:
        patch_np = np.array(patch).astype(np.float32) / 255.0
        vectors.append(patch_np.flatten())

        mean_rgb = patch_np.mean(axis=(0, 1)) * 255.0
        patch_mean_colors.append(mean_rgb)

    return np.array(vectors, dtype=np.float32), np.array(patch_mean_colors, dtype=np.float32)

def target_distribution(q):
    weight = (q ** 2) / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()

class DECAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

class DECModel(nn.Module):
    def __init__(self, autoencoder, n_clusters=5, latent_dim=32, alpha=1.0):
        super().__init__()
        self.autoencoder = autoencoder
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def forward(self, x):
        z, x_recon = self.autoencoder(x)
        q = self.soft_assign(z)
        return z, q, x_recon