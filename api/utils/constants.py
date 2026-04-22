ALGORITHM_LABELS = {
    "kmeans": "K-means",
    "mini_batch_kmeans": "Mini-Batch K-means",
    "gmm": "EM / GMM",
    "bgmm": "Bayesian GMM",
    "agglomerative": "Agglomerative",
    "birch": "BIRCH",
    "dbscan": "DBSCAN",
    "resnet_kmeans": "ResNet + K-means",
    "resnet_gmm": "ResNet + GMM",
    "dec": "DEC",
}

METRIC_LABELS = {
    "silhouette_score": "Silhouette Score",
    "davies_bouldin_score": "Davies-Bouldin Score",
    "calinski_harabasz_score": "Calinski-Harabasz Score",
    "dunn_index": "Dunn Index",
}

# Utility functions to get human-friendly labels for algorithms and metrics
def get_algorithm_label(algorithm_id):
    return ALGORITHM_LABELS.get(algorithm_id, algorithm_id)

# Utility function to get human-friendly labels for metrics
def get_metric_label(metric_name):
    return METRIC_LABELS.get(metric_name, metric_name)