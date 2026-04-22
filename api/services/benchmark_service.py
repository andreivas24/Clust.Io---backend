from ..models import BenchmarkResult, BenchmarkSession


def create_benchmark_session(benchmark_mode, algorithms, categories, params):
    return BenchmarkSession.objects.create(
        benchmark_mode=benchmark_mode,
        selected_algorithms=list(algorithms),
        selected_categories=list(categories),
        parameters=params,
    )

# Utility function to save benchmark results to the database
def save_benchmark_result(session, dataset_image, entry, parameters_used=None):
    metrics = entry.get("metrics", {})

    BenchmarkResult.objects.create(
        session=session,
        dataset_image=dataset_image,
        algorithm_id=entry.get("algorithm", ""),
        algorithm_type=entry.get("type", ""),
        is_patch_based=entry.get("is_patch_based", False),

        processing_time=entry.get("processing_time"),

        silhouette_score=metrics.get("silhouette_score"),
        davies_bouldin_score=metrics.get("davies_bouldin_score"),
        inertia=metrics.get("inertia"),
        calinski_harabasz_score=metrics.get("calinski_harabasz_score"),
        dunn_index=metrics.get("dunn_index"),

        fastest_for_image=entry.get("fastest_for_image", False),

        parameters_used=parameters_used or {},
        error_message=entry.get("error", "") or "",

        original_image_path=entry.get("original_image_path", "") or "",
        clustered_image_path=entry.get("clustered_image_path", "") or "",
        segmentation_map_path=entry.get("segmentation_map_path", "") or "",
    )