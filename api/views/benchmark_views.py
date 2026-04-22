from collections import defaultdict
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

from ..dataset_utils import get_manifest_summary, get_image_paths
from ..models import BenchmarkSession, BenchmarkResult, DatasetImage, UserStudySession
from ..services.algorithm_dispatcher import run_algorithm_dispatch
from ..services.benchmark_service import create_benchmark_session, save_benchmark_result
from ..services.dataset_service import get_or_create_dataset_image_from_path, get_or_create_uploaded_dataset_image
from ..services.media_service import save_uploaded_original_image, save_base64_image, copy_dataset_original_to_media

class RunBenchmarkView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        images = request.FILES.getlist("images")
        algorithms = request.data.getlist("algorithms")

        if not images:
            return Response({"detail": "At least one image is required."}, status=400)

        if not algorithms:
            return Response({"detail": "At least one algorithm must be selected."}, status=400)

        try:
            params = {
                "downsample_enabled": str(request.data.get("downsample_enabled", "false")).lower() == "true",
                "downsample_size": int(request.data.get("downsample_size", 256)),
                "n_clusters": int(request.data.get("n_clusters", 5)),
                "patch_size": int(request.data.get("patch_size", 16)),
                "max_epochs": int(request.data.get("max_epochs", 30)),

                "dbscan_eps": float(request.data.get("dbscan_eps", 10)),
                "dbscan_min_samples": int(request.data.get("dbscan_min_samples", 5)),

                "gmm_covariance_type": request.data.get("gmm_covariance_type", "diag"),

                "birch_threshold": float(request.data.get("birch_threshold", 0.5)),
                "birch_branching_factor": int(request.data.get("birch_branching_factor", 50)),

                "agglomerative_linkage": request.data.get("agglomerative_linkage", "ward"),
                "agglomerative_metric": request.data.get("agglomerative_metric", "euclidean"),
            }
        except ValueError as exc:
            return Response({"detail": f"Invalid benchmark parameters: {str(exc)}"}, status=400)

        benchmark_session = create_benchmark_session(
            benchmark_mode="uploaded",
            algorithms=algorithms,
            categories=["uploaded"],
            params=params,
        )

        results = []

        for image in images:
            image_results = []
            dataset_image = get_or_create_uploaded_dataset_image(image)
            original_image_path = save_uploaded_original_image(image, benchmark_session.id)

            for algorithm_id in algorithms:
                try:
                    image.seek(0)

                    result = run_algorithm_dispatch(algorithm_id, image, params)   

                    entry = {
                        "image_name": image.name,
                        "algorithm": algorithm_id,
                        "processing_time": result.get("processing_time"),
                        "metrics": result.get("metrics", {}),
                        "type": "deep" if algorithm_id in ["resnet_kmeans", "resnet_gmm", "dec"] else "classical",
                        "is_patch_based": algorithm_id in ["resnet_kmeans", "resnet_gmm", "dec"],
                        "parameters_used": result.get("parameters_used", {}),
                        "original_image_path": original_image_path,
                        "clustered_image_path": save_base64_image(
                            result.get("clustered_image"),
                            folder_relative_path=f"benchmark_results/session_{benchmark_session.id}/clustered",
                            filename_prefix=f"{algorithm_id}_clustered"
                        ),
                        "segmentation_map_path": save_base64_image(
                            result.get("segmentation_map_image"),
                            folder_relative_path=f"benchmark_results/session_{benchmark_session.id}/segmentations",
                            filename_prefix=f"{algorithm_id}_segmentation"
                        ),
                    }

                    results.append(entry)
                    image_results.append(entry)

                except Exception as exc:
                    error_entry = {
                        "image_name": image.name,
                        "algorithm": algorithm_id,
                        "processing_time": None,
                        "metrics": {},
                        "type": "deep" if algorithm_id in ["resnet_kmeans", "resnet_gmm", "dec"] else "classical",
                        "is_patch_based": algorithm_id in ["resnet_kmeans", "resnet_gmm", "dec"],
                        "error": str(exc),
                        "original_image_path": original_image_path,
                        "clustered_image_path": "",
                        "segmentation_map_path": "",
                    }

                    results.append(error_entry)
                    image_results.append(error_entry)

            valid_runs = [r for r in image_results if r.get("processing_time") is not None]
            if valid_runs:
                fastest = min(valid_runs, key=lambda x: x["processing_time"])
                fastest["fastest_for_image"] = True

            for entry in image_results:
                save_benchmark_result(
                    session=benchmark_session,
                    dataset_image=dataset_image,
                    entry=entry,
                    parameters_used=params,
                )

        aggregate_map = defaultdict(lambda: {
            "processing_times": [],
            "silhouette_scores": [],
            "davies_bouldin_scores": [],
            "inertias": [],
            "calinski_harabasz_scores": [],
            "dunn_indices": [],
            "fastest_wins": 0,
        })

        for entry in results:
            algorithm_id = entry["algorithm"]
            metrics = entry.get("metrics", {})

            if entry.get("processing_time") is not None:
                aggregate_map[algorithm_id]["processing_times"].append(entry["processing_time"])

            if metrics.get("silhouette_score") is not None:
                aggregate_map[algorithm_id]["silhouette_scores"].append(metrics["silhouette_score"])

            if metrics.get("davies_bouldin_score") is not None:
                aggregate_map[algorithm_id]["davies_bouldin_scores"].append(metrics["davies_bouldin_score"])

            if metrics.get("inertia") is not None:
                aggregate_map[algorithm_id]["inertias"].append(metrics["inertia"])

            if metrics.get("calinski_harabasz_score") is not None:
                aggregate_map[algorithm_id]["calinski_harabasz_scores"].append(metrics["calinski_harabasz_score"])

            if metrics.get("dunn_index") is not None:
                aggregate_map[algorithm_id]["dunn_indices"].append(metrics["dunn_index"])

            if entry.get("fastest_for_image"):
                aggregate_map[algorithm_id]["fastest_wins"] += 1

        aggregates = {}

        deep_algorithms = {"resnet_kmeans", "resnet_gmm", "dec"}
        classical_algorithms = {"kmeans", "mini_batch_kmeans", "gmm", "bgmm", "agglomerative", "birch", "dbscan"}

        for algorithm_id, stats in aggregate_map.items():
            aggregates[algorithm_id] = {
                "avg_processing_time": round(sum(stats["processing_times"]) / len(stats["processing_times"]), 4)
                if stats["processing_times"] else None,

                "avg_silhouette_score": round(sum(stats["silhouette_scores"]) / len(stats["silhouette_scores"]), 4)
                if stats["silhouette_scores"] else None,

                "avg_davies_bouldin_score": round(sum(stats["davies_bouldin_scores"]) / len(stats["davies_bouldin_scores"]), 4)
                if stats["davies_bouldin_scores"] else None,

                "avg_inertia": round(sum(stats["inertias"]) / len(stats["inertias"]), 4)
                if stats["inertias"] else None,

                "avg_calinski_harabasz_score": round(
                    sum(stats["calinski_harabasz_scores"]) / len(stats["calinski_harabasz_scores"]), 4
                ) if stats["calinski_harabasz_scores"] else None,

                "avg_dunn_index": round(
                    sum(stats["dunn_indices"]) / len(stats["dunn_indices"]), 4
                ) if stats["dunn_indices"] else None,

                "fastest_wins": stats["fastest_wins"],
            }

        fastest_algorithm = None
        best_silhouette_algorithm = None

        valid_time_algorithms = [
            (alg, data["avg_processing_time"])
            for alg, data in aggregates.items()
            if data["avg_processing_time"] is not None
        ]
        if valid_time_algorithms:
            fastest_algorithm = min(valid_time_algorithms, key=lambda x: x[1])[0]

        valid_sil_algorithms = [
            (alg, data["avg_silhouette_score"])
            for alg, data in aggregates.items()
            if data["avg_silhouette_score"] is not None
        ]
        if valid_sil_algorithms:
            best_silhouette_algorithm = max(valid_sil_algorithms, key=lambda x: x[1])[0]

        family_stats = {
            "classical": {
                "times": [],
                "silhouettes": [],
                "davies_bouldin": [],
            },
            "deep": {
                "times": [],
                "silhouettes": [],
                "davies_bouldin": [],
            }
        }

        for algorithm_id, stats in aggregates.items():
            family = "deep" if algorithm_id in deep_algorithms else "classical"

            if stats["avg_processing_time"] is not None:
                family_stats[family]["times"].append(stats["avg_processing_time"])

            if stats["avg_silhouette_score"] is not None:
                family_stats[family]["silhouettes"].append(stats["avg_silhouette_score"])

            if stats["avg_davies_bouldin_score"] is not None:
                family_stats[family]["davies_bouldin"].append(stats["avg_davies_bouldin_score"])

        family_averages = {
            "classical": {
                "avg_processing_time": round(sum(family_stats["classical"]["times"]) / len(family_stats["classical"]["times"]), 4)
                if family_stats["classical"]["times"] else None,
                "avg_silhouette_score": round(sum(family_stats["classical"]["silhouettes"]) / len(family_stats["classical"]["silhouettes"]), 4)
                if family_stats["classical"]["silhouettes"] else None,
                "avg_davies_bouldin_score": round(sum(family_stats["classical"]["davies_bouldin"]) / len(family_stats["classical"]["davies_bouldin"]), 4)
                if family_stats["classical"]["davies_bouldin"] else None,
            },
            "deep": {
                "avg_processing_time": round(sum(family_stats["deep"]["times"]) / len(family_stats["deep"]["times"]), 4)
                if family_stats["deep"]["times"] else None,
                "avg_silhouette_score": round(sum(family_stats["deep"]["silhouettes"]) / len(family_stats["deep"]["silhouettes"]), 4)
                if family_stats["deep"]["silhouettes"] else None,
                "avg_davies_bouldin_score": round(sum(family_stats["deep"]["davies_bouldin"]) / len(family_stats["deep"]["davies_bouldin"]), 4)
                if family_stats["deep"]["davies_bouldin"] else None,
            }
        }

        insights = []

        if fastest_algorithm:
            insights.append(f"{fastest_algorithm} was the fastest algorithm overall.")

        if best_silhouette_algorithm:
            insights.append(f"{best_silhouette_algorithm} achieved the highest average silhouette score.")

        valid_db_algorithms = [
            (alg, data["avg_davies_bouldin_score"])
            for alg, data in aggregates.items()
            if data["avg_davies_bouldin_score"] is not None
        ]
        if valid_db_algorithms:
            best_db_algorithm = min(valid_db_algorithms, key=lambda x: x[1])[0]
            insights.append(f"{best_db_algorithm} achieved the best average Davies-Bouldin score.")

        classical_time = family_averages["classical"]["avg_processing_time"]
        deep_time = family_averages["deep"]["avg_processing_time"]

        if classical_time is not None and deep_time is not None:
            if classical_time < deep_time:
                insights.append("Classical methods were faster on average than deep methods.")
            elif deep_time < classical_time:
                insights.append("Deep methods were faster on average than classical methods.")

        classical_sil = family_averages["classical"]["avg_silhouette_score"]
        deep_sil = family_averages["deep"]["avg_silhouette_score"]

        if classical_sil is not None and deep_sil is not None:
            if classical_sil > deep_sil:
                insights.append("Classical methods achieved better average silhouette scores on this benchmark.")
            elif deep_sil > classical_sil:
                insights.append("Deep methods achieved better average silhouette scores on this benchmark.")

        classical_db = family_averages["classical"]["avg_davies_bouldin_score"]
        deep_db = family_averages["deep"]["avg_davies_bouldin_score"]

        if classical_db is not None and deep_db is not None:
            if classical_db < deep_db:
                insights.append("Classical methods achieved better average Davies-Bouldin scores on this benchmark.")
            elif deep_db < classical_db:
                insights.append("Deep methods achieved better average Davies-Bouldin scores on this benchmark.")

        return Response({
            "summary": {
                "benchmark_session_id": benchmark_session.id,
                "benchmark_mode": "uploaded",
                "total_images": len(images),
                "total_algorithms": len(algorithms),
                "total_runs": len(images) * len(algorithms),
                "fastest_algorithm": fastest_algorithm,
                "best_silhouette_algorithm": best_silhouette_algorithm,
            },
            "results": results,
            "aggregates": aggregates,
            "family_averages": family_averages,
            "insights": insights,
        }, status=200)
    
class RunDatasetBenchmarkView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        algorithms = request.data.getlist("algorithms")
        categories = request.data.getlist("categories")
        benchmark_mode = request.data.get("benchmark_mode", "full")
        use_subset = benchmark_mode == "subset20"

        if not algorithms:
            return Response({"detail": "At least one algorithm must be selected."}, status=400)

        try:
            max_images_raw = request.data.get("max_images")
            max_images = int(max_images_raw) if max_images_raw else None

            params = {
                "downsample_enabled": str(request.data.get("downsample_enabled", "false")).lower() == "true",
                "downsample_size": int(request.data.get("downsample_size", 256)),
                "n_clusters": int(request.data.get("n_clusters", 5)),
                "patch_size": int(request.data.get("patch_size", 16)),
                "max_epochs": int(request.data.get("max_epochs", 30)),

                "dbscan_eps": float(request.data.get("dbscan_eps", 10)),
                "dbscan_min_samples": int(request.data.get("dbscan_min_samples", 5)),

                "gmm_covariance_type": request.data.get("gmm_covariance_type", "diag"),

                "birch_threshold": float(request.data.get("birch_threshold", 0.5)),
                "birch_branching_factor": int(request.data.get("birch_branching_factor", 50)),

                "agglomerative_linkage": request.data.get("agglomerative_linkage", "ward"),
                "agglomerative_metric": request.data.get("agglomerative_metric", "euclidean"),
            }
        except ValueError as exc:
            return Response({"detail": f"Invalid benchmark parameters: {str(exc)}"}, status=400)

        try:
            dataset_images = get_image_paths(
                categories=categories if categories else None,
                max_images=max_images,
                use_subset=use_subset,
            )
        except Exception as exc:
            return Response({"detail": str(exc)}, status=500)

        if not dataset_images:
            return Response({"detail": "No dataset images found for the selected filters."}, status=400)
        
        benchmark_session = create_benchmark_session(
            benchmark_mode=benchmark_mode,
            algorithms=algorithms,
            categories=categories,
            params=params,
        )

        results = []

        for image_info in dataset_images:
            image_results = []

            dataset_image = get_or_create_dataset_image_from_path(
                relative_path=image_info["relative_path"],
                category=image_info["category"],
                absolute_path=image_info["absolute_path"],
            )

            original_image_path = copy_dataset_original_to_media(
                absolute_path=image_info["absolute_path"],
                session_id=benchmark_session.id,
            )

            for algorithm_id in algorithms:
                try:
                    with open(image_info["absolute_path"], "rb") as image_file:
                        result = run_algorithm_dispatch(algorithm_id, image_file, params)

                    entry = {
                        "image_name": image_info["filename"],
                        "image_id": image_info["id"],
                        "relative_path": image_info["relative_path"],
                        "category": image_info["category"],
                        "algorithm": algorithm_id,
                        "processing_time": result.get("processing_time"),
                        "metrics": result.get("metrics", {}),
                        "type": "deep" if algorithm_id in ["resnet_kmeans", "resnet_gmm", "dec"] else "classical",
                        "is_patch_based": algorithm_id in ["resnet_kmeans", "resnet_gmm", "dec"],
                        "parameters_used": result.get("parameters_used", {}),
                        "original_image_path": original_image_path,
                        "clustered_image_path": save_base64_image(
                            result.get("clustered_image"),
                            folder_relative_path=f"benchmark_results/session_{benchmark_session.id}/clustered",
                            filename_prefix=f"{algorithm_id}_clustered"
                        ),
                        "segmentation_map_path": save_base64_image(
                            result.get("segmentation_map_image"),
                            folder_relative_path=f"benchmark_results/session_{benchmark_session.id}/segmentations",
                            filename_prefix=f"{algorithm_id}_segmentation"
                        ),
                    }

                    results.append(entry)
                    image_results.append(entry)

                except Exception as exc:
                    error_entry = {
                        "image_name": image_info["filename"],
                        "image_id": image_info["id"],
                        "relative_path": image_info["relative_path"],
                        "category": image_info["category"],
                        "algorithm": algorithm_id,
                        "processing_time": None,
                        "metrics": {},
                        "type": "deep" if algorithm_id in ["resnet_kmeans", "resnet_gmm", "dec"] else "classical",
                        "is_patch_based": algorithm_id in ["resnet_kmeans", "resnet_gmm", "dec"],
                        "parameters_used": {},
                        "error": str(exc),
                        "original_image_path": original_image_path,
                        "clustered_image_path": "",
                        "segmentation_map_path": "",
                    }

                    results.append(error_entry)
                    image_results.append(error_entry)

            valid_runs = [r for r in image_results if r.get("processing_time") is not None]
            if valid_runs:
                fastest = min(valid_runs, key=lambda x: x["processing_time"])
                fastest["fastest_for_image"] = True

            # save all results for this image after fastest flag is known
            for entry in image_results:
                save_benchmark_result(
                    session=benchmark_session,
                    dataset_image=dataset_image,
                    entry=entry,
                    parameters_used=entry.get("parameters_used", {}),
                )

        aggregate_map = defaultdict(lambda: {
            "processing_times": [],
            "silhouette_scores": [],
            "davies_bouldin_scores": [],
            "inertias": [],
            "calinski_harabasz_scores": [],
            "dunn_indices": [],
            "fastest_wins": 0,
        })

        category_algorithm_map = defaultdict(lambda: {
            "processing_times": [],
            "silhouette_scores": [],
            "davies_bouldin_scores": [],
            "inertias": [],
            "calinski_harabasz_scores": [],
            "dunn_indices": [],
            "fastest_wins": 0,
        })

        for entry in results:
            algorithm_id = entry["algorithm"]
            category = entry["category"]
            metrics = entry.get("metrics", {})

            if entry.get("processing_time") is not None:
                aggregate_map[algorithm_id]["processing_times"].append(entry["processing_time"])
                category_algorithm_map[(category, algorithm_id)]["processing_times"].append(entry["processing_time"])

            if metrics.get("silhouette_score") is not None:
                aggregate_map[algorithm_id]["silhouette_scores"].append(metrics["silhouette_score"])
                category_algorithm_map[(category, algorithm_id)]["silhouette_scores"].append(metrics["silhouette_score"])

            if metrics.get("davies_bouldin_score") is not None:
                aggregate_map[algorithm_id]["davies_bouldin_scores"].append(metrics["davies_bouldin_score"])
                category_algorithm_map[(category, algorithm_id)]["davies_bouldin_scores"].append(metrics["davies_bouldin_score"])

            if metrics.get("calinski_harabasz_score") is not None:
                aggregate_map[algorithm_id]["calinski_harabasz_scores"].append(metrics["calinski_harabasz_score"])
                category_algorithm_map[(category, algorithm_id)]["calinski_harabasz_scores"].append(metrics["calinski_harabasz_score"])

            if metrics.get("dunn_index") is not None:
                aggregate_map[algorithm_id]["dunn_indices"].append(metrics["dunn_index"])
                category_algorithm_map[(category, algorithm_id)]["dunn_indices"].append(metrics["dunn_index"])

            if metrics.get("inertia") is not None:
                aggregate_map[algorithm_id]["inertias"].append(metrics["inertia"])
                category_algorithm_map[(category, algorithm_id)]["inertias"].append(metrics["inertia"])

            if entry.get("fastest_for_image"):
                aggregate_map[algorithm_id]["fastest_wins"] += 1
                category_algorithm_map[(category, algorithm_id)]["fastest_wins"] += 1

        def avg(values):
            return round(sum(values) / len(values), 4) if values else None

        aggregates = {}
        for algorithm_id, stats in aggregate_map.items():
            aggregates[algorithm_id] = {
                "avg_processing_time": avg(stats["processing_times"]),
                "avg_silhouette_score": avg(stats["silhouette_scores"]),
                "avg_davies_bouldin_score": avg(stats["davies_bouldin_scores"]),
                "avg_inertia": avg(stats["inertias"]),
                "avg_calinski_harabasz_score": avg(stats["calinski_harabasz_scores"]),
                "avg_dunn_index": avg(stats["dunn_indices"]),
                "fastest_wins": stats["fastest_wins"],
            }

        category_aggregates = {}
        for (category, algorithm_id), stats in category_algorithm_map.items():
            if category not in category_aggregates:
                category_aggregates[category] = {}

            category_aggregates[category][algorithm_id] = {
                "avg_processing_time": avg(stats["processing_times"]),
                "avg_silhouette_score": avg(stats["silhouette_scores"]),
                "avg_davies_bouldin_score": avg(stats["davies_bouldin_scores"]),
                "avg_inertia": avg(stats["inertias"]),
                "avg_calinski_harabasz_score": avg(stats["calinski_harabasz_scores"]),
                "avg_dunn_index": avg(stats["dunn_indices"]),
                "fastest_wins": stats["fastest_wins"],
            }

        fastest_algorithm = None
        best_silhouette_algorithm = None

        valid_time_algorithms = [
            (alg, data["avg_processing_time"])
            for alg, data in aggregates.items()
            if data["avg_processing_time"] is not None
        ]
        if valid_time_algorithms:
            fastest_algorithm = min(valid_time_algorithms, key=lambda x: x[1])[0]

        valid_sil_algorithms = [
            (alg, data["avg_silhouette_score"])
            for alg, data in aggregates.items()
            if data["avg_silhouette_score"] is not None
        ]
        if valid_sil_algorithms:
            best_silhouette_algorithm = max(valid_sil_algorithms, key=lambda x: x[1])[0]

        return Response({
            "summary": {
                "benchmark_session_id": benchmark_session.id,
                "benchmark_mode": benchmark_mode,
                "total_images": len(dataset_images),
                "selected_categories": categories,
                "total_algorithms": len(algorithms),
                "total_runs": len(dataset_images) * len(algorithms),
                "fastest_algorithm": fastest_algorithm,
                "best_silhouette_algorithm": best_silhouette_algorithm,
            },
            "results": results,
            "aggregates": aggregates,
            "category_aggregates": category_aggregates,           
        }, status=200)
    
class DatasetManifestView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        try:
            summary = get_manifest_summary()
            return Response(summary, status=200)
        except Exception as exc:
            return Response({"detail": str(exc)}, status=500)
    
class BenchmarkSessionListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        sessions = BenchmarkSession.objects.order_by("-created_at")[:100]

        data = []
        for session in sessions:
            existing_study = UserStudySession.objects.filter(benchmark_session=session).first()

            data.append({
                "id": session.id,
                "created_at": session.created_at,
                "benchmark_mode": session.benchmark_mode,
                "selected_algorithms": session.selected_algorithms,
                "selected_categories": session.selected_categories,
                "result_count": session.results.count(),
                "has_user_study": existing_study is not None,
                "user_study_id": existing_study.id if existing_study else None,
                "user_study_title": existing_study.title if existing_study else None,
            })

        return Response(data, status=200)
    
class BenchmarkSessionDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id, *args, **kwargs):
        try:
            session = BenchmarkSession.objects.get(id=session_id)
        except BenchmarkSession.DoesNotExist:
            return Response({"detail": "Benchmark session not found."}, status=404)

        results = session.results.select_related("dataset_image").all().order_by("dataset_image__category", "dataset_image__filename", "algorithm_id")

        data = {
            "id": session.id,
            "created_at": session.created_at,
            "benchmark_mode": session.benchmark_mode,
            "selected_algorithms": session.selected_algorithms,
            "selected_categories": session.selected_categories,
            "parameters": session.parameters,
            "results": [
                {
                    "id": result.id,
                    "image_name": result.dataset_image.filename,
                    "category": result.dataset_image.category,
                    "relative_path": result.dataset_image.relative_path,
                    "algorithm_id": result.algorithm_id,
                    "algorithm_type": result.algorithm_type,
                    "is_patch_based": result.is_patch_based,
                    "processing_time": result.processing_time,
                    "silhouette_score": result.silhouette_score,
                    "davies_bouldin_score": result.davies_bouldin_score,
                    "inertia": result.inertia,
                    "calinski_harabasz_score": result.calinski_harabasz_score,
                    "dunn_index": result.dunn_index,
                    "fastest_for_image": result.fastest_for_image,
                    "parameters_used": result.parameters_used,
                    "error_message": result.error_message,
                }
                for result in results
            ]
        }

        return Response(data, status=200)