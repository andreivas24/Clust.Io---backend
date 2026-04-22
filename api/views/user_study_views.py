from collections import defaultdict
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ..services.analytics_service import compute_pearson_correlation
from ..utils.constants import get_algorithm_label, get_metric_label
from django.conf import settings

from ..models import BenchmarkSession, BenchmarkResult, DatasetImage, UserImageRating, UserStudySession

class CreateUserStudySessionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        title = request.data.get("title")
        benchmark_session_id = request.data.get("benchmark_session_id")

        if not title:
            return Response({"detail": "title is required."}, status=400)

        if not benchmark_session_id:
            return Response({"detail": "benchmark_session_id is required."}, status=400)

        try:
            benchmark_session = BenchmarkSession.objects.get(id=benchmark_session_id)
        except BenchmarkSession.DoesNotExist:
            return Response({"detail": "Benchmark session not found."}, status=404)

        existing_study = UserStudySession.objects.filter(benchmark_session=benchmark_session).first()
        if existing_study:
            return Response({
                "detail": "A user study already exists for this benchmark session.",
                "existing_study_id": existing_study.id,
                "existing_study_title": existing_study.title,
            }, status=400)

        study = UserStudySession.objects.create(
            title=title,
            benchmark_session=benchmark_session,
        )

        return Response({
            "id": study.id,
            "title": study.title,
            "benchmark_session_id": benchmark_session.id,
            "created_at": study.created_at,
        }, status=201)
    
class UserStudyDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, study_id, *args, **kwargs):
        try:
            study = UserStudySession.objects.select_related("benchmark_session").get(id=study_id)
        except UserStudySession.DoesNotExist:
            return Response({"detail": "User study not found."}, status=404)

        benchmark_results = study.benchmark_session.results.select_related("dataset_image").all().order_by(
            "dataset_image__category", "dataset_image__filename", "algorithm_id"
        )

        grouped = {}

        for result in benchmark_results:
            image_key = result.dataset_image.relative_path

            if image_key not in grouped:
                grouped[image_key] = {
                    "image_name": result.dataset_image.filename,
                    "category": result.dataset_image.category,
                    "relative_path": result.dataset_image.relative_path,
                    "benchmark_results": [],
                }

            grouped[image_key]["benchmark_results"].append({
                "benchmark_result_id": result.id,
                "algorithm_id": result.algorithm_id,
                "algorithm_type": result.algorithm_type,
                "processing_time": result.processing_time,
                "silhouette_score": result.silhouette_score,
                "davies_bouldin_score": result.davies_bouldin_score,
                "inertia": result.inertia,
                "calinski_harabasz_score": result.calinski_harabasz_score,
                "dunn_index": result.dunn_index,
                "fastest_for_image": result.fastest_for_image,
                "error_message": result.error_message,
                "original_image_url": f"{settings.MEDIA_URL}{result.original_image_path}" if result.original_image_path else "",
                "clustered_image_url": f"{settings.MEDIA_URL}{result.clustered_image_path}" if result.clustered_image_path else "",
                "segmentation_map_url": f"{settings.MEDIA_URL}{result.segmentation_map_path}" if result.segmentation_map_path else "",
            })

        return Response({
            "id": study.id,
            "title": study.title,
            "created_at": study.created_at,
            "benchmark_session_id": study.benchmark_session.id,
            "images": list(grouped.values()),
        }, status=200)

class SubmitUserRatingView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        study_session_id = request.data.get("study_session_id")
        benchmark_result_id = request.data.get("benchmark_result_id")
        quality_score = request.data.get("quality_score")
        naturalness_score = request.data.get("naturalness_score")
        separation_score = request.data.get("separation_score")

        if not study_session_id or not benchmark_result_id or quality_score is None:
            return Response(
                {"detail": "study_session_id, benchmark_result_id and quality_score are required."},
                status=400
            )

        try:
            quality_score = int(quality_score)
            naturalness_score = int(naturalness_score) if naturalness_score not in [None, ""] else None
            separation_score = int(separation_score) if separation_score not in [None, ""] else None
        except ValueError:
            return Response({"detail": "Ratings must be numeric values."}, status=400)

        if not (1 <= quality_score <= 5):
            return Response({"detail": "Quality score must be between 1 and 5."}, status=400)

        if naturalness_score is not None and not (1 <= naturalness_score <= 5):
            return Response({"detail": "Naturalness score must be between 1 and 5."}, status=400)

        if separation_score is not None and not (1 <= separation_score <= 5):
            return Response({"detail": "Separation score must be between 1 and 5."}, status=400)

        try:
            study = UserStudySession.objects.get(id=study_session_id)
        except UserStudySession.DoesNotExist:
            return Response({"detail": "User study not found."}, status=404)

        try:
            benchmark_result = BenchmarkResult.objects.get(id=benchmark_result_id)
        except BenchmarkResult.DoesNotExist:
            return Response({"detail": "Benchmark result not found."}, status=404)

        participant_id = request.user.username

        rating, created = UserImageRating.objects.update_or_create(
            study_session=study,
            benchmark_result=benchmark_result,
            participant_id=participant_id,
            defaults={
                "quality_score": quality_score,
                "naturalness_score": naturalness_score,
                "separation_score": separation_score,
            }
        )

        return Response({
            "id": rating.id,
            "study_session_id": study.id,
            "benchmark_result_id": benchmark_result.id,
            "participant_id": rating.participant_id,
            "quality_score": rating.quality_score,
            "naturalness_score": rating.naturalness_score,
            "separation_score": rating.separation_score,
            "created": created,
        }, status=200 if not created else 201)

class UserStudyRatingsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, study_id, *args, **kwargs):
        participant_id = request.user.username

        ratings = UserImageRating.objects.filter(
            study_session_id=study_id,
            participant_id=participant_id
        )

        data = [
            {
                "benchmark_result_id": rating.benchmark_result_id,
                "quality_score": rating.quality_score,
                "naturalness_score": rating.naturalness_score,
                "separation_score": rating.separation_score,
            }
            for rating in ratings
        ]

        return Response(data, status=200)
    
class UserStudyAnalyticsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, study_id, *args, **kwargs):
        try:
            study = UserStudySession.objects.get(id=study_id)
        except UserStudySession.DoesNotExist:
            return Response({"detail": "User study not found."}, status=404)

        ratings = UserImageRating.objects.select_related(
            "benchmark_result",
            "benchmark_result__dataset_image"
        ).filter(study_session=study)

        unique_participants = ratings.values_list("participant_id", flat=True).distinct()
        participant_count = len(unique_participants)

        if not ratings.exists():
            return Response({
                "study_id": study.id,
                "study_title": study.title,
                "total_ratings": 0,
                "algorithm_averages": {},
                "pearson_correlations": {},
            }, status=200)

        algorithm_buckets = defaultdict(lambda: {
            "quality_scores": [],
            "naturalness_scores": [],
            "separation_scores": [],
            "silhouette_scores": [],
            "davies_bouldin_scores": [],
            "calinski_harabasz_scores": [],
            "dunn_indices": [],
        })

        quality_pairs = {
            "silhouette_score": {"metric": [], "rating": []},
            "davies_bouldin_score": {"metric": [], "rating": []},
            "calinski_harabasz_score": {"metric": [], "rating": []},
            "dunn_index": {"metric": [], "rating": []},
        }

        naturalness_pairs = {
            "silhouette_score": {"metric": [], "rating": []},
            "davies_bouldin_score": {"metric": [], "rating": []},
            "calinski_harabasz_score": {"metric": [], "rating": []},
            "dunn_index": {"metric": [], "rating": []},
        }

        separation_pairs = {
            "silhouette_score": {"metric": [], "rating": []},
            "davies_bouldin_score": {"metric": [], "rating": []},
            "calinski_harabasz_score": {"metric": [], "rating": []},
            "dunn_index": {"metric": [], "rating": []},
        }

        scatter_data = {
            "quality_score": {
                "silhouette_score": [],
                "davies_bouldin_score": [],
                "calinski_harabasz_score": [],
                "dunn_index": [],
            },
            "naturalness_score": {
                "silhouette_score": [],
                "davies_bouldin_score": [],
                "calinski_harabasz_score": [],
                "dunn_index": [],
            },
            "separation_score": {
                "silhouette_score": [],
                "davies_bouldin_score": [],
                "calinski_harabasz_score": [],
                "dunn_index": [],
            },
        }

        for rating in ratings:
            result = rating.benchmark_result
            algorithm_id = result.algorithm_id

            bucket = algorithm_buckets[algorithm_id]

            bucket["quality_scores"].append(rating.quality_score)

            if rating.naturalness_score is not None:
                bucket["naturalness_scores"].append(rating.naturalness_score)

            if rating.separation_score is not None:
                bucket["separation_scores"].append(rating.separation_score)

            if result.silhouette_score is not None:
                bucket["silhouette_scores"].append(result.silhouette_score)

            if result.davies_bouldin_score is not None:
                bucket["davies_bouldin_scores"].append(result.davies_bouldin_score)

            if result.calinski_harabasz_score is not None:
                bucket["calinski_harabasz_scores"].append(result.calinski_harabasz_score)

            if result.dunn_index is not None:
                bucket["dunn_indices"].append(result.dunn_index)

            metric_map = {
                "silhouette_score": result.silhouette_score,
                "davies_bouldin_score": result.davies_bouldin_score,
                "calinski_harabasz_score": result.calinski_harabasz_score,
                "dunn_index": result.dunn_index,
            }

            for metric_name, metric_value in metric_map.items():
                if metric_value is not None and rating.quality_score is not None:
                    scatter_data["quality_score"][metric_name].append({
                        "x": metric_value,
                        "y": rating.quality_score,
                        "algorithm_id": result.algorithm_id,
                    })

                if metric_value is not None and rating.naturalness_score is not None:
                    scatter_data["naturalness_score"][metric_name].append({
                        "x": metric_value,
                        "y": rating.naturalness_score,
                        "algorithm_id": result.algorithm_id,
                    })

                if metric_value is not None and rating.separation_score is not None:
                    scatter_data["separation_score"][metric_name].append({
                        "x": metric_value,
                        "y": rating.separation_score,
                        "algorithm_id": result.algorithm_id,
                    })

            for metric_name, metric_value in metric_map.items():
                if metric_value is not None and rating.quality_score is not None:
                    quality_pairs[metric_name]["metric"].append(metric_value)
                    quality_pairs[metric_name]["rating"].append(rating.quality_score)

                if metric_value is not None and rating.naturalness_score is not None:
                    naturalness_pairs[metric_name]["metric"].append(metric_value)
                    naturalness_pairs[metric_name]["rating"].append(rating.naturalness_score)

                if metric_value is not None and rating.separation_score is not None:
                    separation_pairs[metric_name]["metric"].append(metric_value)
                    separation_pairs[metric_name]["rating"].append(rating.separation_score)

        def avg(values):
            return round(sum(values) / len(values), 4) if values else None

        algorithm_averages = {}
        for algorithm_id, bucket in algorithm_buckets.items():
            algorithm_averages[algorithm_id] = {
                "avg_quality_score": avg(bucket["quality_scores"]),
                "avg_naturalness_score": avg(bucket["naturalness_scores"]),
                "avg_separation_score": avg(bucket["separation_scores"]),
                "avg_silhouette_score": avg(bucket["silhouette_scores"]),
                "avg_davies_bouldin_score": avg(bucket["davies_bouldin_scores"]),
                "avg_calinski_harabasz_score": avg(bucket["calinski_harabasz_scores"]),
                "avg_dunn_index": avg(bucket["dunn_indices"]),
                "rating_count": len(bucket["quality_scores"]),
            }

        final_ranking = []

        for algorithm_id, stats in algorithm_averages.items():
            quality = stats.get("avg_quality_score")
            naturalness = stats.get("avg_naturalness_score")
            separation = stats.get("avg_separation_score")

            if quality is None and naturalness is None and separation is None:
                continue

            quality = quality or 0
            naturalness = naturalness or 0
            separation = separation or 0

            final_score = round(
                0.4 * quality +
                0.3 * naturalness +
                0.3 * separation,
                4
            )

            final_ranking.append({
                "algorithm_id": algorithm_id,
                "final_score": final_score,
                "avg_quality_score": stats.get("avg_quality_score"),
                "avg_naturalness_score": stats.get("avg_naturalness_score"),
                "avg_separation_score": stats.get("avg_separation_score"),
                "rating_count": stats.get("rating_count", 0),
            })

        final_ranking.sort(key=lambda item: item["final_score"], reverse=True)

        classical_algorithms = {
            "kmeans", "mini_batch_kmeans", "gmm", "bgmm",
            "agglomerative", "birch", "dbscan"
        }
        deep_algorithms = {"resnet_kmeans", "resnet_gmm", "dec"}

        def average_group_score(algorithm_ids, score_field, lower_is_better=False):
            values = [
                stats.get(score_field)
                for alg, stats in algorithm_averages.items()
                if alg in algorithm_ids and stats.get(score_field) is not None
            ]
            if not values:
                return None
            value = sum(values) / len(values)
            return round(value, 4)

        classical_vs_deep = {
            "classical": {
                "avg_quality_score": average_group_score(classical_algorithms, "avg_quality_score"),
                "avg_naturalness_score": average_group_score(classical_algorithms, "avg_naturalness_score"),
                "avg_separation_score": average_group_score(classical_algorithms, "avg_separation_score"),
                "avg_silhouette_score": average_group_score(classical_algorithms, "avg_silhouette_score"),
                "avg_davies_bouldin_score": average_group_score(classical_algorithms, "avg_davies_bouldin_score"),
                "avg_calinski_harabasz_score": average_group_score(classical_algorithms, "avg_calinski_harabasz_score"),
                "avg_dunn_index": average_group_score(classical_algorithms, "avg_dunn_index"),
            },
            "deep": {
                "avg_quality_score": average_group_score(deep_algorithms, "avg_quality_score"),
                "avg_naturalness_score": average_group_score(deep_algorithms, "avg_naturalness_score"),
                "avg_separation_score": average_group_score(deep_algorithms, "avg_separation_score"),
                "avg_silhouette_score": average_group_score(deep_algorithms, "avg_silhouette_score"),
                "avg_davies_bouldin_score": average_group_score(deep_algorithms, "avg_davies_bouldin_score"),
                "avg_calinski_harabasz_score": average_group_score(deep_algorithms, "avg_calinski_harabasz_score"),
                "avg_dunn_index": average_group_score(deep_algorithms, "avg_dunn_index"),
            }
        }

        pearson_correlations = {
            "quality_score": {
                metric_name: compute_pearson_correlation(values["metric"], values["rating"])
                for metric_name, values in quality_pairs.items()
            },
            "naturalness_score": {
                metric_name: compute_pearson_correlation(values["metric"], values["rating"])
                for metric_name, values in naturalness_pairs.items()
            },
            "separation_score": {
                metric_name: compute_pearson_correlation(values["metric"], values["rating"])
                for metric_name, values in separation_pairs.items()
            },
        }

        def best_avg_algorithm(score_field):
            valid = [
                (alg, stats.get(score_field))
                for alg, stats in algorithm_averages.items()
                if stats.get(score_field) is not None
            ]
            if not valid:
                return None
            return max(valid, key=lambda x: x[1])

        def best_correlation_metric(score_type):
            correlations = pearson_correlations.get(score_type, {})
            valid = [
                (metric, value)
                for metric, value in correlations.items()
                if value is not None
            ]
            if not valid:
                return None
            return max(valid, key=lambda x: abs(x[1]))

        classical_algorithms = {
            "kmeans", "mini_batch_kmeans", "gmm", "bgmm",
            "agglomerative", "birch", "dbscan"
        }
        deep_algorithms = {"resnet_kmeans", "resnet_gmm", "dec"}

        def average_group_score(algorithm_ids, score_field):
            values = [
                stats.get(score_field)
                for alg, stats in algorithm_averages.items()
                if alg in algorithm_ids and stats.get(score_field) is not None
            ]
            if not values:
                return None
            return round(sum(values) / len(values), 4)

        conclusions = []

        # Most preferred algorithms
        best_quality = best_avg_algorithm("avg_quality_score")
        if best_quality:
            conclusions.append(
                f"The most preferred algorithm by overall quality was {get_algorithm_label(best_quality[0])} "
                f"with an average quality score of {round(best_quality[1], 2)}."
            )

        best_naturalness = best_avg_algorithm("avg_naturalness_score")
        if best_naturalness:
            conclusions.append(
                f"The highest naturalness rating was achieved by {get_algorithm_label(best_naturalness[0])} "
                f"with an average score of {best_naturalness[1]}."
            )

        best_separation = best_avg_algorithm("avg_separation_score")
        if best_separation:
            conclusions.append(
                f"The strongest separation performance perceived by users was {get_algorithm_label(best_separation[0])} "
                f"with an average score of {best_separation[1]}."
            )

        # Metric alignment
        best_quality_metric = best_correlation_metric("quality_score")
        if best_quality_metric:
            conclusions.append(
                f"The metric most aligned with quality ratings was {get_metric_label(best_quality_metric[0])} "
                f"(Pearson = {round(best_quality_metric[1], 3)})."
            )

        best_naturalness_metric = best_correlation_metric("naturalness_score")
        if best_naturalness_metric:
            conclusions.append(
                f"The metric most aligned with naturalness ratings was {get_metric_label(best_naturalness_metric[0])} "
                f"(Pearson = {best_naturalness_metric[1]})."
            )

        best_separation_metric = best_correlation_metric("separation_score")
        if best_separation_metric:
            conclusions.append(
                f"The metric most aligned with separation ratings was {get_metric_label(best_separation_metric[0])} "
                f"(Pearson = {best_separation_metric[1]})."
            )

        # Classical vs deep comparison
        classical_quality = classical_vs_deep["classical"]["avg_quality_score"]
        deep_quality = classical_vs_deep["deep"]["avg_quality_score"]

        if classical_quality is not None and deep_quality is not None:
            if classical_quality > deep_quality:
                conclusions.append(
                    f"Classical methods were preferred over deep methods in terms of quality "
                    f"({round(classical_quality, 2)} vs {round(deep_quality, 2)})."
                )
            elif deep_quality > classical_quality:
                conclusions.append(
                    f"Deep methods were preferred over classical methods in terms of quality "
                    f"({round(deep_quality, 2)} vs {round(classical_quality, 2)})."
                )
            else:
                conclusions.append(
                    f"Classical and deep methods achieved the same average quality score "
                    f"({round(classical_quality, 2)})."
                )

        # Weak correlation observation
        quality_corrs = [
            abs(v) for v in pearson_correlations.get("quality_score", {}).values()
            if v is not None
        ]

        if quality_corrs:
            strongest_quality_corr = max(quality_corrs)
            if strongest_quality_corr < 0.4:
                conclusions.append(
                    "No metric showed a strong correlation with quality ratings, suggesting that "
                    "human perception is only partially captured by internal clustering metrics."
                )

        if final_ranking:
            conclusions.append(
                f"The highest overall ranked algorithm was {get_algorithm_label(final_ranking[0]['algorithm_id'])} "
                f"with a final composite score of {final_ranking[0]['final_score']}."
            )

        return Response({
            "study_id": study.id,
            "study_title": study.title,
            "total_ratings": ratings.count(),
            "participant_count": participant_count,
            "algorithm_averages": algorithm_averages,
            "pearson_correlations": pearson_correlations,
            "scatter_data": scatter_data,
            "conclusions": conclusions,
            "classical_vs_deep": classical_vs_deep,
            "final_ranking": final_ranking,
        }, status=200)