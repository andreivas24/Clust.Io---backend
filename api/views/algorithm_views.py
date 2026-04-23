from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

from ..parameter_suggester import analyze_image_complexity, suggest_parameters_from_analysis
from ..meta_selector import select_best_k_candidate, select_best_dbscan_candidate
from ..services.algorithm_dispatcher import run_algorithm_dispatch, run_k_search_dispatch, run_dbscan_search_dispatch

class SuggestParametersView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        image = request.FILES.get("image")

        if not image:
            return Response({"detail": "Image is required."}, status=400)

        try:
            analysis = analyze_image_complexity(image)
            image.seek(0)

            suggestions = suggest_parameters_from_analysis(analysis)

            return Response({
                "image_analysis": analysis,
                "suggestions": {
                    "n_clusters": suggestions["n_clusters"],
                    "downsample_enabled": suggestions["downsample_enabled"],
                    "downsample_size": suggestions["downsample_size"],
                    "patch_size": suggestions["patch_size"],
                    "dbscan_eps": suggestions["dbscan_eps"],
                    "dbscan_min_samples": suggestions["dbscan_min_samples"],
                    "max_epochs": suggestions["max_epochs"],
                },
                "notes": suggestions["notes"],
            }, status=200)

        except Exception as exc:
            return Response({"detail": str(exc)}, status=500)
        
class RunAlgorithmView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        image = request.FILES.get("image")
        algorithm_id = request.data.get("algorithm_id")

        downsample_enabled = str(request.data.get("downsample_enabled", "false")).lower() == "true"
        downsample_size = int(request.data.get("downsample_size", 256))

        if not image:
            return Response({"detail": "Image file is required."}, status=400)

        if not algorithm_id:
            return Response({"detail": "algorithm_id is required."}, status=400)

        supported_algorithms = [
            "kmeans",
            "mini_batch_kmeans",
            "gmm",
            "bgmm",
            "agglomerative",
            "birch",
            "dbscan",
        ]

        if algorithm_id not in supported_algorithms:
            return Response(
                {"detail": f"Algorithm '{algorithm_id}' is not implemented yet."},
                status=400
            )

        try:
            params = {
                "n_clusters": int(request.data.get("n_clusters", 5)),
                "max_iter": int(request.data.get("max_iter", 300)),
                "init": request.data.get("init", "k-means++"),
                "batch_size": int(request.data.get("batch_size", 256)),
                "gmm_covariance_type": request.data.get("covariance_type", "full"),
                "agglomerative_linkage": request.data.get("linkage", "ward"),
                "agglomerative_metric": request.data.get("metric", "euclidean"),
                "birch_threshold": float(request.data.get("threshold", 0.5)),
                "birch_branching_factor": int(request.data.get("branching_factor", 50)),
                "dbscan_eps": float(request.data.get("eps", 5.0)),
                "dbscan_min_samples": int(request.data.get("min_samples", 5)),
                "dbscan_metric": request.data.get("metric", "euclidean"),
                "patch_size": int(request.data.get("patch_size", 16)),
                "latent_dim": int(request.data.get("latent_dim", 32)),
                "max_epochs": int(request.data.get("max_epochs", 20)),
                "backbone_model": request.data.get("backbone_model", "resnet50"),
                "feature_layer": request.data.get("feature_layer", "avgpool"),
                "downsample_enabled": downsample_enabled,
                "downsample_size": downsample_size,
            }

            result = run_algorithm_dispatch(
                algorithm_id=algorithm_id,
                image_file=image,
                params=params,
            )

            return Response({
                "algorithm": algorithm_id,
                "processing_time": result["processing_time"],
                "width": result["width"],
                "height": result["height"],
                "total_pixels": result["total_pixels"],
                "parameters_used": result["parameters_used"],
                "cluster_distribution": result["cluster_distribution"],
                "cluster_centers": result["cluster_centers"],
                "clustered_image": result["clustered_image"],
                "metrics": result["metrics"],
                "segmentation_map_image": result["segmentation_map_image"],
                "original_width": result["original_width"],
                "original_height": result["original_height"],
                "processed_width": result["processed_width"],
                "processed_height": result["processed_height"],
                "pixel_scatter_sample": result["pixel_scatter_sample"],
                "dbscan_analysis": result.get("dbscan_analysis"),
            }, status=200)

        except ValueError as exc:
            return Response({"detail": f"Invalid numeric parameter values: {str(exc)}"}, status=400)

        except Exception as exc:
            return Response({"detail": f"Error while running algorithm: {str(exc)}"}, status=500)
        
class SuggestBestKView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        image = request.FILES.get("image")
        algorithm_id = request.data.get("algorithm_id")

        if not image:
            return Response({"detail": "Image is required."}, status=400)

        if not algorithm_id:
            return Response({"detail": "algorithm_id is required."}, status=400)

        supported_algorithms = {
            "kmeans",
            "mini_batch_kmeans",
            "gmm",
            "bgmm",
            "agglomerative",
            "birch",
            "resnet_kmeans",
            "resnet_gmm",
            "dec",
        }

        if algorithm_id not in supported_algorithms:
            return Response(
                {"detail": f"Automatic K suggestion is not supported for {algorithm_id}."},
                status=400
            )

        try:
            k_min = int(request.data.get("k_min", 2))
            k_max = int(request.data.get("k_max", 10))

            if k_min >= k_max:
                return Response({"detail": "k_min must be smaller than k_max."}, status=400)

            params = {
                "downsample_enabled": str(request.data.get("downsample_enabled", "false")).lower() == "true",
                "downsample_size": int(request.data.get("downsample_size", 256)),
                "patch_size": int(request.data.get("patch_size", 16)),
                "max_epochs": int(request.data.get("max_epochs", 30)),
                "latent_dim": int(request.data.get("latent_dim", 32)),
                "gmm_covariance_type": request.data.get("gmm_covariance_type", "diag"),

                "birch_threshold": float(request.data.get("birch_threshold", 0.5)),
                "birch_branching_factor": int(request.data.get("birch_branching_factor", 50)),

                "agglomerative_linkage": request.data.get("agglomerative_linkage", "ward"),
                "agglomerative_metric": request.data.get("agglomerative_metric", "euclidean"),
            }
        except ValueError as exc:
            return Response({"detail": f"Invalid numeric parameter values: {str(exc)}"}, status=400)

        tested_results = []

        for k_value in range(k_min, k_max + 1):
            try:
                image.seek(0)

                result = run_k_search_dispatch(
                    algorithm_id=algorithm_id,
                    image_file=image,
                    params=params,
                    k_value=k_value,
                )

                metrics = result.get("metrics", {})

                tested_results.append({
                    "k": k_value,
                    "processing_time": result.get("processing_time"),
                    "silhouette_score": metrics.get("silhouette_score"),
                    "davies_bouldin_score": metrics.get("davies_bouldin_score"),
                    "inertia": metrics.get("inertia"),
                })

            except Exception as exc:
                tested_results.append({
                    "k": k_value,
                    "processing_time": None,
                    "silhouette_score": None,
                    "davies_bouldin_score": None,
                    "inertia": None,
                    "error": str(exc),
                })

        try:
            image.seek(0)
            analysis = analyze_image_complexity(image)
            image.seek(0)
            complexity_level = analysis.get("complexity_level", "medium")
        except Exception:
            complexity_level = "medium"

        # 1. Combined meta-selection
        best_item, selection_reason = select_best_k_candidate(
            tested_results,
            complexity_level=complexity_level
        )

        if best_item is not None:
            return Response({
                "algorithm_id": algorithm_id,
                "best_k": best_item["k"],
                "selection_reason": selection_reason,
                "tested_results": tested_results,
            }, status=200)

        # 2. Fallback to silhouette
        valid_silhouette = [
            item for item in tested_results
            if item["silhouette_score"] is not None
        ]

        if valid_silhouette:
            best_item = max(valid_silhouette, key=lambda x: x["silhouette_score"])
            return Response({
                "algorithm_id": algorithm_id,
                "best_k": best_item["k"],
                "selection_reason": "Fallback selection using highest silhouette score.",
                "tested_results": tested_results,
            }, status=200)

        # 3. Fallback to Davies-Bouldin
        valid_db = [
            item for item in tested_results
            if item["davies_bouldin_score"] is not None
        ]

        if valid_db:
            best_item = min(valid_db, key=lambda x: x["davies_bouldin_score"])
            return Response({
                "algorithm_id": algorithm_id,
                "best_k": best_item["k"],
                "selection_reason": "Fallback selection using lowest Davies-Bouldin score.",
                "tested_results": tested_results,
            }, status=200)

        return Response({
            "algorithm_id": algorithm_id,
            "tested_results": tested_results,
            "best_k": None,
            "selection_reason": "No valid clustering quality scores could be computed.",
        }, status=200)
    
class SuggestBestDbscanParamsView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        image = request.FILES.get("image")

        if not image:
            return Response({"detail": "Image is required."}, status=400)

        try:
            eps_candidates_raw = request.data.get("eps_candidates", "5,7,10,12,15")
            min_samples_candidates_raw = request.data.get("min_samples_candidates", "3,5,10")

            eps_candidates = [float(x.strip()) for x in eps_candidates_raw.split(",") if x.strip()]
            min_samples_candidates = [int(x.strip()) for x in min_samples_candidates_raw.split(",") if x.strip()]

            params = {
                "downsample_enabled": str(request.data.get("downsample_enabled", "false")).lower() == "true",
                "downsample_size": int(request.data.get("downsample_size", 256)),
                "dbscan_metric": request.data.get("dbscan_metric", "euclidean"),
            }
        except ValueError as exc:
            return Response({"detail": f"Invalid DBSCAN search values: {str(exc)}"}, status=400)

        tested_results = []

        for eps_value in eps_candidates:
            for min_samples_value in min_samples_candidates:
                try:
                    image.seek(0)

                    result = run_dbscan_search_dispatch(
                        image_file=image,
                        params=params,
                        eps_value=eps_value,
                        min_samples_value=min_samples_value,
                    )

                    metrics = result.get("metrics", {})
                    dbscan_analysis = result.get("dbscan_analysis", {})

                    tested_results.append({
                        "eps": eps_value,
                        "min_samples": min_samples_value,
                        "processing_time": result.get("processing_time"),
                        "silhouette_score": metrics.get("silhouette_score"),
                        "davies_bouldin_score": metrics.get("davies_bouldin_score"),
                        "noise_ratio": dbscan_analysis.get("noise_ratio"),
                        "detected_clusters": dbscan_analysis.get("n_clusters"),
                    })

                except Exception as exc:
                    tested_results.append({
                        "eps": eps_value,
                        "min_samples": min_samples_value,
                        "processing_time": None,
                        "silhouette_score": None,
                        "davies_bouldin_score": None,
                        "noise_ratio": None,
                        "detected_clusters": None,
                        "error": str(exc),
                    })

        # 1. Combined DBSCAN meta-selection
        best_item, selection_reason = select_best_dbscan_candidate(tested_results)

        if best_item is not None:
            best_params = {
                "eps": best_item["eps"],
                "min_samples": best_item["min_samples"],
            }
            return Response({
                "best_params": best_params,
                "selection_reason": selection_reason,
                "tested_results": tested_results,
            }, status=200)

        # 2. Fallback to silhouette
        valid_silhouette = [
            item for item in tested_results
            if item["silhouette_score"] is not None
        ]

        if valid_silhouette:
            best_item = max(valid_silhouette, key=lambda x: x["silhouette_score"])
            best_params = {
                "eps": best_item["eps"],
                "min_samples": best_item["min_samples"],
            }
            return Response({
                "best_params": best_params,
                "selection_reason": "Fallback selection using highest silhouette score.",
                "tested_results": tested_results,
            }, status=200)

        # 3. Fallback to Davies-Bouldin
        valid_db = [
            item for item in tested_results
            if item["davies_bouldin_score"] is not None
        ]

        if valid_db:
            best_item = min(valid_db, key=lambda x: x["davies_bouldin_score"])
            best_params = {
                "eps": best_item["eps"],
                "min_samples": best_item["min_samples"],
            }
            return Response({
                "best_params": best_params,
                "selection_reason": "Fallback selection using lowest Davies-Bouldin score.",
                "tested_results": tested_results,
            }, status=200)

        return Response({
            "best_params": None,
            "selection_reason": "No valid DBSCAN quality scores could be computed.",
            "tested_results": tested_results,
        }, status=200)
    
class SuggestBatchParametersView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        images = request.FILES.getlist("images")

        if not images:
            return Response({"detail": "At least one image is required."}, status=400)

        try:
            analyses = []

            for image in images:
                analysis = analyze_image_complexity(image)
                analyses.append({
                    "image_name": image.name,
                    **analysis,
                })
                image.seek(0)

            total_images = len(analyses)

            average_width = round(sum(item["width"] for item in analyses) / total_images)
            average_height = round(sum(item["height"] for item in analyses) / total_images)
            average_estimated_colors = round(
                sum(item["estimated_unique_colors"] for item in analyses) / total_images
            )
            average_color_variance = round(
                sum(item["color_variance"] for item in analyses) / total_images, 4
            )

            max_width = max(item["width"] for item in analyses)
            max_height = max(item["height"] for item in analyses)
            max_total_pixels = max(item["total_pixels"] for item in analyses)

            complexity_counts = {"low": 0, "medium": 0, "high": 0}
            for item in analyses:
                complexity_counts[item["complexity_level"]] += 1

            dominant_complexity = max(complexity_counts, key=complexity_counts.get)

            batch_analysis = {
                "total_images": total_images,
                "average_width": average_width,
                "average_height": average_height,
                "max_width": max_width,
                "max_height": max_height,
                "max_total_pixels": max_total_pixels,
                "average_estimated_colors": average_estimated_colors,
                "average_color_variance": average_color_variance,
                "dominant_complexity": dominant_complexity,
            }

            # Construim o analiză "sintetică" pentru batch
            synthetic_analysis = {
                "width": max_width,
                "height": max_height,
                "total_pixels": max_total_pixels,
                "estimated_unique_colors": average_estimated_colors,
                "color_variance": average_color_variance,
                "complexity_level": dominant_complexity,
            }

            suggestions = suggest_parameters_from_analysis(synthetic_analysis)

            notes = list(suggestions["notes"])

            notes.insert(
                0,
                f"Batch analysis completed for {total_images} image(s). Dominant complexity: {dominant_complexity}."
            )

            if total_images > 1:
                notes.append(
                    "Suggested parameters were generated to balance performance and quality across the entire uploaded batch."
                )

            return Response({
                "mode": "batch",
                "batch_analysis": batch_analysis,
                "per_image_analysis": analyses,
                "suggestions": {
                    "n_clusters": suggestions["n_clusters"],
                    "downsample_enabled": suggestions["downsample_enabled"],
                    "downsample_size": suggestions["downsample_size"],
                    "patch_size": suggestions["patch_size"],
                    "dbscan_eps": suggestions["dbscan_eps"],
                    "dbscan_min_samples": suggestions["dbscan_min_samples"],
                    "max_epochs": suggestions["max_epochs"],
                },
                "notes": notes,
            }, status=200)

        except Exception as exc:
            return Response({"detail": str(exc)}, status=500)