def run_algorithm_dispatch(algorithm_id, image_file, params):
    from ..algorithms import (
        process_agglomerative,
        process_bgmm,
        process_birch,
        process_dbscan,
        process_gmm,
        process_kmeans,
        process_mini_batch_kmeans,
    )

    if algorithm_id == "kmeans":
        return process_kmeans(
            image_file=image_file,
            n_clusters=params["n_clusters"],
            max_iter=params.get("max_iter", 300),
            init=params.get("init", "k-means++"),
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "mini_batch_kmeans":
        return process_mini_batch_kmeans(
            image_file=image_file,
            n_clusters=params["n_clusters"],
            batch_size=params.get("batch_size", 256),
            max_iter=params.get("max_iter", 100),
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "gmm":
        return process_gmm(
            image_file=image_file,
            n_components=params["n_clusters"],
            covariance_type=params["gmm_covariance_type"],
            max_iter=params.get("max_iter", 100),
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "agglomerative":
        return process_agglomerative(
            image_file=image_file,
            n_clusters=params["n_clusters"],
            linkage=params["agglomerative_linkage"],
            metric=params["agglomerative_metric"],
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "birch":
        return process_birch(
            image_file=image_file,
            threshold=params["birch_threshold"],
            branching_factor=params["birch_branching_factor"],
            n_clusters=params["n_clusters"],
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "dbscan":
        return process_dbscan(
            image_file=image_file,
            eps=params["dbscan_eps"],
            min_samples=params["dbscan_min_samples"],
            metric=params.get("dbscan_metric", "euclidean"),
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "bgmm":
        return process_bgmm(
            image_file=image_file,
            n_components=params["n_clusters"],
            covariance_type=params["gmm_covariance_type"],
            weight_concentration_prior_type="dirichlet_process",
            max_iter=params.get("max_iter", 200),
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_id}")

def run_k_search_dispatch(algorithm_id, image_file, params, k_value):
    from ..algorithms import (
        process_agglomerative,
        process_bgmm,
        process_birch,
        process_gmm,
        process_kmeans,
        process_mini_batch_kmeans,
    )

    if algorithm_id == "kmeans":
        return process_kmeans(
            image_file=image_file,
            n_clusters=k_value,
            max_iter=params.get("max_iter", 300),
            init=params.get("init", "k-means++"),
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "mini_batch_kmeans":
        return process_mini_batch_kmeans(
            image_file=image_file,
            n_clusters=k_value,
            batch_size=params.get("batch_size", 256),
            max_iter=params.get("max_iter", 100),
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "gmm":
        return process_gmm(
            image_file=image_file,
            n_components=k_value,
            covariance_type=params["gmm_covariance_type"],
            max_iter=params.get("max_iter", 100),
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "agglomerative":
        return process_agglomerative(
            image_file=image_file,
            n_clusters=k_value,
            linkage=params["agglomerative_linkage"],
            metric=params["agglomerative_metric"],
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "birch":
        return process_birch(
            image_file=image_file,
            threshold=params["birch_threshold"],
            branching_factor=params["birch_branching_factor"],
            n_clusters=k_value,
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    elif algorithm_id == "bgmm":
        return process_bgmm(
            image_file=image_file,
            n_components=k_value,
            covariance_type=params["gmm_covariance_type"],
            weight_concentration_prior_type="dirichlet_process",
            max_iter=params.get("max_iter", 200),
            downsample_enabled=params["downsample_enabled"],
            downsample_size=params["downsample_size"],
        )

    else:
        raise ValueError(f"Automatic K selection is not supported for algorithm: {algorithm_id}")


def run_dbscan_search_dispatch(image_file, params, eps_value, min_samples_value):
    from ..algorithms import process_dbscan

    return process_dbscan(
        image_file=image_file,
        eps=eps_value,
        min_samples=min_samples_value,
        metric=params.get("dbscan_metric", "euclidean"),
        downsample_enabled=params["downsample_enabled"],
        downsample_size=params["downsample_size"],
    )