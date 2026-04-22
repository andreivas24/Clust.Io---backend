import math


def normalize(values, reverse=False):
    if not values:
        return []

    vmin = min(values)
    vmax = max(values)

    if math.isclose(vmin, vmax):
        return [1.0 for _ in values]

    normalized = []
    for value in values:
        score = (value - vmin) / (vmax - vmin)
        if reverse:
            score = 1.0 - score
        normalized.append(score)

    return normalized


def get_semantic_k_prior(k, complexity_level):
    """
    Returns a semantic plausibility prior in [0,1]
    depending on image complexity and K value.
    """

    if complexity_level == "low":
        preferred = {3: 1.0, 4: 1.0, 5: 0.9, 2: 0.65, 6: 0.7, 7: 0.4, 8: 0.2, 9: 0.1, 10: 0.0}
    elif complexity_level == "medium":
        preferred = {4: 1.0, 5: 1.0, 6: 0.95, 3: 0.75, 7: 0.75, 2: 0.45, 8: 0.55, 9: 0.3, 10: 0.1}
    else:  # high
        preferred = {5: 1.0, 6: 1.0, 7: 0.95, 8: 0.85, 4: 0.8, 3: 0.45, 2: 0.2, 9: 0.65, 10: 0.45}

    return preferred.get(k, 0.3)


def select_best_k_candidate(tested_results, complexity_level="medium"):
    """
    Combined meta-score:
      40% silhouette
      20% Davies-Bouldin
      5% processing time
      35% semantic prior

    Also:
    - discourages K=2 for medium/high complexity images
    - prefers slightly larger K when scores are close
    """

    candidates = [
        item for item in tested_results
        if item.get("silhouette_score") is not None
        and item.get("davies_bouldin_score") is not None
        and item.get("processing_time") is not None
    ]

    if not candidates:
        return None, "No candidates had all metrics required for combined meta-selection."

    silhouette_values = [item["silhouette_score"] for item in candidates]
    db_values = [item["davies_bouldin_score"] for item in candidates]
    time_values = [item["processing_time"] for item in candidates]
    prior_values = [get_semantic_k_prior(item["k"], complexity_level) for item in candidates]

    sil_norm = normalize(silhouette_values, reverse=False)
    db_norm = normalize(db_values, reverse=True)
    time_norm = normalize(time_values, reverse=True)

    best_item = None
    best_score = -999999

    for index, item in enumerate(candidates):
        meta_score = (
            0.40 * sil_norm[index]
            + 0.20 * db_norm[index]
            + 0.05 * time_norm[index]
            + 0.35 * prior_values[index]
        )

        # Explicit penalty for trivial K=2 on more complex images
        if complexity_level == "medium" and item["k"] == 2:
            meta_score -= 0.15
        elif complexity_level == "high" and item["k"] == 2:
            meta_score -= 0.25

        item["semantic_prior"] = round(prior_values[index], 4)
        item["meta_score"] = round(meta_score, 4)

        if meta_score > best_score:
            best_score = meta_score
            best_item = item

    # Tolerance rule:
    # if larger K values are very close to the best score,
    # prefer the largest K among those close candidates
    tolerance = 0.05
    close_candidates = [
        item for item in candidates
        if (best_score - item["meta_score"]) <= tolerance
    ]

    if close_candidates:
        best_item = max(close_candidates, key=lambda x: x["k"])

    return best_item, (
        f"Selected using combined meta-score with semantic prior for {complexity_level}-complexity images: "
        "40% silhouette + 20% Davies-Bouldin + 5% processing time + 35% semantic K prior, "
        "with an extra penalty for trivial K=2 solutions on medium/high-complexity images."
    )

def select_best_dbscan_candidate(tested_results):
    """
    Uses a combined meta-score for DBSCAN:
        0.5 * silhouette_norm
      + 0.25 * db_norm
      + 0.15 * time_norm
      + 0.10 * noise_quality

    Lower noise ratio is better.
    Candidates with 0 detected clusters are deprioritized if possible.
    """

    candidates = [
        item for item in tested_results
        if item.get("silhouette_score") is not None
        and item.get("davies_bouldin_score") is not None
        and item.get("processing_time") is not None
    ]

    if not candidates:
        return None, "No candidates had all metrics required for combined DBSCAN meta-selection."

    non_zero_cluster_candidates = [
        item for item in candidates
        if (item.get("detected_clusters") or 0) > 0
    ]

    if non_zero_cluster_candidates:
        candidates = non_zero_cluster_candidates

    silhouette_values = [item["silhouette_score"] for item in candidates]
    db_values = [item["davies_bouldin_score"] for item in candidates]
    time_values = [item["processing_time"] for item in candidates]
    noise_values = [
        item["noise_ratio"] if item.get("noise_ratio") is not None else 1.0
        for item in candidates
    ]

    sil_norm = normalize(silhouette_values, reverse=False)
    db_norm = normalize(db_values, reverse=True)
    time_norm = normalize(time_values, reverse=True)
    noise_norm = normalize(noise_values, reverse=True)  # lower noise = better

    best_item = None
    best_score = -1

    for index, item in enumerate(candidates):
        meta_score = (
            0.5 * sil_norm[index]
            + 0.25 * db_norm[index]
            + 0.15 * time_norm[index]
            + 0.10 * noise_norm[index]
        )

        item["meta_score"] = round(meta_score, 4)

        if meta_score > best_score:
            best_score = meta_score
            best_item = item

    return best_item, (
        "Selected using combined DBSCAN meta-score: "
        "50% silhouette + 25% Davies-Bouldin + 15% processing time + 10% noise quality."
    )