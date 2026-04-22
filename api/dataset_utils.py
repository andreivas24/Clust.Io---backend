from pathlib import Path
import json

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_dataset_root():
    return Path(__file__).resolve().parents[2] / "Dataset"


def get_subset_manifest_path():
    return get_dataset_root() / "subset_manifest_20.json"


def build_manifest():
    dataset_root = get_dataset_root()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found at: {dataset_root}")

    manifest = []

    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        for image_path in sorted(category_dir.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            manifest.append({
                "id": image_path.stem,
                "filename": image_path.name,
                "relative_path": f"{category}/{image_path.name}",
                "category": category,
                "source": "local_dataset",
                "notes": "",
            })

    return manifest


def get_manifest_summary():
    manifest = build_manifest()

    category_counts = {}
    for item in manifest:
        category = item["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "total_images": len(manifest),
        "categories": category_counts,
        "items": manifest,
    }


def load_subset_manifest():
    subset_manifest_path = get_subset_manifest_path()

    if not subset_manifest_path.exists():
        raise FileNotFoundError(f"Subset manifest not found at: {subset_manifest_path}")

    with open(subset_manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_paths(categories=None, max_images=None, use_subset=False):
    dataset_root = get_dataset_root()

    if use_subset:
        subset_manifest = load_subset_manifest()
        results = []

        selected_categories = categories if categories else list(subset_manifest.keys())

        for category in selected_categories:
            if category not in subset_manifest:
                continue

            for item in subset_manifest[category]:
                filename = item["filename"]
                abs_path = dataset_root / category / filename

                if abs_path.exists():
                    results.append({
                        "id": Path(filename).stem,
                        "filename": filename,
                        "relative_path": f"{category}/{filename}",
                        "category": category,
                        "absolute_path": abs_path,
                    })

        if max_images is not None:
            results = results[:max_images]

        return results

    manifest = build_manifest()
    filtered = manifest

    if categories:
        categories_set = set(categories)
        filtered = [item for item in filtered if item["category"] in categories_set]

    if max_images is not None:
        filtered = filtered[:max_images]

    results = []
    for item in filtered:
        abs_path = dataset_root / item["category"] / item["filename"]

        if abs_path.exists():
            results.append({
                "id": item["id"],
                "filename": item["filename"],
                "relative_path": item["relative_path"],
                "category": item["category"],
                "absolute_path": abs_path,
            })

    return results