from pathlib import Path
from PIL import Image
import random
import json

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
OUTPUT_PATH = PROJECT_ROOT / "Dataset" / "subset_manifest_20.json"

CATEGORIES = ["Animals", "Flowers", "Landscapes", "People"]
SUBSET_SIZE = 20


def get_image_info(image_path: Path):
    with Image.open(image_path) as img:
        width, height = img.size

    total_pixels = width * height
    aspect_ratio = round(width / height, 4) if height else 0

    return {
        "filename": image_path.name,
        "path": str(image_path),
        "width": width,
        "height": height,
        "total_pixels": total_pixels,
        "aspect_ratio": aspect_ratio,
    }


def split_by_resolution(images):
    sorted_images = sorted(images, key=lambda x: x["total_pixels"])
    n = len(sorted_images)

    small = sorted_images[: n // 3]
    medium = sorted_images[n // 3 : 2 * n // 3]
    large = sorted_images[2 * n // 3 :]

    return small, medium, large


def sample_group(group, k):
    if len(group) <= k:
        return group[:]
    return random.sample(group, k)


def build_subset_for_category(category_name):
    category_path = DATASET_ROOT / category_name

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        p for p in category_path.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    images = [get_image_info(p) for p in image_files]

    if len(images) < SUBSET_SIZE:
        raise ValueError(f"{category_name} has fewer than {SUBSET_SIZE} images.")

    small, medium, large = split_by_resolution(images)

    selected = []
    selected.extend(sample_group(small, 7))
    selected.extend(sample_group(medium, 7))
    selected.extend(sample_group(large, 6))

    # dacă lipsesc imagini dintr-o grupă, completează din rest
    if len(selected) < SUBSET_SIZE:
        selected_names = {item["filename"] for item in selected}
        remaining = [img for img in images if img["filename"] not in selected_names]
        needed = SUBSET_SIZE - len(selected)
        selected.extend(sample_group(remaining, needed))

    # sortare finală pentru consistență
    selected = sorted(selected, key=lambda x: x["filename"])

    return selected


def main():
    manifest = {}

    for category in CATEGORIES:
        manifest[category] = build_subset_for_category(category)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Subset manifest saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()