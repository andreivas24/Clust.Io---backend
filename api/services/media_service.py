import os
import base64
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from django.conf import settings


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_media_paths(folder_relative_path: str, filename: str):
    relative_path = Path(folder_relative_path) / filename
    absolute_path = Path(settings.MEDIA_ROOT) / relative_path
    return relative_path.as_posix(), absolute_path


def save_uploaded_original_image(image_file, session_id):
    extension = os.path.splitext(image_file.name)[1] or ".png"
    filename = f"{uuid.uuid4().hex[:12]}{extension}"

    relative_path, absolute_path = build_media_paths(
        f"benchmark_results/session_{session_id}/originals",
        filename
    )

    ensure_directory(absolute_path.parent)

    image_file.seek(0)
    with open(absolute_path, "wb+") as destination:
        if hasattr(image_file, "chunks"):
            for chunk in image_file.chunks():
                destination.write(chunk)
        else:
            destination.write(image_file.read())
    image_file.seek(0)

    return relative_path


def save_base64_image(base64_data, folder_relative_path, filename_prefix):
    if not base64_data:
        return ""

    filename = f"{filename_prefix}_{uuid.uuid4().hex[:12]}.png"
    relative_path, absolute_path = build_media_paths(folder_relative_path, filename)

    ensure_directory(absolute_path.parent)

    try:
        if isinstance(base64_data, bytes):
            base64_data = base64_data.decode("utf-8")

        if isinstance(base64_data, str) and "," in base64_data and "base64" in base64_data:
            base64_data = base64_data.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_data)

        with open(absolute_path, "wb") as f:
            f.write(image_bytes)

        return relative_path

    except Exception as exc:
        print("Failed to save base64 image:", exc)
        print("Type of base64_data:", type(base64_data))
        print("Preview:", str(base64_data)[:100])
        return ""


def save_cluster_result_image(result, session_id, image_name, algorithm_id):
    results_dir_relative = f"benchmark_results/session_{session_id}/results"
    safe_image_name = os.path.splitext(os.path.basename(image_name))[0]
    filename = f"{safe_image_name}_{algorithm_id}_{uuid.uuid4().hex[:8]}.png"

    relative_path, absolute_path = build_media_paths(results_dir_relative, filename)
    ensure_directory(absolute_path.parent)

    pil_image = None

    if result.get("segmented_image_pil") is not None:
        pil_image = result["segmented_image_pil"]

    elif result.get("segmented_image_array") is not None:
        arr = result["segmented_image_array"]
        if isinstance(arr, np.ndarray):
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(arr)

    if pil_image is None:
        return ""

    pil_image.save(absolute_path)
    return relative_path


def copy_dataset_original_to_media(absolute_path, session_id):
    extension = os.path.splitext(os.path.basename(absolute_path))[1] or ".png"
    filename = f"{uuid.uuid4().hex[:12]}{extension}"

    relative_path, target_absolute_path = build_media_paths(
        f"benchmark_results/session_{session_id}/originals",
        filename
    )

    ensure_directory(target_absolute_path.parent)
    shutil.copy2(absolute_path, target_absolute_path)

    return relative_path