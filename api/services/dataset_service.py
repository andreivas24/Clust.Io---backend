from PIL import Image
from ..models import DatasetImage

def get_or_create_dataset_image_from_path(relative_path: str, category: str, absolute_path):
    """
    Creates or retrieves a DatasetImage entry for a server-side dataset image.
    """
    filename = absolute_path.name

    try:
        with Image.open(absolute_path) as img:
            width, height = img.size
    except Exception:
        width, height = None, None

    dataset_image, _ = DatasetImage.objects.get_or_create(
        filename=relative_path,
        defaults={
            "category": category,
            "relative_path": relative_path,
            "width": width,
            "height": height,
        }
    )

    # optional sync if empty
    updated = False
    if dataset_image.category != category:
        dataset_image.category = category
        updated = True
    if dataset_image.relative_path != relative_path:
        dataset_image.relative_path = relative_path
        updated = True
    if dataset_image.width is None and width is not None:
        dataset_image.width = width
        updated = True
    if dataset_image.height is None and height is not None:
        dataset_image.height = height
        updated = True

    if updated:
        dataset_image.save()

    return dataset_image

def get_or_create_uploaded_dataset_image(image_file):
    """
    Stores metadata for uploaded benchmark images as DatasetImage entries.
    """
    filename = image_file.name
    relative_path = f"uploaded/{filename}"

    try:
        image_file.seek(0)
        with Image.open(image_file) as img:
            width, height = img.size
        image_file.seek(0)
    except Exception:
        width, height = None, None

    dataset_image, _ = DatasetImage.objects.get_or_create(
        filename=relative_path,
        defaults={
            "category": "uploaded",
            "relative_path": relative_path,
            "width": width,
            "height": height,
        }
    )

    updated = False
    if dataset_image.width is None and width is not None:
        dataset_image.width = width
        updated = True
    if dataset_image.height is None and height is not None:
        dataset_image.height = height
        updated = True
    if updated:
        dataset_image.save()

    return dataset_image

def register_dataset_original_image_path(relative_path):
    """
    Reuses existing dataset image path as original image path.
    """
    return relative_path.replace("\\", "/")