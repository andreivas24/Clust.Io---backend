from django.db import models
from django.conf import settings

class UserProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile"
    )
    profile_image = models.ImageField(upload_to="profile_images/", null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} Profile"


class DatasetImage(models.Model):
    category = models.CharField(max_length=100)
    filename = models.CharField(max_length=255, unique=True)
    relative_path = models.CharField(max_length=500)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.category} - {self.filename}"


class BenchmarkSession(models.Model):
    BENCHMARK_MODES = [
        ("full", "Full Dataset"),
        ("subset20", "20 Image Subset"),
        ("uploaded", "Uploaded Images"),
    ]

    created_at = models.DateTimeField(auto_now_add=True)
    benchmark_mode = models.CharField(max_length=50, choices=BENCHMARK_MODES, default="full")
    selected_algorithms = models.JSONField(default=list)
    selected_categories = models.JSONField(default=list)
    parameters = models.JSONField(default=dict)

    def __str__(self):
        return f"BenchmarkSession #{self.id} ({self.benchmark_mode})"


class BenchmarkResult(models.Model):
    original_image_path = models.CharField(max_length=500, blank=True, default="")
    clustered_image_path = models.CharField(max_length=500, blank=True, default="")
    segmentation_map_path = models.CharField(max_length=500, blank=True, default="")

    session = models.ForeignKey(BenchmarkSession, on_delete=models.CASCADE, related_name="results")
    dataset_image = models.ForeignKey(DatasetImage, on_delete=models.CASCADE, related_name="benchmark_results")

    algorithm_id = models.CharField(max_length=100)
    algorithm_type = models.CharField(max_length=50, blank=True, default="")
    is_patch_based = models.BooleanField(default=False)

    processing_time = models.FloatField(null=True, blank=True)

    silhouette_score = models.FloatField(null=True, blank=True)
    davies_bouldin_score = models.FloatField(null=True, blank=True)
    inertia = models.FloatField(null=True, blank=True)
    calinski_harabasz_score = models.FloatField(null=True, blank=True)
    dunn_index = models.FloatField(null=True, blank=True)

    fastest_for_image = models.BooleanField(default=False)

    parameters_used = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True, default="")

    def __str__(self):
        return f"{self.algorithm_id} on {self.dataset_image.filename}"


class UserStudySession(models.Model):
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    benchmark_session = models.OneToOneField(
        BenchmarkSession,
        on_delete=models.CASCADE,
        related_name="user_studies",
        null=True,
        blank=True,
    )

    def __str__(self):
        return self.title


class UserImageRating(models.Model):
    study_session = models.ForeignKey(UserStudySession, on_delete=models.CASCADE, related_name="ratings")
    benchmark_result = models.ForeignKey(BenchmarkResult, on_delete=models.CASCADE, related_name="ratings")

    participant_id = models.CharField(max_length=100)
    quality_score = models.IntegerField()
    naturalness_score = models.IntegerField(null=True, blank=True)
    separation_score = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.participant_id} -> {self.benchmark_result.algorithm_id}"