from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from api.views.algorithm_views import RunAlgorithmView, SuggestBatchParametersView, SuggestBestDbscanParamsView, SuggestBestKView, SuggestParametersView
from api.views.auth_views import ForgotPasswordView, ResetPasswordView, UploadProfileImageView, UserCreate, UserDashboardView, UserDetailView, google_login_callback, validate_google_token
from api.views.benchmark_views import BenchmarkSessionDetailView, BenchmarkSessionListView, DatasetManifestView, RunBenchmarkView, RunDatasetBenchmarkView
from api.views.user_study_views import CreateUserStudySessionView, SubmitUserRatingView, UserStudyAnalyticsView, UserStudyDetailView, UserStudyRatingsView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/user/register/', UserCreate.as_view(), name='user-create'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api-auth/', include('rest_framework.urls')),
    path('accounts/', include('allauth.urls')),
    path('callback/', google_login_callback, name='callback'),
    path('api/auth/user/', UserDetailView.as_view(), name='user_detail'),
    path("api/auth/upload-profile-image/", UploadProfileImageView.as_view(), name="upload-profile-image"),
    path("api/auth/forgot-password/", ForgotPasswordView.as_view()),
    path("api/auth/reset-password/", ResetPasswordView.as_view()),
    path('api/google/validate_token/', validate_google_token, name='validate_token'),
    path('dashboard/', UserDashboardView.as_view(), name='dashboard'),
    path('api/algorithms/run/', RunAlgorithmView.as_view(), name='run-algorithm'),
    path('api/benchmark/run/', RunBenchmarkView.as_view(), name='run-benchmark'),
    path('api/algorithms/suggest-parameters/', SuggestParametersView.as_view(), name='suggest-parameters'),
    path('api/benchmark/suggest-parameters/', SuggestBatchParametersView.as_view(), name='suggest-batch-parameters'),
    path('api/algorithms/suggest-best-k/', SuggestBestKView.as_view(), name='suggest-best-k'),
    path('api/algorithms/suggest-best-dbscan/', SuggestBestDbscanParamsView.as_view(), name='suggest-best-dbscan'),
    path('api/datasets/manifest/', DatasetManifestView.as_view(), name='dataset-manifest'),
    path('api/datasets/run-benchmark/', RunDatasetBenchmarkView.as_view(), name='run-dataset-benchmark'),
    path('api/benchmark/sessions/', BenchmarkSessionListView.as_view(), name='benchmark-session-list'),
    path('api/benchmark/sessions/<int:session_id>/', BenchmarkSessionDetailView.as_view(), name='benchmark-session-detail'),
    path('api/user-studies/', CreateUserStudySessionView.as_view(), name='create-user-study'),
    path('api/user-studies/<int:study_id>/', UserStudyDetailView.as_view(), name='user-study-detail'),
    path('api/user-studies/submit-rating/', SubmitUserRatingView.as_view(), name='submit-user-rating'),
    path('api/user-studies/<int:study_id>/ratings/', UserStudyRatingsView.as_view(), name='user-study-ratings'),
    path('api/user-studies/<int:study_id>/analytics/', UserStudyAnalyticsView.as_view(), name='user-study-analytics'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)