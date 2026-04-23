from django.contrib import admin
from django.urls import path
from api.health import healthz

urlpatterns = [
    path("healthz/", healthz),
    path("admin/", admin.site.urls),
]