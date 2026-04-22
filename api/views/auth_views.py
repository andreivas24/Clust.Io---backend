from django.shortcuts import redirect
from django.contrib.auth import get_user_model
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from rest_framework import generics
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

from allauth.socialaccount.models import SocialAccount, SocialToken
from rest_framework_simplejwt.tokens import RefreshToken

from ..serializers import UserSerializer
from ..models import UserProfile

import json

User = get_user_model()

class UserCreate(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [AllowAny]

class UserDetailView(generics.RetrieveUpdateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user

class UserDashboardView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user = request.user
        profile = getattr(user, "profile", None)

        profile_image_url = None
        if profile and profile.profile_image:
            profile_image_url = request.build_absolute_uri(profile.profile_image.url)

        user_data = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_staff": user.is_staff,
            "is_active": user.is_active,
            "profile_image_url": profile_image_url,
        }

        return Response(user_data)

class ForgotPasswordView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")

        if not email:
            return Response({"detail": "Email is required."}, status=400)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response(
                {"detail": "If an account with this email exists, a reset link has been sent."},
                status=200
            )

        uid = urlsafe_base64_encode(force_bytes(user.pk))
        token = default_token_generator.make_token(user)

        frontend_url = getattr(settings, "FRONTEND_URL", "http://localhost:5173")
        reset_link = f"{frontend_url}/reset-password/{uid}/{token}/"

        send_mail(
            subject="Reset your Clust.io password",
            message=(
                "You requested a password reset.\n\n"
                f"Use the following link to set a new password:\n{reset_link}\n\n"
                "If you did not request this, you can ignore this email."
            ),
            from_email=None,
            recipient_list=[email],
            fail_silently=False,
        )

        return Response(
            {"detail": "If an account with this email exists, a reset link has been sent."},
            status=200
        )

class ResetPasswordView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        uid = request.data.get("uid")
        token = request.data.get("token")
        new_password = request.data.get("new_password")

        if not uid or not token or not new_password:
            return Response(
                {"detail": "uid, token and new_password are required."},
                status=400
            )

        try:
            user_id = urlsafe_base64_decode(uid).decode()
            user = User.objects.get(pk=user_id)
        except Exception:
            return Response({"detail": "Invalid reset link."}, status=400)

        if not default_token_generator.check_token(user, token):
            return Response({"detail": "Invalid or expired token."}, status=400)

        user.set_password(new_password)
        user.save()

        return Response({"detail": "Password reset successful."}, status=200)
    
class UploadProfileImageView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        image = request.FILES.get("profile_image")

        if not image:
            return Response({"detail": "Profile image is required."}, status=400)

        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        profile.profile_image = image
        profile.save()

        image_url = request.build_absolute_uri(profile.profile_image.url) if profile.profile_image else None

        return Response({
            "detail": "Profile image uploaded successfully.",
            "profile_image_url": image_url,
        }, status=200)
    
@login_required
def google_login_callback(request):
    user = request.user
    social_accounts = SocialAccount.objects.filter(user=user)
    print("Social Account for user:", social_accounts)

    social_account = social_accounts.first()

    frontend_url = getattr(settings, "FRONTEND_URL", "http://localhost:5173")

    if not social_account:
        print("No social account for user:", user)
        return redirect(f"{frontend_url}/login/callback/?error=NoSocialAccount")

    token = SocialToken.objects.filter(account=social_account, account__provider='google').first()

    if token:
        print("Google token found:", token.token)
        refresh = RefreshToken.for_user(user)
        access_token = str(refresh.access_token)
        refresh_token = str(refresh)
        return redirect(
            f"{frontend_url}/login/callback/?access_token={access_token}&refresh_token={refresh_token}"
        )
    else:
        print("No Google token found for user:", user)
        return redirect(f"{frontend_url}/login/callback/?error=NoGoogleToken")
    
@csrf_exempt
def validate_google_token(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            google_access_token = data.get('access_token')
            print(google_access_token)

            if not google_access_token:
                return JsonResponse({'detail': 'Access token is missing.'}, status=400)
            return JsonResponse({'valid': True})
        except json.JSONDecodeError:
            return JsonResponse({'detail': 'Invalid JSON.'}, status=400)
    return JsonResponse({'detail': 'Method not allowed.'}, status=405)