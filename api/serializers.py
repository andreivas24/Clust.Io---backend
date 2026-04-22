from rest_framework import serializers
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    profile_image_url = serializers.SerializerMethodField()
    password = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ["id", "username", "email", "password", "is_staff", "is_active", "profile_image_url"]
        extra_kwargs = { "email": {"required": True} }

    def get_profile_image_url(self, obj):
        request = self.context.get("request")
        profile = getattr(obj, "profile", None)

        if profile and profile.profile_image:
            if request:
                return request.build_absolute_uri(profile.profile_image.url)
            return profile.profile_image.url

        return None
    
    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data.get("email"),
            password=validated_data["password"]
        )
        return user
    
    def update(self, instance, validated_data):
        instance.email = validated_data.get("email", instance.email)

        if "password" in validated_data:
            instance.set_password(validated_data["password"])

        instance.save()
        return instance