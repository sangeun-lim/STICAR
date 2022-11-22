from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()


class SignupSerializer(serializers.ModelSerializer):
  email = serializers.EmailField(min_length=4, max_length=30)
  password = serializers.CharField(min_length=8, max_length= 20, write_only=True)
  password_confirm = password
  name = serializers.CharField(max_length=20)
  nickname = serializers.CharField(max_length=50)

  class Meta:
    model = User
    # fields = ('email', 'password', 'password_confirm', 'name', 'nickname',)
    fields = '__all__'


class UserSerializer(serializers.ModelSerializer):
	# 응답 데이터로 password가 담겨오지 않도록 write only 속성으로 오버라이딩
	password = serializers.CharField(write_only=True)

	class Meta:
		model = User
		fields = '__all__'

class IdSerializer(serializers.ModelSerializer):
  email = serializers.CharField(min_length=4, max_length=16)
  class Meta:
    model = User
    fields = ('email',)

class PasswordSerializer(UserSerializer):
  class Meta:
    model = User
    fields = ('password', )

class UserProfileSerializer(serializers.ModelSerializer):
	profile_img_url = serializers.ImageField(use_url=True)
	class Meta:
		model = User
		fields = ('email', 'name', 'nickname', 'profile_img_url','point')

class UserPointSerializer(serializers.ModelSerializer):
	class Meta:
		model = User
		fields = ('point')

class SignInSerializer(serializers.Serializer):
    email = serializers.CharField(max_length=255, required=True)
    password = serializers.CharField(max_length=255, required=True, write_only=True)