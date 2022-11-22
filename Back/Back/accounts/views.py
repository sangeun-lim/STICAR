from urllib import request
from django.shortcuts import render
from django.shortcuts import get_object_or_404, get_list_or_404
from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import check_password
from django.contrib.auth.models import update_last_login
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.views import (
	TokenObtainPairView,
	TokenRefreshView,
)
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.authentication import JWTAuthentication
from .serializers import(
	UserSerializer,
	SignupSerializer,
	IdSerializer,
	PasswordSerializer,
	UserPointSerializer,
	UserProfileSerializer,
	SignInSerializer,
)
from django.http import JsonResponse

# Create your views here.
User = get_user_model()

#회원가입
# @swagger_auto_schema(method='POST', request_body=UserSerializer)
@swagger_auto_schema(
	methods=['POST'],
	request_body=openapi.Schema(
		type=openapi.TYPE_OBJECT,
		properties={
			'email': openapi.Schema(type=openapi.TYPE_STRING, description='이메일'),
			'password': openapi.Schema(type=openapi.TYPE_STRING, description='비밀번호'),
			'passwordconfirm': openapi.Schema(type=openapi.TYPE_STRING, description='비밀번호확인'),
			'name': openapi.Schema(type=openapi.TYPE_STRING, description='이름'),
			'nickname': openapi.Schema(type=openapi.TYPE_STRING, description='닉네임'),
		}
	),
	responses={201: openapi.Response('회원가입 성공'),
			   400: openapi.Response('회원가입 실패')
			   })
@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
	# email = request.data.get('email')
	password = request.data.get('password')
	password_confirm = request.data.get('password_confirm')
	# name = request.data.get('name')
	# nickname = request.data.get('nickname')


	# 이메일(아이디) 일치여부 검사
	# if User.objects.filter(email=request.get('email')).exists():
	if User.objects.filter(email=request.data.get('email')).exists():
		return Response({'error: 이미 가입된 아이디 입니다.'},status=status.HTTP_400_BAD_REQUEST)

	# 비밀번호 확인 일치여부 검사
	if password != password_confirm:
		return Response({'error: 비밀번호 확인이 일치하지 않습니다.'},status=status.HTTP_400_BAD_REQUEST)

	# 닉네임 일치여부 검사
	if User.objects.filter(nickname=request.data.get('nickname')).exists():
		return Response({'error': '이미 존재하는 닉네임입니다.'}, status=status.HTTP_400_BAD_REQUEST)

	# UserSerializer를 통해 사용자가 넘겨준 데이터 직렬화
	serializer = SignupSerializer(data=request.data)
	# user = User.objects.create_user('test4@test.com','test1234', 'test','test4')

	# validation (password도 같이 직렬화)
	if serializer.is_valid(raise_exception=True):
		user = serializer.save()
	# password 해싱 -> password -> 문자열 데이터로, set_password 메서드는 User 객체 저장 x
		user.set_password(request.data.get('password'))
	# 유저 객체 저장
		user.save()
	serializer.is_valid(raise_exception=True)
	return Response(serializer.data, status=status.HTTP_201_CREATED)





# 아이디 중복 확인
@swagger_auto_schema(
  methods=['GET'],
	responses={201: openapi.Response('사용가능한 아이디입니다.'),
			   400: openapi.Response('중복된 아이디입니다.')
			   })
@api_view(['GET'])
@permission_classes([AllowAny])
def idcheck(request):
	if User.objects.filter(email=request.data.get('email')).exists():
		return Response({'error: ID 중복'}, status=status.HTTP_400_BAD_REQUEST)

	else:
		return Response({'사용가능한 ID입니다.'},status=status.HTTP_200_OK)

# 로그인
@swagger_auto_schema(method='POST', request_body=SignInSerializer)
@api_view(['POST'])
@permission_classes([AllowAny])
def signin(request):
	# email = request.data.get('email')
	# password = request.data.get('password')

	user = authenticate(email=request.data.get('email'), password=request.data.get('password'))
	if user is None:
		return Response({'message': '아이디 또는 비밀번호가 일치하지 않습니다.'}, status=status.HTTP_401_UNAUTHORIZED)

	refresh = RefreshToken.for_user(user)
	update_last_login(None, user)

	return Response({'access_token': str(refresh.access_token),
					'refresh_token': str(refresh),}, status=status.HTTP_200_OK)


#회원 탈퇴
@swagger_auto_schema(method='DELETE', request_body=PasswordSerializer)
@api_view(['DELETE'])
def delete(request):
	user = get_object_or_404(User, email=request.data['email'])

	if user == request.user and user.check_password(request.data.get('password')):
		user.delete()
		return Response(status=status.HTTP_204_NO_CONTENT)

	else:
		return Response({'error: 본인 인증 실패'}, status=status.HTTP_401_UNAUTHORIZED)

@permission_classes((IsAuthenticated,))
# 프로필 조회
@swagger_auto_schema(
	methods=['GET'],
	responses={200: openapi.Response('조회 성공', UserProfileSerializer()),
			   400: openapi.Response('조회 실패')
			   })
# 프로필 사진, 닉네임 수정
@swagger_auto_schema(
	methods=['PUT'],
	request_body=openapi.Schema(
		type=openapi.TYPE_OBJECT,
		properties={
			'nickname': openapi.Schema(type=openapi.TYPE_STRING, description='닉네임'),
			'profile_img_url': openapi.Schema(type=openapi.TYPE_FILE, description='프로필사진'),
		}
	),
	responses={201: openapi.Response('수정 성공', UserProfileSerializer()),
			   400: openapi.Response('수정 실패')
			   })
@api_view(['GET', 'PUT'])
def profile(request):

	# 프로필 조회
	if request.method == 'GET':
		serializer = UserProfileSerializer(request.user)
		return Response(serializer.data, status=status.HTTP_200_OK)

	# 프로필 이지미, 닉네임 수정
	elif request.method == 'PUT':

		# mypage 이미지 받기
		profile_img_url = request.data.get('profile_img_url')
		nickname = request.data["nickname"]
		# 닉네임 수정 시 일치여부 검사
		if request.user.nickname != nickname and User.objects.filter(nickname=nickname).exists():
			return Response({'error': '이미 존재하는 별명입니다.'}, status=status.HTTP_400_BAD_REQUEST)

		serializer = UserProfileSerializer(instance=request.user, data=request.data)
		if serializer.is_valid(raise_exception=True):
			serializer.save(profile_img_url=profile_img_url)
			return Response(serializer.data, status=status.HTTP_201_CREATED)
		return Response({'error': '업로드한 파일에 문제가 있습니다.'}, status=status.HTTP_400_BAD_REQUEST)


#비밀번호 변경
@swagger_auto_schema(
	methods=['PUT'],
	request_body=openapi.Schema(
		type=openapi.TYPE_OBJECT,
		properties={
			# 'currentpassword' : openapi.Schema(type=openapi.TYPE_STRING,description='현재 비밀번호'),
			'password': openapi.Schema(type=openapi.TYPE_STRING, description='비밀번호'),
			'passwordconfirm': openapi.Schema(type=openapi.TYPE_STRING, description='비밀번호확인'),
		}
	),
	responses={201: openapi.Response('변경 성공'),
			   400: openapi.Response('변경 실패')
			   })
@api_view(['PUT'])
@permission_classes((IsAuthenticated,))
def password(request):
	# currentpassword = request.data['password']
	password = request.data['password']
	passwordconfirm = request.data['passwordconfirm']

	if password != passwordconfirm:
		return Response({'error': '비밀번호가 일치하지 않습니다.'}, status=status.HTTP_400_BAD_REQUEST)
	else:
		serializer = PasswordSerializer(instance=request.user, data=request.data)
		if serializer.is_valid(raise_exception=True):
			user = serializer.save()
			user.set_password(password)
			user.save()
			return Response({'access': '비밀번호가 변경되었습니다.'}, status=status.HTTP_201_CREATED)
		else:
			return Response({'error': '비정상적인 접근입니다.'}, status=status.HTTP_400_BAD_REQUEST)