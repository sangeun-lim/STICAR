from django.urls import path
from . import views
from rest_framework_simplejwt.views import (
	TokenObtainPairView,
	TokenRefreshView,
	TokenVerifyView )
# from rest_framework.routers import DefaultRouter

app_name = 'accounts'

urlpatterns =[
	path('signup/', views.signup, name='signup'),
	path('signin/', views.signin, name='signin'),
	path('profile/<user_id>/', views.profile, name='profile'),
	path('delete/', views.delete, name='delete'),
	path('password/', views.password, name='password'),
	path('idcheck/', views.idcheck, name='idcheck'),
 	# 토큰
	path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
	path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
	path('token/verify/', TokenVerifyView.as_view(), name='token_verify'),
]