from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.conf import settings

BASE_URL = getattr(settings, 'BASE_URL', None)


class UserManager(BaseUserManager):
	# 일반 유저 생성
    # def create_user(self, email, password):
    #     if not email:
    #         raise ValueError(('The Email must be set'))
    #     # if not name:
    #     #     raise ValueError(('The Name must be set'))
    #     # if not nickname:
    #     #     raise ValueError(('The nickname must be set'))
    #     user = self.model(
    #         email= self.normalize_email(email)
    #        )
    #     user.set_password(password)
    #     user.save(using=self._db)
    #     return user
    def create_user(self, email, password, name, nickname):
        if not email:
            raise ValueError(('The Email must be set'))
        if not name:
            raise ValueError(('The Name must be set'))
        if not nickname:
            raise ValueError(('The nickname must be set'))
        user = self.model(
            email= self.normalize_email(email)
            ,name=name,
            nickname=nickname)
        user.set_password(password)
        user.save(using=self._db)
        return user

	# 관리자 생성
    def create_superuser(self, email, password,**extra_fields):
        user = self.create_user(email, password = password)
        user.is_admin = True
        user.save(using=self._db)
        return user

class User(AbstractBaseUser):
	email = models.EmailField(max_length=30, unique=True, null=False, blank=False)
	name = models.CharField(max_length=20, null=False)
	nickname = models.CharField(max_length=50, null=False)
	profile_img_url = models.CharField(max_length=100, default=f"{BASE_URL}media/default_profile.jpg")  # 장고 외부 url 프로필 이미지용
	point = models.IntegerField(default=0)
	is_admin = models.BooleanField(default=False)
	is_active = models.BooleanField(default=False)

	objects = UserManager()

	USERNAME_FIELD = 'email'
	REQUIRED_FIELD = []

	def __str__(self):
		return self.email


# class Meta:
# 	db_table = 'user'
