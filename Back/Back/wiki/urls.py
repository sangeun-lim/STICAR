from django.urls import path, include
from . import views

app_name = 'posts'

urlpatterns = [
    # path('', views.index),
    path('wiki/get',views.all),
    path('wiki/post', views.large),
    path('wiki/postt', views.laa),
    path('wiki/userwiki', views.allbrands),
]