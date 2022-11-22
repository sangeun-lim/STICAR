from django.urls import include, path
from rest_framework import routers
from .views import CarViewSet,CarViewSet2

router = routers.DefaultRouter()
router.register(r'car', CarViewSet)
# router.register(r'car/brand', CarViewSet2)
# router.register(r'species', SpeciesViewSet)

urlpatterns = [
   path('', include(router.urls)),
]