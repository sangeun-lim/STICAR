from rest_framework import viewsets
from .serializers import CarSerializer
from .models import TbCar
from rest_framework.views import APIView
from rest_framework.response import Response

class CarViewSet(viewsets.ModelViewSet):
   queryset = TbCar.objects.all()
   serializer_class = CarSerializer

class CarViewSet2(APIView):
   queryset = TbCar.objects.filter(manufacturer='현대')
   serializer_class = CarSerializer