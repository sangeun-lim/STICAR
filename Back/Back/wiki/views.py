from django.shortcuts import render

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http.response import HttpResponse
from .models import TbCar as Car, VwWikiLarge
from .serializers import CarSerializer , BrandSerializer, WikiSerializer

@api_view(['GET'])
def all(request):
    Cars = Car.objects.all()
    serailized_Cars= CarSerializer(Cars, many=True)
    return Response(serailized_Cars.data)

@api_view(['GET'])
def large(request):
    Cars = Car.objects.filter(manufacturer='현대')
    serailized_Cars= BrandSerializer(Cars, many=True)
    return Response(serailized_Cars.data)

@api_view(['GET'])
def laa(request):
    brand = request.GET['brand']
    # print('\n\n',brand,'\n\n')
    Cars = Car.objects.filter(manufacturer='현대')
    serailized_Cars= BrandSerializer(Cars, many=True)
    return Response(serailized_Cars.data)

@api_view(['GET'])
def allbrands(request):
    Cars = VwWikiLarge.objects.all()
    serailized_Cars= WikiSerializer(Cars, many=True)
    return Response(serailized_Cars.data) 