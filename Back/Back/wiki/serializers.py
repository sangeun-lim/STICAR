from dataclasses import field, fields
from rest_framework import serializers
from .models import TbCar as Car, VwWikiLarge

class CarSerializer(serializers.ModelSerializer):
   class Meta:
       model = Car
       fields = ('car_id', 
                 'car_name', 
                 'manufacturer', 
                 'price', 
                 'type', 
                 'efficiency'
                 )
       
class BrandSerializer(serializers.ModelSerializer):
    class Meta:
        model = Car
        fields = (
            'car_id', 
            'car_name', 
            'manufacturer', 
            'price', 
            'type', 
            'efficiency'
        )
        
class WikiSerializer(serializers.ModelSerializer):
    class Meta:
        model = VwWikiLarge
        fields = (
            'manufacturer',
            'total',
            'inwiki'
        )