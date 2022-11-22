from rest_framework import serializers
from .models import TbCar as Car

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