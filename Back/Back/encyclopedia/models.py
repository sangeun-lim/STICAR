from django.db import models

class TbCar(models.Model):
    car_id = models.IntegerField(db_column='CAR_ID', primary_key=True)  # Field name made lowercase.
    car_name = models.CharField(db_column='CAR_NAME', max_length=45, blank=True, null=True)  # Field name made lowercase.
    manufacturer = models.CharField(db_column='MANUFACTURER', max_length=45, blank=True, null=True)  # Field name made lowercase.
    price = models.CharField(db_column='PRICE', max_length=45, blank=True, null=True)  # Field name made lowercase.
    type = models.CharField(db_column='TYPE', max_length=45, blank=True, null=True)  # Field name made lowercase.
    efficiency = models.CharField(db_column='EFFICIENCY', max_length=45, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'TB_CAR'


class TbCarImage(models.Model):
    car_img_id = models.IntegerField(db_column='CAR_IMG_ID', primary_key=True)  # Field name made lowercase.
    car = models.ForeignKey(TbCar, models.DO_NOTHING, db_column='CAR_ID')  # Field name made lowercase.
    created_date = models.DateTimeField(db_column='CREATED_DATE', blank=True, null=True)  # Field name made lowercase.
    updated_date = models.DateTimeField(db_column='UPDATED_DATE', blank=True, null=True)  # Field name made lowercase.
    content_type = models.CharField(db_column='CONTENT_TYPE', max_length=45, blank=True, null=True)  # Field name made lowercase.
    file_name = models.CharField(db_column='FILE_NAME', max_length=45, blank=True, null=True)  # Field name made lowercase.
    size = models.IntegerField(db_column='SIZE', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'TB_CAR_IMAGE'
        unique_together = (('car_img_id', 'car'),)