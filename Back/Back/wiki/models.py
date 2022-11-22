from django.db import models
from django_db_views.db_view import DBView

class TbUserWiki(models.Model):
    user_id = models.IntegerField(db_column='USER_ID', primary_key=True)  # Field name made lowercase.
    inwiki = models.CharField(db_column='inWIKI', max_length=45)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'TB_USER_WIKI'
        unique_together = (('user_id', 'inwiki'),)

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


class AccountsUser(models.Model):
    id = models.BigAutoField(primary_key=True)
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    email = models.CharField(unique=True, max_length=30)
    name = models.CharField(max_length=20)
    nickname = models.CharField(max_length=50)
    profile_img_url = models.CharField(max_length=100)
    point = models.IntegerField()
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    is_superuser = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'accounts_user'


class VwWikiLarge(DBView):
    manufacturer = models.CharField(db_column='manufacturer', max_length=45, primary_key=True)  # Field name made lowercase.
    total = models.CharField(db_column='total', max_length=45, blank=True, null=True)  # Field name made lowercase.
    inwiki = models.CharField(db_column='inwiki', max_length=45, blank=True, null=True)  # Field name made lowercase.
    # efficiency = models.CharField(db_column='EFFICIENCY', max_length=45, blank=True, null=True)  # Field name made lowercase.
    
    class Meta:
        managed = False
        db_table = 'VW_WIKILARGE'
        
# class VwWikiSmall(DBView):
    
