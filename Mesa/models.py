from django.db import models

class User(models.Model):
    userId = models.AutoField(primary_key = True)
    userName = models.CharField(max_length = 500)
    contactNumber = models.CharField(max_length = 500)
    emailId = models.CharField(max_length = 500)
    password = models.CharField(max_length = 500)
    dateOfBirth = models.DateField()
    markForDelete = models.BooleanField(default = False)

class Chapter(models.Model):
    id=models.AutoField(primary_key=True)
    name=models.CharField(max_length=500)
    content=models.TextField()
