from rest_framework import serializers
from Mesa.models import  User, Chapter

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('userId', 'userName', 'contactNumber', 'emailId', 'password', 'dateOfBirth', 'markForDelete')

class ChapterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chapter
        fields = ('id','name','content')
