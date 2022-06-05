from django.urls import re_path as url
from Mesa import views
from django.urls import include, path
from rest_framework import routers

router = routers.DefaultRouter(trailing_slash=False)
router.register(r'chapters', views.ChapterViewSet)

urlpatterns=[
     path('', include(router.urls)),
     url(r'^user$',views.userApi),
     url(r'^user/validate$',views.validateUserApi),
     url(r'^keyword/(?P<chapter_id>\d+)/$', views.keywordApi, name='keyword')
]