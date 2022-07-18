from Mesa import views
from django.urls import include, path, re_path
from rest_framework import routers

router = routers.DefaultRouter(trailing_slash=False)
router.register(r'chapters', views.ChapterViewSet)

urlpatterns=[
     path('', include(router.urls)),
     re_path(r'^user$',views.userApi),
     re_path(r'^user/validate$',views.validateUserApi),
     re_path(r'^keyword/(?P<chapter_id>\d+)/$', views.keywordApi, name='keyword'),
     re_path(r'^grammar/(?P<chapter_id>\d+)/$', views.grammarApi, name='grammar'),
     re_path(r'^summarize/(?P<chapter_id>\d+)/$', views.summarizeApi, name='summarize'),
     re_path(r'^mcq/(?P<chapter_id>\d+)/$', views.mcqApi, name='mcq')
]