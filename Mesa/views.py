from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import null
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
from Mesa.extract_keywords import extractKeywordsFromContent
from Mesa.translation import translatemethod
from Mesa.summarization import summarizemethod
from Mesa.models import Chapter, User
from Mesa.serializers import UserSerializer, ChapterSerializer
#from Mesa.mcq import extractMCQ
from rest_framework import viewsets

# Create your views here.
summ_result=''
mcqs={}
class ChapterViewSet(viewsets.ModelViewSet):
    queryset = Chapter.objects.all()
    serializer_class = ChapterSerializer

@csrf_exempt
def userApi(request, id=0):
    if request.method == 'GET':
        users = User.objects.all()
        users_serializer = UserSerializer(users, many = True)
        return JsonResponse(users_serializer.data, safe=False)
    elif request.method == 'POST':
        user_info = JSONParser().parse(request)
        users_serializer = UserSerializer(data = user_info)
        if users_serializer.is_valid():
            users_serializer.save()
            return JsonResponse("User information added successfully.", safe=False)
        return JsonResponse("Failed to add information.", safe=False)
    elif request.method == 'PUT':
        user_info = JSONParser().parse(request)
        user = User.objects.get(UserId = user_info['userId'])
        users_serializer = UserSerializer(user, data = user_info)
        if users_serializer.is_valid():
            users_serializer.save()
            return JsonResponse("Update successfully.", safe=False)
        return JsonResponse("Failed to update.", safe=False)

@csrf_exempt
def validateUserApi(request, id=0):
    if request.method == 'POST':
        user_info = JSONParser().parse(request)
        
        isValid = True
        msg = "Login successfully"
        info = None

        try:
            user = User.objects.get(emailId = user_info['emailId'], password = user_info['password'])
            info = UserSerializer(user, many = False).data
        except:
            isValid = False
            msg = "Either username or password is invalid"            
        
        resp = {"msg": msg, "isValid": isValid, "info": info}                    
        return JsonResponse(resp, safe=False)        

@csrf_exempt
def keywordApi(request, chapter_id):
    if request.method=='GET':
        chapter=Chapter.objects.get(id = chapter_id)                    
        result = extractKeywordsFromContent(chapter.content)   
        finalRes = {"keywords": result}     
        return JsonResponse(finalRes, safe=False)

@csrf_exempt
def summarizeApi(request, chapter_id):
    if request.method=='GET':
        chapter=Chapter.objects.get(id = chapter_id)                    
        summ_result = summarizemethod(chapter.content)   
        data_to_send = {"Summarized Text": summ_result}     
        return JsonResponse(data_to_send, safe=False)

# @csrf_exempt
# def translateApi(request):
#     if request.method=='GET':
#         trans_result = translatemethod(summ_result)   
#         data_to_send = {"Translated Text": trans_result}     
#         return JsonResponse(data_to_send, safe=False)

# @csrf_exempt
# def mcqApi(request, chapter_id):
#     if request.method=='GET':
#         chapter=Chapter.objects.get(id = chapter_id)                    
#         mcqs = extractMCQ(chapter.content)        
#         return JsonResponse(mcqs, safe=False)

@csrf_exempt
def translateApi(request, chapter_id):
    if request.method=='GET':
        chapter=Chapter.objects.get(id = chapter_id) 
        summ_result = summarizemethod(chapter.content)                    
        trans_result = translatemethod(summ_result)   
        data_to_send = {"Translated Text": trans_result}     
        return JsonResponse(data_to_send, safe=False)