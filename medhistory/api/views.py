from rest_framework import viewsets,mixins
from medhistory.models import History
from .serializers import HistorySerializer,HistoryUploadSerializer,DiseaseSerializer,HistoryAnotherSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authentication import BasicAuthentication
from django.views.decorators.csrf import csrf_exempt
from accounts.models import MyUser
from django.http import Http404
from rest_framework.decorators import api_view
from .eye import *
import io
import base64
from .skin import *
from .diabetes import *
class HistoryViewSet(mixins.CreateModelMixin,
                                viewsets.GenericViewSet):

    
    queryset = History.objects.all()
    serializer_class = HistoryUploadSerializer
    #authentication_classes = (CsrfExemptSessionAuthentication, BasicAuthentication)
    #permission_classes = (permissions.IsAuthenticatedOrReadOnly,
    #                      IsOwnerOrReadOnly,)
    @csrf_exempt
    def perform_create(self,serializer):
        #print("$$$$$$$$$$$",se.request.user)
        serializer.save(user=self.request.user)

class HistoryList(APIView):
    def get(self, request, format=None):
        historyobjs = History.objects.filter(user=request.user)
        serializer = HistorySerializer(historyobjs, many=True)
        return Response(serializer.data)

class UserDiseaseList(APIView):
    def get(self, request,adhaar_card=None,format=None):
        historyobjs = History.objects.filter(user=request.user)
        serializer = DiseaseSerializer(historyobjs, many=True)
        return Response(serializer.data)


class BPHistory(APIView):
    def get_object(self,adhaar_card=None):
        try:
            return MyUser.objects.get(adhaar_card=adhaar_card)
        except MyUser.DoesNotExist:
            raise Http404

    def get(self, request, adhaar_card=None, format=None):
        my_user = self.get_object(adhaar_card)
        k=History.objects.filter(user=my_user)
        serializer = HistoryAnotherSerializer(k,many=True)
        return Response(serializer.data)


@api_view(['GET','POST'])
def examine(request,image_str):
    #image_string = io.BytesIO(base64.b64decode(image_str))
    #print(image_str)
    #image=base64.decodestring(image_str)
    #image_64_encode = base64.encodestring(image)
    points=check_cataract(image_str)
    print ("mukul nimit")
    return Response({'points':points})

@api_view(['GET','POST'])
def examine_skin(request,image_str):
    #image_string = io.BytesIO(base64.b64decode(image_str))
    #print(image_str)
    #image=base64.decodestring(image_str)
    #image_64_encode = base64.encodestring(image)
    points=check_cancer(image_str)
    return Response({'points':points})

@api_view(['GET','POST'])
def examine_diabities(request,a,b,c,d,e,f,g,h):
    #image_string = io.BytesIO(base64.b64decode(image_str))
    #print(image_str)
    #image=base64.decodestring(image_str)
    #image_64_encode = base64.encodestring(image)
    points=checkDiabetes(a,b,c,d,e,f,g,h)
    return Response({'points':points})



