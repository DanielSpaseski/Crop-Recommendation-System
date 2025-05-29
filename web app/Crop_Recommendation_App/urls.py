from django.urls import path
from .views import predict_crop

urlpatterns = [
    path('', predict_crop, name='predict_crop'),
]