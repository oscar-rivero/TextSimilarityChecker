from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('check/', views.check, name='check'),
    path('report/', views.report, name='report'),
]