"""
URLs for the chatbot application.
"""
from django.urls import path

from . import views

urlpatterns = [
    path('', views.IndexView.as_view(), name='chatbot_index'),
]
