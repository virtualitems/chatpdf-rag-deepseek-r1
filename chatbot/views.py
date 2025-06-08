"""
Views for the chatbot application.
"""
from rest_framework.views import APIView
from rest_framework.response import Response

class IndexView(APIView):
    """
    A simple API view that returns a greeting message.
    """
    def get(self, request, *args, **kwargs):
        return Response({'message': 'This is the chatbot index page.'})
