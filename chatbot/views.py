"""
Views for the chatbot application.
"""
import urllib.parse
from logging import getLogger

from django.conf import settings

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import ChatbotQuerySerializer, ChatbotResponseSerializer
from .rag.rag_engine import RAGEngine
from .rag.config import RETRIEVAL_K, RETRIEVAL_THRESHOLD

logger = getLogger(__name__)

logger.setLevel(settings.LOG_LEVEL)

class IndexView(APIView):
    """
    API view for interacting with the chatbot.
    GET method accepts a query parameter and returns the chatbot's response.
    """

    # Instance of the RAG engine
    _rag_engine = None

    @property
    def rag_engine(self):
        """
        Lazy initialization of the RAG engine.
        """
        if self._rag_engine is None:
            self._rag_engine = RAGEngine()
        return self._rag_engine

    def post(self, request, *args, **kwargs):
        """
        Handle POST requests to query the chatbot.

        Request body:
        - query: The URL-encoded question to the chatbot

        Returns:
        - A response with the chatbot's answer
        """
        # Check if request body is empty
        if not request.data:
            return Response({'message': 'Please provide a query in the request body.'}, status=status.HTTP_400_BAD_REQUEST)

        # Check if query is in the request data
        if 'query' not in request.data:
            return Response({'message': 'Please include a "query" field in the request body.'}, status=status.HTTP_400_BAD_REQUEST)

        # Get and decode the URL-encoded query
        encoded_query = request.data.get('query', '')

        # URL decoding
        decoded_query = urllib.parse.unquote(encoded_query)

        logger.info('Received query: %s', decoded_query)

        # Validate the query parameter
        serializer = ChatbotQuerySerializer(data={'query': decoded_query})
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        query = serializer.validated_data['query']

        try:
            # Get response from RAG engine
            response_text = self.rag_engine.ask(
                query,
                k=RETRIEVAL_K,
                score_threshold=RETRIEVAL_THRESHOLD
            )

            # Create response serializer
            response_data = {
                'text': response_text,
                'status': 'success'
            }

            response_serializer = ChatbotResponseSerializer(data=response_data)
            response_serializer.is_valid()
            return Response(response_serializer.data)

        except ValueError as exc:
            # Handle the case where the RAG engine couldn't find relevant documents
            return Response(
                {'text': str(exc), 'status': 'error'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception:
            # Handle other exceptions
            return Response(
                {'text': 'Error processing your request', 'status': 'error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
