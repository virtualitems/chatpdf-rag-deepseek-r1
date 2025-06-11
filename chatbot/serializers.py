"""
Serializers for the chatbot application.
"""
from rest_framework import serializers


class ChatbotQuerySerializer(serializers.Serializer):
    """
    Serializer for chatbot queries.
    """
    query = serializers.CharField(required=True, help_text="Consulta del usuario")


class ChatbotResponseSerializer(serializers.Serializer):
    """
    Serializer for chatbot responses.
    """
    text = serializers.CharField(help_text="Respuesta del chatbot")
    status = serializers.CharField(help_text="Estado de la respuesta")
