"""Interfaz de usuario para el sistema RAG basado en Streamlit."""

import streamlit as st
from streamlit_chat import message

from .rag_engine import RAGEngine
from .config import RETRIEVAL_K, RETRIEVAL_THRESHOLD

# Configuración de la página
st.set_page_config(page_title='RAG con DeepSeek Coder')


class ChatInterface:
    """
    Interfaz de chat para el sistema RAG.

    Esta clase implementa la interfaz de usuario basada en Streamlit
    para interactuar con el sistema RAG.
    """

    @staticmethod
    def display_messages():
        """
        Muestra el historial de mensajes del chat.
        """
        st.subheader('Historial de Conversación')
        for i, (msg, is_user) in enumerate(st.session_state['messages']):
            message(msg, is_user=is_user, key=str(i))
        st.session_state['thinking_spinner'] = st.empty()

    @staticmethod
    def process_input():
        """
        Procesa la entrada del usuario y genera una respuesta.
        """
        user_input = st.session_state['user_input'].strip()
        if not user_input:
            return

        with st.session_state['thinking_spinner'], st.spinner('Pensando...'):
            try:
                assistant_response = st.session_state['assistant'].ask(
                    user_input,
                    k=RETRIEVAL_K,
                    score_threshold=RETRIEVAL_THRESHOLD
                )
            except ValueError as e:
                assistant_response = str(e)

        # Añadir mensajes al historial
        st.session_state['messages'].append((user_input, True))
        st.session_state['messages'].append((assistant_response, False))

        # Limpiar el campo de entrada
        st.session_state['user_input'] = ''

    @staticmethod
    def render():
        """
        Renderiza la interfaz de usuario.
        """
        # Inicializar el estado de la sesión si es necesario
        if len(st.session_state) == 0:
            st.session_state['messages'] = []
            st.session_state['assistant'] = RAGEngine()
            st.session_state['user_input'] = ''

        # Título de la aplicación
        st.header('RAG con DeepSeek Coder')
        st.session_state['ingestion_spinner'] = st.empty()

        # Mostrar mensajes y entrada de texto
        ChatInterface.display_messages()
        st.text_input(
            'Mensaje',
            key='user_input',
            on_change=ChatInterface.process_input
        )

        # Botón para limpiar el chat
        if st.button('Limpiar Conversación'):
            st.session_state['messages'] = []
            st.session_state['assistant'].clear()


if __name__ == '__main__':
    ChatInterface.render()
