"""Motor principal de RAG (Retrieval Augmented Generation)."""

import logging

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from .document_loader import DocumentLoader
from .vector_store import VectorStoreManager
from .config import LLM_MODEL, RAG_PROMPT_TEMPLATE, RETRIEVAL_K, RETRIEVAL_THRESHOLD

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Motor principal de Retrieval Augmented Generation (RAG).

    Esta clase combina la carga de documentos, el almacenamiento vectorial y
    la generación de respuestas para implementar un sistema RAG completo.

    Attributes:
        document_loader: Cargador de documentos.
        vector_store_manager: Gestor del Vector Store.
        llm: Modelo de lenguaje grande.
        prompt: Plantilla para estructurar consultas al LLM.
    """

    def __init__(self, llm_model: str = LLM_MODEL):
        """
        Inicializa el motor RAG con sus componentes.

        Args:
            llm_model: Nombre del modelo de LLM a utilizar.
        """
        self.document_loader = DocumentLoader()
        self.vector_store_manager = VectorStoreManager()
        self.llm = ChatOllama(model=llm_model)
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        logger.info('Motor RAG inicializado con modelo %s', llm_model)

    def ingest(self, file_path: str) -> bool:
        """
        Ingesta un documento al sistema RAG.

        Args:
            file_path: Ruta del documento a ingestar.

        Returns:
            True si la ingesta fue exitosa, False en caso contrario.
        """
        try:
            documents = self.document_loader.load_document(file_path)
            logger.info('agregando %d documentos desde %s', len(documents), file_path)
            success = self.vector_store_manager.add_documents(documents)
            return success
        except Exception as e:
            logger.error('Error durante la ingesta del documento %s: %s', file_path, e)
            return False

    def ask(self, query: str, k: int = RETRIEVAL_K, score_threshold: float = RETRIEVAL_THRESHOLD) -> str:
        """
        Responde a una consulta utilizando el sistema RAG.

        Args:
            query: Consulta del usuario.
            k: Número de documentos a recuperar.
            score_threshold: Umbral de similitud para la recuperación.

        Returns:
            Respuesta generada por el modelo.

        Raises:
            ValueError: Si no hay documentos ingestados o no se pueden recuperar.
        """
        # Recuperar documentos relevantes
        retrieved_docs = self.vector_store_manager.retrieve_documents(
            query, k=k, score_threshold=score_threshold
        )

        if not retrieved_docs:
            return 'No se encontró contexto relevante en los documentos para responder tu pregunta.'

        # Formatear la entrada para el modelo
        formatted_input = {
            'context': '\n\n'.join(doc.page_content for doc in retrieved_docs),
            'question': query
        }

        # Construir y ejecutar la cadena RAG
        chain = (
            RunnablePassthrough()  # Pasa la entrada tal cual
            | self.prompt          # Formatea la entrada para el LLM
            | self.llm            # Consulta al LLM
            | StrOutputParser()    # Parsea la salida del LLM
        )

        logger.info('Generando respuesta usando el LLM')
        return chain.invoke(formatted_input)

    def clear(self):
        """
        Limpia el estado del motor RAG, eliminando todos los documentos ingestados.
        """
        self.vector_store_manager.clear()
        logger.info('Estado del motor RAG reiniciado')
