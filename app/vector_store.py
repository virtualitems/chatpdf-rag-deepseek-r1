"""Módulo para gestionar el almacenamiento vectorial de documentos."""

import logging
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from .config import EMBEDDING_MODEL, VECTOR_STORE_CONFIG, RETRIEVAL_K, RETRIEVAL_THRESHOLD

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Gestiona el almacenamiento y recuperación de documentos vectorizados.

    Esta clase proporciona métodos para crear, actualizar y consultar
    un Vector Store basado en Chroma DB.

    Attributes:
        embeddings: Modelo de embeddings utilizado.
        vector_store: Almacenamiento vectorial.
        retriever: Recuperador de documentos similares.
    """

    def __init__(self):
        """Inicializa el gestor de Vector Store con configuraciones predeterminadas."""
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store: Optional[Chroma] = None
        self.retriever: Optional[VectorStoreRetriever] = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """
        Intenta cargar un Vector Store existente o crea uno nuevo si es necesario.
        """
        try:
            # Intentar cargar un Vector Store existente
            self.vector_store = Chroma(
                persist_directory=VECTOR_STORE_CONFIG['persist_directory'],
                embedding_function=self.embeddings,
                collection_name=VECTOR_STORE_CONFIG.get('collection_name', 'default')
            )

            if self.vector_store.get()['ids']:
                # Si hay documentos en el Vector Store, configurar el retriever
                logger.info('Vector Store existente cargado correctamente')
                self._configure_retriever()
            else:
                logger.info('Vector Store existente está vacío')

        except Exception as e:
            logger.error('Error al cargar Vector Store: %s', e)
            self.vector_store = None
            self.retriever = None

    def _configure_retriever(self, k: int = RETRIEVAL_K, score_threshold: float = RETRIEVAL_THRESHOLD):
        """
        Configura el retriever con los parámetros especificados.

        Args:
            k: Número de documentos a recuperar.
            score_threshold: Umbral de similitud para la recuperación.
        """
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(
                search_type='similarity_score_threshold',
                search_kwargs={'k': k, 'score_threshold': score_threshold}
            )

    def add_documents(self, documents: List[Document]) -> bool:
        """
        Añade documentos al Vector Store.

        Args:
            documents: Lista de documentos a añadir.

        Returns:
            True si los documentos se añadieron correctamente, False en caso contrario.
        """
        if not documents:
            logger.warning('No hay documentos para añadir al Vector Store')
            return False

        try:
            if self.vector_store:
                # Añadir documentos al Vector Store existente
                self.vector_store.add_documents(documents=documents)
                logger.info('Se añadieron %d documentos al Vector Store existente', len(documents))
            else:
                # Crear un nuevo Vector Store con los documentos
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=VECTOR_STORE_CONFIG['persist_directory'],
                    collection_name=VECTOR_STORE_CONFIG.get('collection_name', 'default')
                )
                logger.info('Se creó un nuevo Vector Store con %d documentos', len(documents))

            self._configure_retriever()
            self.persist()
            return True

        except Exception as e:
            logger.error('Error al añadir documentos al Vector Store: %s', e)
            return False

    def retrieve_documents(self, query: str, k: int = RETRIEVAL_K,
                          score_threshold: float = RETRIEVAL_THRESHOLD) -> List[Document]:
        """
        Recupera documentos del Vector Store basados en una consulta.

        Args:
            query: Consulta para buscar documentos similares.
            k: Número de documentos a recuperar.
            score_threshold: Umbral de similitud para la recuperación.

        Returns:
            Lista de documentos recuperados.

        Raises:
            ValueError: Si el Vector Store no está inicializado.
        """
        if not self.vector_store:
            raise ValueError('Vector Store no inicializado. Por favor añada documentos primero.')

        # Actualizar configuración del retriever si los parámetros son diferentes
        if not self.retriever or self.retriever.search_kwargs.get('k') != k or \
           self.retriever.search_kwargs.get('score_threshold') != score_threshold:
            self._configure_retriever(k, score_threshold)

        if not self.retriever:
            raise ValueError('No se pudo configurar el retriever.')

        logger.info('Recuperando documentos para la consulta: %s', query)
        return self.retriever.invoke(query)

    def persist(self):
        """
        Persiste el Vector Store en disco.
        """
        if self.vector_store:
            self.vector_store.persist()
            logger.info('Vector Store guardado en disco')

    def clear(self):
        """
        Limpia el Vector Store y el retriever.
        """
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                logger.info('Vector Store eliminado')
            except Exception as e:
                logger.error('Error al eliminar la colección: %s', e)

        self.vector_store = None
        self.retriever = None
        logger.info('Vector Store y retriever reiniciados')
