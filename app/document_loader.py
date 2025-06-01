"""Módulo para cargar diferentes tipos de documentos."""

from pathlib import Path
from typing import List, Union
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

from .config import TEXT_SPLITTER_CONFIG

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Clase para cargar y procesar documentos de diferentes formatos.

    Esta clase proporciona métodos para cargar documentos PDF y TXT y
    preprocesarlos para su uso en un sistema RAG.

    Attributes:
        text_splitter: Divisor de texto para dividir documentos en chunks.
    """

    def __init__(self):
        """Inicializa el cargador de documentos con un divisor de texto configurado."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG['chunk_size'],
            chunk_overlap=TEXT_SPLITTER_CONFIG['chunk_overlap']
        )

    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Carga un documento desde una ruta de archivo.

        Args:
            file_path: Ruta al archivo a cargar.

        Returns:
            Lista de documentos procesados y divididos en chunks.

        Raises:
            ValueError: Si el formato de archivo no es compatible.
        """
        file_path_str = str(file_path)
        logger.info(f'Cargando documento: {file_path_str}')

        # Seleccionar el cargador apropiado según la extensión del archivo
        if file_path_str.lower().endswith('.pdf'):
            return self._load_pdf(file_path_str)
        elif file_path_str.lower().endswith('.txt'):
            return self._load_txt(file_path_str)
        else:
            raise ValueError(f'Formato de archivo no compatible: {file_path_str}. Solo se admiten PDF y TXT.')

    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Carga un documento PDF.

        Args:
            file_path: Ruta al archivo PDF.

        Returns:
            Lista de documentos procesados y divididos en chunks.
        """
        logger.info(f'Cargando PDF: {file_path}')
        docs = PyPDFLoader(file_path=file_path).load()
        return self._process_documents(docs)

    def _load_txt(self, file_path: str) -> List[Document]:
        """
        Carga un documento de texto.

        Args:
            file_path: Ruta al archivo TXT.

        Returns:
            Lista de documentos procesados y divididos en chunks.
        """
        logger.info(f'Cargando TXT: {file_path}')
        docs = TextLoader(file_path=file_path).load()
        return self._process_documents(docs)

    def _process_documents(self, docs: List[Document]) -> List[Document]:
        """
        Procesa documentos: los divide en chunks y filtra metadatos complejos.

        Args:
            docs: Lista de documentos a procesar.

        Returns:
            Lista de documentos procesados y divididos en chunks.
        """
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        logger.info(f'Documento procesado en {len(chunks)} chunks')
        return chunks
