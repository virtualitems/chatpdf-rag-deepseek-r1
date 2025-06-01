"""Script de inicialización para cargar documentos en el sistema RAG."""

import os
import time
import logging

from .rag_engine import RAGEngine
from .config import DOCS_DIRECTORY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents():
    """
    Carga documentos desde el directorio docs en el motor RAG.

    Esta función automáticamente ingesta archivos PDF y TXT desde la carpeta docs.
    """
    engine = RAGEngine()
    docs_dir = DOCS_DIRECTORY

    if not docs_dir.exists():
        logger.error('Directorio de documentos no encontrado: %s', docs_dir)
        return

    logger.info('Cargando documentos desde: %s', docs_dir)

    # Listar todos los archivos en el directorio docs
    files = os.listdir(docs_dir)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    txt_files = [f for f in files if f.lower().endswith('.txt')]

    # Procesar archivos PDF
    for pdf_file in pdf_files:
        file_path = os.path.join(docs_dir, pdf_file)
        logger.info('Ingiriendo PDF: %s', pdf_file)
        t0 = time.time()
        success = engine.ingest(file_path)
        t1 = time.time()

        if success:
            logger.info('PDF %s ingresado en %.2f segundos', pdf_file, t1 - t0)
        else:
            logger.error('Error al ingerir PDF %s', pdf_file)

    # Procesar archivos TXT
    for txt_file in txt_files:
        file_path = os.path.join(docs_dir, txt_file)
        logger.info('Ingiriendo TXT: %s', txt_file)
        t0 = time.time()
        success = engine.ingest(file_path)
        t1 = time.time()

        if success:
            logger.info('TXT %s ingresado en %.2f segundos', txt_file, t1 - t0)
        else:
            logger.error('Error al ingerir TXT %s', txt_file)


if __name__ == '__main__':
    load_documents()
