"""Script de inicialización para cargar documentos en el sistema RAG."""

import os
import time
import logging
from pathlib import Path

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
        logger.error(f'Directorio de documentos no encontrado: {docs_dir}')
        return

    logger.info(f'Cargando documentos desde: {docs_dir}')

    # Listar todos los archivos en el directorio docs
    files = os.listdir(docs_dir)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    txt_files = [f for f in files if f.lower().endswith('.txt')]

    # Procesar archivos PDF
    for pdf_file in pdf_files:
        file_path = os.path.join(docs_dir, pdf_file)
        logger.info(f'Ingiriendo PDF: {pdf_file}')
        t0 = time.time()
        success = engine.ingest(file_path)
        t1 = time.time()

        if success:
            logger.info(f'PDF {pdf_file} ingresado en {t1 - t0:.2f} segundos')
        else:
            logger.error(f'Error al ingerir PDF {pdf_file}')

    # Procesar archivos TXT
    for txt_file in txt_files:
        file_path = os.path.join(docs_dir, txt_file)
        logger.info(f'Ingiriendo TXT: {txt_file}')
        t0 = time.time()
        success = engine.ingest(file_path)
        t1 = time.time()

        if success:
            logger.info(f'TXT {txt_file} ingresado en {t1 - t0:.2f} segundos')
        else:
            logger.error(f'Error al ingerir TXT {txt_file}')


if __name__ == '__main__':
    load_documents()
