'''Configuración para la aplicación RAG.'''

import logging
from pathlib import Path
from typing import Dict, Any

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes para la recuperación
RETRIEVAL_K = 1
RETRIEVAL_THRESHOLD = 0.2

# Configuración de modelos
LLM_MODEL = 'mistral:7b'
EMBEDDING_MODEL = 'mxbai-embed-large'

# Configuración de rutas
PERSIST_DIRECTORY = 'chroma_db'
DOCS_DIRECTORY = Path(__file__).parent.parent / 'docs'

# Configuración del prompt
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant answering questions based on the provided documents.
Context:
{context}

Question:
{question}

Answer concisely and accurately in three sentences or less.
Answer only with the relevant information from the context.
Answer in Spanish.
"""

# Configuración del vector store
VECTOR_STORE_CONFIG: Dict[str, Any] = {
    'persist_directory': PERSIST_DIRECTORY,
    'collection_name': 'document_collection'
}

# Configuración del chunking
TEXT_SPLITTER_CONFIG: Dict[str, Any] = {
    'chunk_size': 1024,
    'chunk_overlap': 100
}
