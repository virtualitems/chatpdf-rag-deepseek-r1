from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import os
import logging

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self, llm_model: str = 'mistral:7b', embedding_model: str = 'mxbai-embed-large'):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            Context:
            {context}

            Question:
            {question}

            Answer concisely and accurately in three sentences or less.
            """
        )
        self.persist_directory = 'chroma_db'

        # Try to load existing vector store
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info('Loaded existing vector store from disk')
                self.retriever = self.vector_store.as_retriever(
                    search_type='similarity_score_threshold',
                    search_kwargs={'k': 5, 'score_threshold': 0.2},
                )
            except Exception as e:
                logger.error(f'Error loading vector store: {e}')
                self.vector_store = None
                self.retriever = None
        else:
            self.vector_store = None
            self.retriever = None

    def ingest(self, file_path: str):
        """
        Ingest a file (PDF or TXT), split its contents, and store the embeddings in the vector store.
        """
        logger.info(f'Starting ingestion for file: {file_path}')

        # Choose the appropriate loader based on file extension
        if file_path.lower().endswith('.pdf'):
            docs = PyPDFLoader(file_path=file_path).load()
        elif file_path.lower().endswith('.txt'):
            docs = TextLoader(file_path=file_path).load()
        else:
            raise ValueError(f'Unsupported file format: {file_path}. Only PDF and TXT are supported.')

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        # If we already have a vector store, add these documents to it
        if self.vector_store:
            logger.info('Adding documents to existing vector store')
            self.vector_store.add_documents(documents=chunks)
        else:
            # Otherwise, create a new vector store
            logger.info('Creating new vector store')
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )

        # Update the retriever
        self.retriever = self.vector_store.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={'k': 5, 'score_threshold': 0.2},
        )

        logger.info('Ingestion completed. Document embeddings stored successfully.')

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError('No vector store found. Please ingest a document first.')

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type='similarity_score_threshold',
                search_kwargs={'k': k, 'score_threshold': score_threshold},
            )

        logger.info(f'Retrieving context for query: {query}')
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return 'No relevant context found in the document to answer your question.'

        formatted_input = {
            'context': '\n\n'.join(doc.page_content for doc in retrieved_docs),
            'question': query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info('Generating response using the LLM.')
        return chain.invoke(formatted_input)

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info('Clearing vector store and retriever.')
        self.vector_store = None
        self.retriever = None
