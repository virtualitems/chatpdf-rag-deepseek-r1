# Local ChatPDF with RAG and Mistral-7B

**ChatPDF** is a Retrieval-Augmented Generation (RAG) application that allows users to interact with PDF documents through both a web interface and CLI. The system uses Mistral-7B and advanced embedding models for efficient and accurate question-answering in Spanish.

## Features

- **PDF Upload**: Upload one or multiple PDF documents to enable question-answering across their combined content.
- **RAG Workflow**: Combines retrieval and generation for high-quality responses.
- **Customizable Retrieval**: Adjust the number of retrieved results (`k`) and similarity threshold to fine-tune performance.
- **Memory Management**: Easily clear vector store and retrievers to reset the system.
- **Streamlit Interface**: A user-friendly web application for seamless interaction.

---

## Installation

Follow the steps below to set up and run the application:

### 1. Clone the Repository

```bash
git clone https://github.com/paquino11/chatpdf-rag-deepseek-r1.git
cd chatpdf-rag-deepseek-r1
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure to include the following packages in your `requirements.txt`:

```
streamlit
langchain
langchain_ollama
langchain_community
streamlit-chat
pypdf
chromadb
```

### 4. Pull Required Models for Ollama

Pull the required models via the `ollama` CLI:

```bash
ollama pull mistral:7b
ollama pull mxbai-embed-large
```

## Usage

### Web Interface

Run the Streamlit app:

```bash
streamlit run app/gui.py
```

### Command Line Interface

Use the CLI for quick queries:

```bash
# Ask a question about your documents
python cli.py "¿Cuál es el horario de atención?"

# Pipe content to the CLI
echo "¿Cuáles son los servicios?" | python cli.py
```

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration parameters
│   ├── document_loader.py # Document loading utilities
│   ├── gui.py            # Streamlit web interface
│   ├── prepare.py        # Document ingestion script
│   ├── rag_engine.py     # Core RAG implementation
│   └── vector_store.py   # Vector store management
├── cli.py                # Command line interface
├── docs/                 # Directory for documents
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

---

## Configuration

You can modify the following parameters in `config.py` to suit your needs:

1. **Models**:
   - Default LLM: `mistral:7b`
   - Default Embedding: `mxbai-embed-large`
   - Change these in the configuration file by updating `LLM_MODEL` and `EMBEDDING_MODEL`
   - Any Ollama-compatible model can be used

2. **Chunking Parameters**:
   - `chunk_size=1024` and `chunk_overlap=100`
   - Adjust for larger or smaller document splits

3. **Retrieval Settings**:
   - Adjust `k` (number of retrieved results) and `score_threshold` in `ask()` to control the quality of retrieval.

---

## Requirements

- **Python**: 3.8+
- **Streamlit**: Web framework for the user interface.
- **Ollama**: For embedding and LLM models.
- **LangChain**: Core framework for RAG.
- **PyPDF**: For PDF document processing.
- **ChromaDB**: Vector store for document embeddings.

---

## Troubleshooting

### Common Issues

1. **Missing Models**:
   - Ensure you've pulled the required models using `ollama pull`.

2. **Vector Store Errors**:
   - Delete the `chroma_db/` directory if you encounter dimensionality errors:
     ```bash
     rm -rf chroma_db/
     ```

3. **Streamlit Not Launching**:
   - Verify dependencies are installed correctly using `pip install -r requirements.txt`.

---

## Future Enhancements

- **Memory Integration**: Add persistent memory to maintain conversational context across sessions.
- **Advanced Analytics**: Include visual insights from retrieved content.
- **Expanded Model Support**: Support additional embedding and LLM providers.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Ollama](https://ollama.ai/)

