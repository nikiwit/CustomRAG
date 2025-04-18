# Advanced CustomRAG - Local Question Answering System

This project implements an advanced Retrieval-Augmented Generation (RAG) system that runs entirely locally on your machine. It uses a sophisticated multi-stage pipeline to intelligently process your questions and provide highly relevant answers from your document collection.

## Features

* **Local First:** Runs completely offline (after downloading models and dependencies). No API keys or data sent to external services (unless you change the LLM/Embedding configuration).
* **Advanced Query Processing:** Uses NLP techniques including:
    * Query normalization with lemmatization and stemming
    * Query expansion for improved document recall
    * Intelligent query type classification (factual, procedural, conceptual, etc.)
* **Multi-Strategy Retrieval:** Implements multiple document retrieval approaches:
    * Semantic search using embeddings
    * Keyword-based search for exact term matching
    * Hybrid search combining semantic and keyword approaches
    * Maximum Marginal Relevance (MMR) for diverse results
* **Smart Context Generation:** Prioritizes and organizes retrieved information by:
    * Scoring document relevance to your specific question
    * Dynamically compressing context when needed
    * Preserving the most important information for the LLM
* **Custom Knowledge Base:** Simply place your documents (e.g., `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.epub`) into the `data` directory. Support for specific formats depends on installing optional dependencies.
* **Conversation Context:** Maintains conversation memory for follow-up questions.
* **Open Source Stack:** Built using popular Python libraries:
    * **Orchestration:** LangChain
    * **NLP Processing:** NLTK
    * **LLM:** Ollama (running models like DeepSeek-R1 locally)
    * **Embeddings:** Sentence Transformers (Hugging Face `all-MiniLM-L6-v2`)
    * **Vector Store:** ChromaDB (local persistent storage)
* **GPU Acceleration:** Automatically uses CUDA or MPS (Apple Silicon) when available.

## Prerequisites

* **Python:** Version 3.8 or higher. Download from [https://www.python.org/](https://www.python.org/)
* **Ollama:** Must be installed and running. Download from [https://ollama.com/](https://ollama.com/)
* **Git:** For cloning the repository. Download from [https://git-scm.com/](https://git-scm.com/)

## Installation

1.  **Clone the Repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone https://github.com/nikiwit/CustomRAG.git
    cd CustomRAG
    ```

2.  **Create and Activate Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    # Navigate into the project directory (cd CustomRAG) first if you haven't already
    python -m venv venv # Or use python3 if needed on your system. Feel free to use other names like .venv or RAGenv instead of venv.
    ```
    Activate the environment:
    * On **macOS/Linux** (bash/zsh):
        ```bash
        source venv/bin/activate
        ```
    * On **Windows** (Command Prompt):
        ```cmd
        .\venv\Scripts\activate
        ```
    * On **Windows** (PowerShell):
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *(You should see `(venv)` or your chosen environment name at the beginning of your terminal prompt)*

3.  **Install Dependencies:**
    Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(This reads the `requirements.txt` file and installs libraries like LangChain, ChromaDB, Sentence-Transformers, NLTK, and Unstructured with its extras for file parsing)*

    **Platform-Specific Installation Notes:**

    * **Apple Silicon Macs (M1/M2/M3):** If you encounter build errors related to `onnx` or SSE4.1 instructions:
        ```bash
        # Install PyTorch separately first
        pip install torch

        # Then install the core requirements without building problematic packages
        pip install --no-deps chromadb
        pip install pydantic hnswlib typing_extensions overrides
        pip install langchain langchain-community langchain-chroma langchain-huggingface sentence-transformers requests
        pip install nltk numpy
        
        # Finally install document processing libraries
        pip install pypdf pdfplumber docx2txt python-docx python-pptx
        pip install ebooklib beautifulsoup4 lxml tqdm python-dotenv chardet html2text
        pip install unstructured
        pip install "unstructured[pptx]" "unstructured[epub]"
        ```

    * **Windows:** You might need to install additional build tools:
        1. Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
        2. When installing, select "Desktop development with C++"

4.  **Download the LLM via Ollama:**
    This project is configured by default to use the `deepseek-r1:8b` model. You need to download it using the Ollama command line tool.
    ```bash
    ollama pull deepseek-r1:8b
    ```
    *(Ensure the Ollama application/server is running in the background before executing this command and before running the main application later. See Troubleshooting section)*

    *(Optional: If you wish to use a different model supported by Ollama, pull the desired model (e.g., `ollama pull llama3`) and update the `LLM_MODEL_NAME` variable within the `customrag.py` script)*

## How to Use

1.  **Add Your Documents:** Place your knowledge base files into the `data/` directory.
    * Supported formats depend on the installed dependencies specified in `requirements.txt`. The default setup includes support for `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, and `.epub` via the `unstructured` library extras. See `requirements.txt` for details on enabling support for other types if needed.

2.  **Ensure Ollama is Running:** Before running the script, verify that the Ollama application/server is active in the background. See the "Ollama Not Running / Connection Errors" section under "Notes & Troubleshooting" for details on how to check this.

3.  **Run the Application:**
    Make sure your virtual environment is activated. Then, run the main script from your terminal:
    ```bash
    python app.py
    ```
    *(Or use `python3 customrag.py` if that's how you invoke Python 3 on your system)*

4.  **First Run:** The very first time you run the script (or after adding/changing documents and deleting the `vector_store/` directory), it will perform these steps:
    * Load all supported documents found in the `data/` directory.
    * Split the documents' content into smaller text chunks.
    * Generate numerical embeddings for each chunk (this can take significant time depending on the number/size of documents and your computer's processing power).
    * Create and save a local vector store containing these embeddings in the `vector_store/` directory. Please be patient during this process.

5.  **Subsequent Runs:** On future runs, if the `vector_store/` directory exists, the script will load the pre-computed embeddings directly from it, making the startup process much faster.

6.  **Ask Questions:** Once the application is ready, you can type your questions related to the content of your documents. The system will:
   * Analyze and classify your query
   * Select the optimal retrieval strategy
   * Retrieve and prioritize the most relevant information
   * Generate a response based on the retrieved context

7.  **Commands:**
   * Type `exit` or `quit` to stop the application
   * Type `clear` to reset the conversation memory
   * Type `reindex` to reindex all documents
   * Type `stats` to see document statistics
   * Type `help` to print the list of available commands

## Folder Structure

```
CustomRAG/
    ├── data/                # <-- Place your knowledge base items there (e.g., .pdf, .docx, .txt)
    ├── vector_store/        # Stores the generated vector embeddings (created automatically, ignored by Git)
    ├── venv/                # Python virtual environment folder (if named 'venv', ignored by Git)
    ├── customrag.py         # Main application Python script
    ├── .gitignore           # Specifies intentionally untracked files/folders for Git
    ├── requirements.txt     # List of Python dependencies to install
    └── README.md            # This file
```

## Advanced Features

### Query Processing

The system analyzes your questions to determine the most effective way to retrieve information:

* **Query Type Classification:** Automatically detects if your question is factual, procedural, conceptual, comparative, or exploratory.
* **Query Expansion:** Generates variations of your query using synonyms to improve retrieval recall.
* **Normalization:** Applies stemming and lemmatization to match more relevant documents.

### Retrieval Strategies

Different retrieval strategies are automatically selected based on your query type:

* **Semantic Search:** Used for conceptual and procedural questions to find content with similar meaning.
* **Keyword Search:** Used for factual questions to find exact matches.
* **Hybrid Search:** Combines semantic and keyword approaches with configurable weights.
* **MMR (Maximum Marginal Relevance):** Used for exploratory and comparative questions to ensure diversity in results.

### Context Processing

Retrieved documents are intelligently processed before being sent to the LLM:

* **Relevance Scoring:** Documents are scored based on how well they match your query.
* **Priority-Based Selection:** High-relevance documents are prioritized for context inclusion.
* **Context Compression:** When necessary, the system intelligently compresses content to fit more information into the context window.

## Configuration Options

The system can be configured using environment variables:

* `CUSTOMRAG_DATA_PATH`: Path to your documents (default: "data" directory)
* `CUSTOMRAG_VECTOR_PATH`: Path for storing vector embeddings (default: "vector_store" directory)
* `CUSTOMRAG_EMBEDDING_MODEL`: Embedding model to use (default: "all-MiniLM-L6-v2")
* `CUSTOMRAG_LLM_MODEL`: Ollama model to use (default: "deepseek-r1:8b")
* `CUSTOMRAG_CHUNK_SIZE`: Document chunk size (default: 500)
* `CUSTOMRAG_CHUNK_OVERLAP`: Overlap between chunks (default: 150)
* `CUSTOMRAG_RETRIEVER_K`: Number of chunks to retrieve (default: 10)
* `CUSTOMRAG_SEARCH_TYPE`: Default search type - "semantic", "keyword", "hybrid", or "mmr" (default: "hybrid")
* `CUSTOMRAG_KEYWORD_RATIO`: Weight given to keyword results in hybrid search (default: 0.3)
* `CUSTOMRAG_QUERY_EXPANSION`: Enable query expansion - "True" or "False" (default: "True")
* `CUSTOMRAG_FORCE_REINDEX`: Force reindex on startup - "True" or "False" (default: "False")

Example of setting environment variables (Linux/Mac):
```bash
export CUSTOMRAG_SEARCH_TYPE="hybrid"
export CUSTOMRAG_KEYWORD_RATIO="0.4"
```

## Document Format Support

The system supports various document formats:

* **Text Files** (`.txt`, `.md`): Basic text files.
* **PDF Files** (`.pdf`): Using PyPDFLoader or PDFPlumberLoader.
* **Word Documents** (`.docx`, `.doc`): Using Docx2txtLoader.
* **PowerPoint Files** (`.ppt`, `.pptx`): Using UnstructuredPowerPointLoader.
* **EPUB Books** (`.epub`): Using either UnstructuredEPubLoader or a custom loader with ebooklib and html2text.

For EPUB files, the system first tries to use the UnstructuredEPubLoader, and if that fails, it falls back to a custom implementation using ebooklib, BeautifulSoup, and html2text.

## Notes & Troubleshooting

* **Performance:** Running Large Language Models (LLMs) locally requires significant RAM (memory) and computational power. Performance will vary based on your computer's hardware specifications (CPU, GPU if used by Ollama, available RAM). The default `deepseek-r1:8b` model (around 5GB download, 8 Billion parameters) is reasonably balanced but typically requires several gigabytes of RAM while running. Expect responses to be slower than cloud-based API services. Larger models will demand more resources.

* **First Run Time:** As mentioned, generating embeddings for many or large documents can take time (potentially several minutes or longer). Subsequent runs are much faster as they load the saved `vector_store/`.

* **Updating Knowledge Base:** If you add, remove, or modify files in the `data/` directory, the existing information in `vector_store/` becomes outdated. To include the latest changes in the RAG system's knowledge, you can either:
  * Type `reindex` in the application interface to trigger reindexing, or
  * Delete the entire `vector_store/` directory, and the script will perform the full re-indexing process on its next run.

* **NLP Resources:** On first run, the system may download NLTK resources (punkt, stopwords, wordnet). This happens automatically and only once.

* **Ollama Not Running / Connection Errors:** The script needs to connect to the Ollama application running as a background server. If the script fails immediately mentioning "connection refused," "could not connect," or similar network errors, the Ollama server is likely not running or accessible.
    * **How to Check if Ollama is Running:** The most reliable cross-platform method is to use the Ollama command line in your terminal:
        ```bash
        ollama list
        ```
        If this command successfully lists your downloaded models (or shows an empty list) without connection errors, the server is running. An error like `Error: could not connect to Ollama server` indicates it's not running.
    * **How to Start Ollama:** If Ollama isn't running, start the Ollama application using your operating system's standard method (e.g., launching it from your installed applications, using a desktop shortcut, or potentially via the command line on some systems). This should start the background server process.

* **Installation Issues:**
  * **Package build errors:** Some packages may fail to build, especially on certain platforms like Apple Silicon. Follow the platform-specific installation instructions in the Installation section.
  * **Dependency conflicts:** If you encounter dependency conflicts, consider installing dependencies in stages as described in the platform-specific notes.
  * **PyTorch issues:** If PyTorch is causing problems, install it separately first with `pip install torch` before installing other requirements.
  * **NLTK issues:** If you see NLTK-related errors, try manually downloading resources with `python -m nltk.downloader punkt stopwords wordnet`.

* **Memory Issues (RAM):** If the script crashes, especially during embedding or when answering questions, your system might be running out of RAM. Close other resource-heavy applications. If issues persist, consider using a smaller LLM via Ollama (e.g., `phi3:medium`, `llama3:8b`). Remember to `ollama pull` the new model name and update the `LLM_MODEL_NAME` variable or environment setting.

* **GPU Acceleration:** The system will automatically detect and use:
  * CUDA for NVIDIA GPUs
  * MPS for Apple Silicon Macs
  
  No additional configuration is needed as this is handled by the `VectorStoreManager.get_embedding_device()` method.