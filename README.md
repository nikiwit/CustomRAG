# Custom RAG - Local Question Answering

This project implements a simple Retrieval-Augmented Generation (RAG) system that runs entirely locally on your machine. It uses documents you provide as a knowledge base to answer your questions.

## Features

* **Local First:** Runs completely offline (after downloading models and dependencies). No API keys or data sent to external services (unless you change the LLM/Embedding configuration).
* **Custom Knowledge Base:** Simply place your documents (e.g., `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.epub`) into the `data` directory. Support for specific formats depends on installing optional dependencies.
* **Document Processing:** Automatically handles various file formats, splitting them into appropriate chunks for indexing and retrieval.
* **Conversation Context:** Maintains conversation memory for follow-up questions.
* **Open Source Stack:** Built using popular Python libraries:
    * **Orchestration:** LangChain
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
    *(This reads the `requirements.txt` file and installs libraries like LangChain, ChromaDB, Sentence-Transformers, and Unstructured with its extras for file parsing)*

    **Platform-Specific Installation Notes:**

    * **Apple Silicon Macs (M1/M2/M3):** If you encounter build errors related to `onnx` or SSE4.1 instructions:
        ```bash
        # Install PyTorch separately first
        pip install torch

        # Then install the core requirements without building problematic packages
        pip install --no-deps chromadb
        pip install pydantic hnswlib typing_extensions overrides
        pip install langchain langchain-community langchain-chroma langchain-huggingface sentence-transformers requests
        
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

    *(Optional: If you wish to use a different model supported by Ollama, pull the desired model (e.g., `ollama pull llama3`) and update the `LLM_MODEL_NAME` variable within the `app.py` script)*

## How to Use

1.  **Add Your Documents:** Place your knowledge base files into the `data/` directory.
    * Supported formats depend on the installed dependencies specified in `requirements.txt`. The default setup includes support for `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, and `.epub` via the `unstructured` library extras. See `requirements.txt` for details on enabling support for other types if needed.

2.  **Ensure Ollama is Running:** Before running the script, verify that the Ollama application/server is active in the background. See the "Ollama Not Running / Connection Errors" section under "Notes & Troubleshooting" for details on how to check this.

3.  **Run the Application:**
    Make sure your virtual environment is activated. Then, run the main script from your terminal:
    ```bash
    python app.py
    ```
    *(Or use `python3 app.py` if that's how you invoke Python 3 on your system)*

4.  **First Run:** The very first time you run the script (or after adding/changing documents and deleting the `vector_store/` directory), it will perform these steps:
    * Load all supported documents found in the `data/` directory.
    * Split the documents' content into smaller text chunks.
    * Generate numerical embeddings for each chunk (this can take significant time depending on the number/size of documents and your computer's processing power).
    * Create and save a local vector store containing these embeddings in the `vector_store/` directory. Please be patient during this process.

5.  **Subsequent Runs:** On future runs, if the `vector_store/` directory exists, the script will load the pre-computed embeddings directly from it, making the startup process much faster.

6.  **Ask Questions:** Once the script prints the `--- RAG System Ready ---` message, you can type your questions related to the content of your documents into the terminal and press Enter.

7.  **Commands:**
   * Type `exit` or `quit` to stop the application
   * Type `clear` to reset the conversation memory
   * Type `reindex` to reindex all documents
   * Type `stats` to see document statistics

## Folder Structure

```
CustomRAG/
    ├── data/                # <-- Place your knowledge base items there (e.g., .pdf, .docx, .txt)
    ├── vector_store/        # Stores the generated vector embeddings (created automatically, ignored by Git)
    ├── venv/                # Python virtual environment folder (if named 'venv', ignored by Git)
    ├── app.py               # Main application Python script
    ├── .gitignore           # Specifies intentionally untracked files/folders for Git
    ├── requirements.txt     # List of Python dependencies to install
    └── README.md            # This file
```

*(Note: Your virtual environment folder might have a different name like `.venv` or `RAGenv`. Ensure your `.gitignore` file lists the name you use)*

## Document Format Support

The system supports various document formats:

* **Text Files** (`.txt`, `.md`): Basic text files.
* **PDF Files** (`.pdf`): Using PyPDFLoader or PDFPlumberLoader.
* **Word Documents** (`.docx`, `.doc`): Using Docx2txtLoader.
* **PowerPoint Files** (`.ppt`, `.pptx`): Using UnstructuredPowerPointLoader.
* **EPUB Books** (`.epub`): Using either UnstructuredEPubLoader or a custom loader with ebooklib and html2text.

For EPUB files, the system first tries to use the UnstructuredEPubLoader, and if that fails, it falls back to a custom implementation using ebooklib, BeautifulSoup, and html2text.

## Code Modifications

If you encounter an error about a missing `split_documents` method, add the following method to the `DocumentProcessor` class in `app.py`:

```python
@staticmethod
def split_documents(documents: List, chunk_size: int = None, chunk_overlap: int = None) -> List:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk (in characters)
        chunk_overlap: Overlap between chunks (in characters)
        
    Returns:
        List of document chunks
    """
    if not documents:
        logger.warning("No documents to split")
        return []
        
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
        
    if chunk_overlap is None:
        chunk_overlap = Config.CHUNK_OVERLAP
        
    logger.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    try:
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Split documents
        chunked_documents = text_splitter.split_documents(documents)
        
        # Remove any empty chunks
        valid_chunks = [chunk for chunk in chunked_documents if chunk.page_content and chunk.page_content.strip()]
        
        # Log statistics
        logger.info(f"Created {len(valid_chunks)} chunks from {len(documents)} documents")
        
        return valid_chunks
        
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        return []
```

## Notes & Troubleshooting

* **Performance:** Running Large Language Models (LLMs) locally requires significant RAM (memory) and computational power. Performance will vary based on your computer's hardware specifications (CPU, GPU if used by Ollama, available RAM). The default `deepseek-r1:8b` model (around 5GB download, 8 Billion parameters) is reasonably balanced but typically requires several gigabytes of RAM while running. Expect responses to be slower than cloud-based API services. Larger models will demand more resources.

* **First Run Time:** As mentioned, generating embeddings for many or large documents can take time (potentially several minutes or longer). Subsequent runs are much faster as they load the saved `vector_store/`.

* **Updating Knowledge Base:** If you add, remove, or modify files in the `data/` directory, the existing information in `vector_store/` becomes outdated. To include the latest changes in the RAG system's knowledge, you can either:
  * Type `reindex` in the application interface to trigger reindexing, or
  * Delete the entire `vector_store/` directory, and the script will perform the full re-indexing process on its next run.

* **Ollama Not Running / Connection Errors:** The `app.py` script needs to connect to the Ollama application running as a background server. If the script fails immediately mentioning "connection refused," "could not connect," or similar network errors, the Ollama server is likely not running or accessible.
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
  * **EPUB support issues:** If you have trouble with EPUB files, make sure you have both `unstructured[epub]` and `html2text` packages installed.

* **`libmagic` Warning (Optional Enhancement):** During document loading, you might see a warning like `libmagic is unavailable...`. This message comes from the `unstructured` parsing library. `libmagic` is a system library that helps detect file types accurately based on their content, which can be more reliable than just using the file extension.
    * The script **will generally run fine without it**.
    * **Installation is optional** but can improve robustness.
    * **On macOS:** Install via Homebrew: `brew install libmagic`
    * **On Debian/Ubuntu Linux:** Use apt: `sudo apt-get update && sudo apt-get install libmagic1`
    * **On other systems:** Use your system's package manager to install the `libmagic` library. No `pip install` is usually needed for this specific warning; `unstructured` uses the system library if available.

* **`CropBox missing` Warnings (Informational):** When processing PDF files, you might see multiple warnings like `CropBox missing from /Page, defaulting to MediaBox`.
    * These indicate the PDF file is missing some optional layout metadata (the CropBox).
    * The PDF parsing library is informing you it's using a default value based on the page's physical size (the MediaBox).
    * These warnings are generally **safe to ignore** and usually don't negatively impact text extraction. Only investigate if you notice problems with the content retrieved from your PDFs.

* **Memory Issues (RAM):** If `app.py` crashes, especially during embedding or when answering questions, your system might be running out of RAM. Close other resource-heavy applications. If issues persist, consider using a smaller LLM via Ollama (e.g., `phi3:medium`, `llama3:8b`). Remember to `ollama pull` the new model name and update the `LLM_MODEL_NAME` variable in `app.py`.

* **Dependencies:** Ensure all Python packages from `requirements.txt` are installed in your *active* virtual environment. If you get `ImportError` (e.g., "ModuleNotFoundError"), double-check that your virtual environment is activated (you should see `(venv)` or similar in your prompt) and try running `pip install -r requirements.txt` again.

* **Using GPU Acceleration:** The system will automatically detect and use:
  * CUDA for NVIDIA GPUs
  * MPS for Apple Silicon Macs
  
  No additional configuration is needed as this is handled by the `VectorStoreManager.get_embedding_device()` method.