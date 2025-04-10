# Custom RAG

This project implements a simple Retrieval-Augmented Generation (RAG) system that runs entirely locally on your machine. It uses documents you provide as a knowledge base to answer your questions.

## Features

* **Local First:** Runs completely offline (after downloading models). No API keys or data sent to external services (unless you change the LLM/Embedding configuration).

* **Custom Knowledge Base:** Simply place your `.txt`, `.md` (and potentially other types like `.pdf` if dependencies are installed) files into the `data` directory.

* **Open Source Stack:** Built using popular Python libraries:
    * **Orchestration:** LangChain
    * **LLM:** Ollama (running models like DeepSeek locally)
    * **Embeddings:** Sentence Transformers (Hugging Face `all-MiniLM-L6-v2`)
    * **Vector Store:** ChromaDB (local persistent storage)

## Prerequisites

* **Python:** Version 3.8 or higher.
* **Ollama:** Must be installed and running. Download from [https://ollama.com/](https://ollama.com/).
* **Git:** For cloning the repository (if applicable).

## Installation

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <your-github-repo-url>
    cd CustomRAG
    ```

2.  **Create and Activate Python Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows use: .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the LLM via Ollama:**
    This project is configured to use `deepseek-r1:8b`. Pull it using:
    ```bash
    ollama pull deepseek-r1:8b
    ```
    *(Ensure Ollama is running in the background)*

    *(Optional: If you want to use a different Ollama model, pull it and update `LLM_MODEL_NAME` in `app.py`)*

## How to Use

1.  **Add Your Documents:** Place your knowledge base files (e.g., `.txt`, `.md`) into the `data/` directory. The system will read these files on its first run (or if the `vector_store/` directory is deleted).
    * *Note:* If you want PDF support, ensure you have `pypdf` installed (`pip install pypdf`) and uncomment the relevant line in `requirements.txt` if needed. `unstructured` should handle basic types.

2.  **Run the Application:**
    ```bash
    python app.py
    ```

3.  **First Run:** The first time you run the script (or after adding/changing documents and deleting the `vector_store/` directory), it will:
    * Load the documents from `data/`.
    * Split them into chunks.
    * Generate embeddings (this might take some time depending on the number of documents and your CPU/M2 chip).
    * Create and save a local vector store in the `vector_store/` directory.

4.  **Subsequent Runs:** If the `vector_store/` directory exists, the script will load the pre-computed embeddings, making startup much faster.

5.  **Ask Questions:** Once you see the "RAG System Ready..." message, type your questions into the terminal and press Enter.

6.  **Exit:** Type `exit` or `quit` to stop the application.

## Folder Structure

```
CustomRAG/
├── data/             # <-- PLACE YOUR KNOWLEDGE BASE FILES HERE (.txt, .md, etc.)
├── vector_store/     # Stores the generated vector embeddings (created automatically)
├── venv/             # Python virtual environment (if created)
├── app.py            # Main application script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Notes & Troubleshooting

* **Performance:** Running LLMs locally requires significant RAM and computational power. Performance will vary based on your specific M2 chip (Pro, Max) and available RAM. The `deepseek-r1:8b` model (around 5GB download, 8B parameters) is reasonably balanced, but expect it to use several gigabytes of RAM when loaded and be slower than smaller models or API-based services. Larger models will require more resources.

* **First Run Time:** Generating embeddings (the numerical representation of your documents) for a large number of documents can take a while, potentially several minutes or more depending on the size of your knowledge base and your hardware. Please be patient on the first run or after clearing the `vector_store/`. Subsequent runs that load the existing store will be much faster.

* **Updating Knowledge Base:** If you add, remove, or modify files in the `data/` directory, the existing vector store in `vector_store/` will become outdated. To force the system to re-read your files and rebuild the index with the latest information, you **must delete the entire `vector_store/` directory** and then restart `app.py`.

* **Ollama Not Running / Connection Errors:** The `app.py` script **requires** the Ollama application to be running in the background because it acts as a server providing access to the `deepseek-r1:8b` model. If `app.py` fails immediately with errors mentioning "connection refused," "could not connect," or similar network issues, it almost certainly means the Ollama server isn't active.
    * **How to Check if Ollama is Running:**
        * **Menu Bar (macOS - Easiest):** Look for the Ollama llama icon 🦙 in your Mac's menu bar (usually at the top-right of the screen). If the icon is visible, the server application is running.
        * **Terminal Command:** Open your Terminal (the same place you run `python app.py`) and type `ollama list`. Press Enter. If the command executes successfully (showing your downloaded models, like `deepseek-r1:8b`, or just an empty list if none are fully downloaded) without any connection errors, then the server is running and accessible. If you get an error like `Error: could not connect to Ollama server`, it is not running or not accessible.
        * **Activity Monitor (macOS):** You can also open `Activity Monitor` (use Spotlight: `Cmd + Space`, type `Activity Monitor`, press Enter), go to the search bar in the top-right of the window, and type `Ollama`. If you see active `Ollama` processes listed, it is running.
    * **How to Start Ollama:** If you've confirmed Ollama is not running, simply start the Ollama application. Find it in your `/Applications` folder or search for "Ollama" using Spotlight (`Cmd + Space`) and launch it. This will start the necessary background server and usually make the menu bar icon appear.

* **`libmagic` Warning (Optional Enhancement):** You might see a warning like `libmagic is unavailable...`. This comes from the document parsing library (`unstructured`). `libmagic` helps detect file types more accurately based on content, not just extension.
    * The script **will run fine without it**, typically using file extensions.
    * **Installation is optional but recommended** for potentially more robust file handling.
    * **On macOS:** Install via Homebrew: `brew install libmagic`
    * **On other OS:** Use your system's package manager (e.g., `apt-get install libmagic1` on Debian/Ubuntu). No `pip install` is needed for this specific warning.

* **`CropBox missing` Warnings (Informational):** When processing PDFs, you might see multiple warnings like `CropBox missing from /Page, defaulting to MediaBox`.
    * These indicate the PDF file is missing some optional layout metadata.
    * The parsing library is informing you it's using a default value (the page's physical size).
    * These warnings are generally **safe to ignore** and usually don't affect the text extraction. Only investigate further if you find the content retrieved from your PDFs is inaccurate or incomplete.

* **Memory Issues (RAM):** If the `app.py` script crashes unexpectedly, especially when processing many documents or asking complex questions, you might be running out of available RAM. The `deepseek-r1:8b` model itself requires a substantial amount of RAM to load and run inferences. Try closing other memory-intensive applications. If problems persist, you might consider trying a smaller Ollama model (e.g., `phi3:medium` or `llama3:8b`) by changing `LLM_MODEL_NAME` in `app.py` (remember to `ollama pull` the new model first).

* **Dependencies:** Ensure all Python packages listed in `requirements.txt` are correctly installed within your active virtual environment (`venv`). If you encounter `ImportError` messages when running `app.py`, double-check your virtual environment is activated (`source venv/bin/activate`) and try running `pip install -r requirements.txt` again.