import os
import shutil
import time

# Use DirectoryLoader from langchain_community
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
# Define paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data")
PERSIST_PATH = os.path.join(SCRIPT_DIR, "vector_store")

# Embedding model (running locally using Sentence Transformers)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Ollama LLM model name (ensure this model is pulled: 'ollama pull deepseek-r1:8b')
LLM_MODEL_NAME = "deepseek-r1:8b"

# File types to process (glob patterns)
# Default "**/*.*" loads all files, letting DirectoryLoader figure them out.
# If you want to be specific, you could use: "**/*.[pP][dD][fF]", "**/*.[dD][oO][cC][xX]", etc.
# For simplicity, we'll let DirectoryLoader try everything.
LOADER_GLOB = "**/*.*"

# --- Helper Functions ---

def load_documents(path, glob_pattern):
    """
    Loads documents from the specified directory using DirectoryLoader.
    Handles various formats like .txt, .md, .pdf, .docx, .pptx
    if the required dependencies are installed (via 'unstructured' extras).
    """
    print(f"Loading documents from: {path} using pattern: {glob_pattern}")
    # DirectoryLoader uses UnstructuredFileLoader by default for many types.
    # Ensure 'unstructured' and its dependencies (e.g., pypdf, python-pptx, python-docx)
    # are installed via requirements.txt.
    loader = DirectoryLoader(
        path,
        glob=glob_pattern,
        show_progress=True,
        use_multithreading=True, # Can speed up loading
        # loader_kwargs={'extract_images': False} # Example: Optionally configure underlying loaders
        silent_errors=True # Set to False to see errors from individual file loads
    )
    try:
        start_time = time.time()
        documents = loader.load()
        end_time = time.time()

        if not documents:
            print(f"No documents were loaded from {path}. Please ensure files exist and required libraries are installed.")
            # You might want to check specific dependencies here if needed.
            # e.g., check if 'pypdf' is installed if you expect PDFs.
            return None

        print(f"Loaded {len(documents)} document sections in {end_time - start_time:.2f} seconds.")
        # Note: A single file can be split into multiple 'documents' by the loader based on its structure.
        # You can optionally log the names of loaded files if needed (requires more complex loader setup or post-processing).

        # Filter out documents with empty content, which can sometimes happen
        valid_documents = [doc for doc in documents if doc.page_content.strip()]
        if len(valid_documents) < len(documents):
             print(f"Filtered out {len(documents) - len(valid_documents)} documents with empty content.")

        if not valid_documents:
             print("No documents with valid content found after filtering.")
             return None

        return valid_documents

    except Exception as e:
        print(f"An error occurred during document loading: {e}")
        # Specific errors might indicate missing dependencies, e.g., for specific file types.
        return None


def split_documents(documents):
    """Splits documents into smaller chunks."""
    if not documents:
        return []
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Size of each chunk in characters
        chunk_overlap=200 # Characters overlap between chunks
    )
    start_time = time.time()
    chunks = text_splitter.split_documents(documents)
    end_time = time.time()
    print(f"Split into {len(chunks)} chunks in {end_time - start_time:.2f} seconds.")
    return chunks

def get_vector_store(chunks, embeddings, persist_directory):
    """Creates or loads the vector store using ChromaDB."""
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vector store from: {persist_directory}")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print(f"Creating new vector store in: {persist_directory}")
        if not chunks:
             print("No chunks provided. Cannot create vector store.")
             return None
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        print(f"Embedding {len(chunks)} chunks... This may take some time.")
        start_time = time.time()
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        end_time = time.time()
        print(f"Vector store created and persisted in {end_time - start_time:.2f} seconds.")
    return vector_store

# --- Main RAG Logic ---

def setup_rag_chain(vector_store, llm_model_name):
    """Sets up the RAG chain using LCEL."""
    if vector_store is None:
        print("Vector store is not initialized. Cannot setup RAG chain.")
        return None

    # Configure the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", # Or "mmr" for Max Marginal Relevance
        search_kwargs={'k': 5} # Retrieve top 5 relevant chunks
    )

    # Define the prompt template - Adjust as needed for better responses
    template = """
    You are a helpful assistant. Answer the question based only on the following context provided.
    If the context does not contain the answer, state that you don't have enough information from the provided documents.
    Do not make up information. Be concise.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the Ollama LLM
    # You can add other parameters like temperature, top_k, etc.
    llm = Ollama(model=llm_model_name)
    print(f"Initialized LLM with model: {llm_model_name}")

    # Format context for the prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser() # Parses the LLM output into a string
    )

    return rag_chain

# --- Application Entry Point ---

if __name__ == "__main__":
    print("--- Starting Local RAG Application ---")

    # Initialize embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vector_store = None
    # Check if vector store needs to be created or updated
    # Simple check: If directory exists, load it. Otherwise, create it.
    # For robust updates, you might need more complex logic (e.g., check file modification times).
    if not os.path.exists(PERSIST_PATH) or not os.listdir(PERSIST_PATH):
        print("No existing vector store found or directory is empty. Creating a new one...")
        # Load documents
        documents = load_documents(DATA_PATH, LOADER_GLOB)
        if documents:
            # Split documents
            chunks = split_documents(documents)
            # Create and persist vector store if chunks exist
            if chunks:
                 vector_store = get_vector_store(chunks, embeddings, PERSIST_PATH)
            else:
                 print("No processable content found in documents after splitting.")
        else:
            print("Could not create vector store because no documents were loaded.")
            print(f"Please ensure files (.txt, .md, .pdf, .docx, .pptx) exist in '{DATA_PATH}'")
            print("And that you have run 'pip install -r requirements.txt' after updating it.")
    else:
        # Load existing vector store
        print(f"Found existing vector store.")
        vector_store = get_vector_store(None, embeddings, PERSIST_PATH) # Pass None for chunks as we are loading

    # Proceed only if vector store is successfully loaded or created
    if vector_store:
        # Setup RAG chain
        rag_chain = setup_rag_chain(vector_store, LLM_MODEL_NAME)

        if rag_chain:
            print("\n--- RAG System Ready ---")
            print("Ask questions about the documents in your 'data' folder.")
            print("Type 'exit' or 'quit' to stop.")
            # Interaction loop
            while True:
                try:
                    query = input("\nYour Question: ")
                    if query.strip().lower() in ["exit", "quit"]:
                        print("Exiting...")
                        break
                    if query.strip():
                        print("Thinking...")
                        start_time = time.time()
                        # Invoke the RAG chain with the user's query
                        response = rag_chain.invoke(query)
                        end_time = time.time()
                        print(f"\nAnswer (took {end_time - start_time:.2f}s):")
                        print(response)
                    else:
                        print("Please enter a question.")
                except Exception as e:
                     print(f"\nAn error occurred during query processing: {e}")
                     # Optionally add more robust error handling or logging
    else:
        print("\n--- RAG System Initialization Failed ---")
        print("Could not load or create the vector store.")
        print(f"Please check the '{DATA_PATH}' directory for supported documents")
        print("and ensure all dependencies from 'requirements.txt' are installed.")

    print("\n--- RAG Application Finished ---")