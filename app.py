import os
import shutil
import time # Added for potential timing feedback

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
        search_kwargs={'k': 3} # Retrieve top 3 relevant chunks – increase it if want more comprehensive answer and completeness is crucial
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
    if not os.path.exists(PERSIST_PATH) or not os.listdir(PERSIST_PATH):
        print("No existing vector store found or directory is empty. Creating a new one...")
        documents = load_documents(DATA_PATH, LOADER_GLOB)
        if documents:
            chunks = split_documents(documents)
            if chunks:
                vector_store = get_vector_store(chunks, embeddings, PERSIST_PATH)
            else:
                print("No processable content found in documents after splitting.")
        else:
            print("Could not create vector store because no documents were loaded.")
            print(f"Please ensure files (.txt, .md, .pdf, .docx, .pptx) exist in '{DATA_PATH}'")
            print("And that you have run 'pip install -r requirements.txt'.") # Removed 'after updating it' for clarity
    else:
        print(f"Found existing vector store.")
        vector_store = get_vector_store(None, embeddings, PERSIST_PATH)

    if vector_store:
        # --- Prepare RAG components ---
        print("Setting up RAG components...")

        # 1. Retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 3} # Retrieve top 3 relevant chunks – increase it if want more comprehensive answer and completeness is crucial
        )
        print(f"Retriever configured (k={retriever.search_kwargs.get('k', 'default')})")

        # 2. Prompt Template
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
        print("Prompt template prepared.")

        # 3. LLM
        llm = Ollama(model=LLM_MODEL_NAME)
        print(f"Initialized LLM with model: {LLM_MODEL_NAME}")

        # 4. Output Parser
        output_parser = StrOutputParser()
        print("Output parser prepared.")

        # Helper function to format docs (can be defined here or kept global)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        print("\n--- RAG System Ready ---")
        print("Ask questions about the documents in your 'data' folder.")
        print("Type 'exit' or 'quit' to stop.")

        # --- Interaction loop with enhanced thinking process output ---
        while True:
            try:
                query = input("\nYour Question: ")
                if query.strip().lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break
                if not query.strip():
                    print("Please enter a question.")
                    continue

                print("\n--- Thinking Process ---")
                start_time = time.time()

                # Step 1: Retrieve relevant documents
                print(f"1. Retrieving documents for: '{query[:50]}...'") # Show truncated query
                retrieval_start_time = time.time()
                retrieved_docs = retriever.invoke(query)
                retrieval_end_time = time.time()
                print(f"   Retrieved {len(retrieved_docs)} chunks in {retrieval_end_time - retrieval_start_time:.2f}s.")

                if not retrieved_docs:
                     print("   No relevant documents found in the vector store for this query.")
                     # Decide if you want to proceed without context or stop
                     print("\nAnswer:")
                     print("I could not find relevant information in the provided documents to answer this question.")
                     continue # Skip LLM call if no docs found

                # Display sources of retrieved documents (if metadata exists)
                print("   Sources found:")
                sources = set(doc.metadata.get('source', 'Unknown') for doc in retrieved_docs)
                for source in sources:
                    print(f"     - {os.path.basename(source)}") # Show just the filename

                # Step 2: Format context
                print("2. Formatting retrieved documents into context...")
                formatted_context = format_docs(retrieved_docs)
                # Optional: Print context length or snippet
                # print(f"   Context length: {len(formatted_context)} characters")

                # Step 3: Prepare prompt and stream LLM generation
                print(f"3. Generating answer using LLM '{LLM_MODEL_NAME}' (streaming)...")
                generation_start_time = time.time()

                # Define the generation part of the chain (same as before)
                rag_chain_generation = (
                    prompt
                    | llm
                    | output_parser # StrOutputParser handles streaming correctly
                )

                print("\n--- Streaming LLM Output (including <think> process) ---")
                # Use stream() and iterate through the chunks
                try:
                    full_response_content = "" # To potentially store the full response if needed
                    stream = rag_chain_generation.stream({
                        "context": formatted_context,
                        "question": query
                    })
                    for chunk in stream:
                        print(chunk, end="", flush=True)
                        full_response_content += chunk # Optionally accumulate the full response

                    print() 

                except Exception as stream_error:
                    print(f"\nAn error occurred during streaming: {stream_error}")
                    import traceback
                    traceback.print_exc()

                generation_end_time = time.time()
                print("\n--- End Streaming LLM Output ---")
                print(f"   Answer streamed in {generation_end_time - generation_start_time:.2f}s.")

                # Step 4: Display final answer
                total_end_time = time.time()
                print("--- End Thinking Process ---")
                print(f"(Total query time: {total_end_time - start_time:.2f}s)")

            except Exception as e:
                print(f"\nAn error occurred during query processing: {e}")
                import traceback
                traceback.print_exc()

    else:
        print("\n--- RAG System Initialization Failed ---")
        print("Could not load or create the vector store.")
        print(f"Please check the '{DATA_PATH}' directory for supported documents")
        print("and ensure all dependencies from 'requirements.txt' are installed.")

    print("\n--- RAG Application Finished ---")