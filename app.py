import os
import shutil
import sqlite3
import time
import logging
import torch
import sys
import json
import re
from typing import List, Dict, Any, Optional, Union


from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader
# Remove the PowerPoint loader import as it's causing issues
# If you need PowerPoint support, install the 'unstructured' package with pip install "unstructured[pptx]"
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import requests

# --- Enhanced Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data")
PERSIST_PATH = os.path.join(SCRIPT_DIR, "vector_store")

# --- Improved Speed / Accuracy Parameters ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Consider "bge-large-en-v1.5" for even better accuracy
LLM_MODEL_NAME = "deepseek-r1:8b"  # Consider "llama3:8b", "phi3", or quantized models for speed
CHUNK_SIZE = 500  # Smaller chunks for better retrieval granularity
CHUNK_OVERLAP = 150  # Increased overlap to maintain context between chunks
RETRIEVER_K = 5  # Increased number of chunks to retrieve
RETRIEVER_SEARCH_TYPE = "mmr"  # Using MMR to encourage diversity in retrieved chunks
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama API URL - this is the API port, not the model runner port
FORCE_REINDEX = False  # Set to True to force reindexing of documents

# --- Predefined Responses for Common Interactions ---
GREETING_PATTERNS = [
    r'\b(?:hi|hello|hey|greetings|howdy|good\s*(?:morning|afternoon|evening)|what\'s\s*up)\b',
    r'\bhow\s+are\s+you\b',
]

GREETING_RESPONSES = [
    "Hello! I'm your document assistant. How can I help you with your documents today?",
    "Hi there! I'm ready to help answer questions about your documents. What would you like to know?",
    "Greetings! I'm here to assist with information from your documents. What are you looking for?",
]

FAREWELL_PATTERNS = [
    r'\b(?:bye|goodbye|see\s*you|farewell|exit|quit)\b',
]

# --- Helper Functions ---

def detect_greeting_or_farewell(query: str) -> Optional[str]:
    """Detects if the input is a greeting or farewell and returns an appropriate response."""
    query_lower = query.lower()

    # Check for greetings
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            import random
            return random.choice(GREETING_RESPONSES)

    # Check for farewells (but don't respond - let the main loop handle these)
    for pattern in FAREWELL_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return None

    # Not a greeting or farewell
    return None

def get_embedding_device():
    """Function to check for GPU and set device"""
    if torch.cuda.is_available():
        logging.info("CUDA GPU available. Using GPU for embeddings.")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # For Apple Silicon
        logging.info("Apple Silicon MPS available. Using MPS for embeddings.")
        return 'mps'
    else:
        logging.info("No GPU detected. Using CPU for embeddings.")
        return 'cpu'

def get_file_loader(file_path: str):
    """Returns the appropriate loader based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return PyPDFLoader(file_path)
    elif ext in ['.docx', '.doc']:
        return Docx2txtLoader(file_path)
    elif ext in ['.ppt', '.pptx']:
        try:
            return UnstructuredPowerPointLoader(file_path)
        except Exception as e:
            logging.error(f"Error loading PowerPoint file {file_path}: {str(e)}")
            return None
    elif ext in ['.txt', '.md', '.csv']:
        return TextLoader(file_path)
    else:
        logging.warning(f"Unsupported file type: {ext} for file {file_path}")
        return None

def load_documents_improved(path: str, glob_pattern: str):
    """Enhanced document loading with better file type handling and error reporting"""
    logging.info(f"Loading documents from: {path} using pattern: {glob_pattern}")

    try:
        # First, get all files matching the pattern
        all_files = []
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, path)
                if any(rel_path.endswith(ext) for ext in ['.pdf', '.txt', '.docx', '.doc', '.md', '.ppt', '.pptx']):
                    all_files.append(file_path)

        if not all_files:
            logging.warning(f"No compatible documents found in {path}. Looking for PDF, DOCX, TXT, MD, PPT/PPTX files.")
            return None

        logging.info(f"Found {len(all_files)} compatible files. Loading content...")

        # Load each file with its appropriate loader
        all_documents = []
        for file_path in all_files:
            try:
                logging.info(f"Loading: {file_path}")
                loader = get_file_loader(file_path)
                if loader:
                    docs = loader.load()
                    if docs:
                        # Add source metadata to each document
                        for doc in docs:
                            if not hasattr(doc, 'metadata') or doc.metadata is None:
                                doc.metadata = {}
                            doc.metadata['source'] = file_path
                            doc.metadata['filename'] = os.path.basename(file_path)

                        all_documents.extend(docs)
                        logging.info(f"Successfully loaded {len(docs)} sections from {file_path}")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")
                continue

        if not all_documents:
            logging.warning("No document content could be extracted successfully.")
            return None

        # Filter out empty documents
        valid_documents = [doc for doc in all_documents if doc.page_content and doc.page_content.strip()]

        logging.info(f"Successfully loaded {len(valid_documents)} document sections from {len(all_files)} files.")
        return valid_documents

    except Exception as e:
        logging.error(f"An error occurred during document loading: {e}", exc_info=True)
        return None

def split_documents(documents):
    """Split documents into smaller chunks with improved chunking strategy"""
    if not documents:
        return []

    logging.info(f"Splitting {len(documents)} document sections into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    # Use a more robust text splitter for better semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Prioritize splitting at paragraph breaks
        length_function=len,
        keep_separator=True
    )

    start_time = time.time()
    chunks = text_splitter.split_documents(documents)
    end_time = time.time()

    # Add some diagnostic info about chunk lengths
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunks) if chunks else 0

    logging.info(f"Split into {len(chunks)} chunks in {end_time - start_time:.2f} seconds.")
    logging.info(f"Average chunk length: {avg_length:.2f} characters")

    return chunks

def get_vector_store(chunks, embeddings, persist_directory):
    """Creates or loads the vector store using ChromaDB with improved error handling."""
    
    # If force reindexing or no store exists, reset the environment
    if FORCE_REINDEX or not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        reset_chroma_db(persist_directory)
    
    # If there's existing data, try to load it
    if os.path.exists(persist_directory) and os.listdir(persist_directory) and not FORCE_REINDEX and chunks is None:
        logging.info(f"Loading existing vector store from: {persist_directory}")
        start_time = time.time()
        try:
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            logging.info(f"Loaded vector store with {vector_store._collection.count()} documents in {time.time() - start_time:.2f} seconds.")
            return vector_store
        except Exception as e:
            logging.error(f"Error loading vector store: {e}", exc_info=True)
            logging.info("Will try to create a new vector store...")
            reset_chroma_db(persist_directory)
    
    # Create a new vector store
    if chunks:
        logging.info(f"Creating new vector store in: {persist_directory}")
        logging.info(f"Embedding {len(chunks)} chunks using {embeddings.model_name} on device {embeddings.client.device}...")
        
        try:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            logging.info(f"Vector store created successfully with {len(chunks)} chunks.")
            return vector_store
        except sqlite3.OperationalError as sqlerr:
            logging.error(f"SQLite error: {sqlerr}")
            if "readonly database" in str(sqlerr):
                logging.error("Database is readonly. This usually means permission issues or a locked database.")
                logging.error("Try manually deleting the vector_store directory and restarting the application.")
            return None
        except Exception as e:
            logging.error(f"Error creating vector store: {e}", exc_info=True)
            return None
    else:
        logging.error("No chunks provided and no existing store found. Cannot create vector store.")
        return None

# --- RAG Chain Setup ---

def format_docs(docs):
    """Format retrieved documents for inclusion in the prompt with source information."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown source')
        filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')

        formatted_text = f"Document {i+1} (from {filename}):\n{doc.page_content}\n\n"
        formatted_docs.append(formatted_text)

    return "\n".join(formatted_docs)

def stream_ollama_response(prompt, model_name):
    """Stream response from Ollama API directly, showing tokens as they come"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": True
    }

    full_response = ""

    # Add more detailed logging
    logging.info(f"Connecting to Ollama API at {url}")
    logging.info(f"Using model: {model_name}")

    try:
        # Test connection to Ollama API first
        try:
            test_url = f"{OLLAMA_BASE_URL}/api/tags"
            test_response = requests.get(test_url, timeout=5)
            if test_response.status_code == 200:
                logging.info(f"Successfully connected to Ollama API. Models available: {len(test_response.json().get('models', []))}")
            else:
                logging.error(f"Ollama API test failed: {test_response.status_code} - {test_response.text}")
        except Exception as test_error:
            logging.error(f"Failed to connect to Ollama API for testing: {str(test_error)}")

        # Proceed with the actual request
        with requests.post(url, headers=headers, json=data, stream=True, timeout=10) as response:
            if response.status_code != 200:
                logging.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error: Failed to generate response (HTTP {response.status_code})"

            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line.decode('utf-8'))
                        if 'response' in json_line:
                            token = json_line['response']
                            full_response += token
                            # Just print the token, don't interpret any content as follow-up questions
                            print(token, end='', flush=True)

                        # Check if it's the final message
                        if json_line.get('done', False):
                            break
                    except json.JSONDecodeError:
                        logging.error(f"Failed to parse JSON: {line}")
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        return f"Error during generation: {str(e)}"

    print()  # Add a newline at the end for better formatting
    return full_response

def setup_rag_chain_with_memory(vector_store, llm_model_name, retriever_k, retriever_search_type):
    """Sets up the improved RAG chain with better prompting and conversation handling."""
    if vector_store is None:
        logging.error("Vector store is not initialized. Cannot setup RAG chain.")
        return None, None, None  # Return None for chain, retriever, and memory

    logging.info(f"Configuring retriever (type={retriever_search_type}, k={retriever_k})")
    retriever_kwargs = {'k': retriever_k}
    if retriever_search_type == "mmr":
        retriever_kwargs['fetch_k'] = retriever_k * 3  # Fetch 3x chunks before MMR diversification
        retriever_kwargs['lambda_mult'] = 0.5  # Balance between relevance and diversity

    retriever = vector_store.as_retriever(
        search_type=retriever_search_type,
        search_kwargs=retriever_kwargs
    )

    # Improved system prompt with better instructions
    template = """
    You are a helpful AI assistant answering questions about a collection of documents. You have access to information contained in these documents, and you'll answer questions about their content.

    Guidelines for your responses:
    1. If you find the answer in the documents, respond directly and cite your sources
    2. If the documents contain partial information, use it and make clear where your information comes from
    3. If the question is about general knowledge unrelated to the documents, you can answer it like a normal conversation
    4. If the question is a greeting or casual conversation, respond naturally as a friendly assistant
    5. If you cannot find specific information in the documents, say "I don't have specific information about that in the documents" rather than making up information
    6. Always use specific information from the documents when available instead of giving generic answers

    Context from retrieved documents:
    {context}

    Chat History:
    {chat_history}

    Question: {question}

    Helpful Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    logging.info(f"Initializing LLM: {llm_model_name}")
    # Create standard Ollama instance for non-streaming scenarios
    llm = Ollama(model=llm_model_name)

    # Initialize the conversation memory with a system message
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )

    # Add a system message to memory
    system_message = SystemMessage(content="I am an AI assistant that helps with answering questions about documents. I can also engage in casual conversation.")
    memory.chat_memory.messages.append(system_message)

    # Create an improved implementation of the chain
    def rag_chain(input_dict, stream=True):
        # Get the question
        question = input_dict.get("question", "")

        # Check if it's a greeting or simple conversational message
        greeting_response = detect_greeting_or_farewell(question)
        if greeting_response:
            return greeting_response

        # Retrieve context docs based on the question
        try:
            context_docs = retriever.invoke(question)
            context = format_docs(context_docs)
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            context = "Error retrieving documents."

        # Format chat history
        chat_history = input_dict.get("chat_history", [])
        formatted_history = ""
        for message in chat_history:
            if isinstance(message, HumanMessage):
                formatted_history += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                formatted_history += f"AI: {message.content}\n"
            elif isinstance(message, SystemMessage):
                # Skip system messages in the formatted history
                continue

        # Prepare the prompt input
        prompt_input = {
            "context": context,
            "chat_history": formatted_history,
            "question": question
        }

        # Generate the prompt message
        chain_response = prompt.invoke(prompt_input).to_string()

        # Add some debugging about document retrieval
        logging.info(f"Retrieved {len(context_docs)} document chunks for query: '{question}'")

        # Use streaming or regular response based on parameter
        if stream:
            # Call the direct Ollama API for streaming
            response = stream_ollama_response(chain_response, llm_model_name)
            return response
        else:
            # Use the standard LangChain Ollama for non-streaming
            response = llm.invoke(chain_response)
            print(response)
            return response

    logging.info("RAG chain with conversation memory created successfully.")
    return rag_chain, retriever, memory

def reset_chroma_db(persist_directory):
    """Completely reset the ChromaDB environment to fix readonly issues"""
    logging.info(f"Resetting ChromaDB environment at {persist_directory}")
    
    # First, clear any global references that might keep connections open
    try:
        import gc
        gc.collect()  # Force garbage collection to release any lingering connections
        time.sleep(0.5)  # Give system time to release resources
    except Exception as e:
        logging.warning(f"Error during garbage collection: {e}")
    
    # Check if directory exists
    if os.path.exists(persist_directory):
        try:
            # On macOS, try to fix permissions before removal
            if sys.platform == 'darwin':
                # Attempt to fix permissions recursively
                logging.info(f"Attempting to fix permissions on {persist_directory}")
                try:
                    os.system(f"chmod -R 755 {persist_directory}")
                except Exception as e:
                    logging.warning(f"Error fixing permissions: {e}")
            
            # Try different approaches to remove the directory
            try:
                # Try Python's built-in removal
                shutil.rmtree(persist_directory)
            except Exception as rm_error:
                logging.warning(f"Failed to remove using shutil.rmtree: {rm_error}")
                # Fall back to system commands if Python fails
                if sys.platform == 'darwin' or sys.platform.startswith('linux'):
                    logging.info("Attempting fallback removal using system command")
                    os.system(f"rm -rf {persist_directory}")
                elif sys.platform == 'win32':
                    os.system(f"rd /s /q {persist_directory}")
        except Exception as e:
            logging.error(f"Failed to reset Chroma directory: {e}")
            return False
    
    # Create a fresh directory with proper permissions
    try:
        # Ensure parent directory exists
        parent_dir = os.path.dirname(persist_directory)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # Create the vector store directory with proper permissions
        os.makedirs(persist_directory, exist_ok=True)
        
        # Set appropriate permissions
        if sys.platform != 'win32':  # Skip on Windows
            os.chmod(persist_directory, 0o755)  # rwxr-xr-x
        
        return True
    except Exception as e:
        logging.error(f"Failed to create new Chroma directory: {e}")
        return False

# --- Enhanced Application Entry Point ---

if __name__ == "__main__":
    logging.info("--- Starting Enhanced Local RAG Application with Conversation Memory ---")

    # Initialize embedding model (with device selection)
    embedding_device = get_embedding_device()
    logging.info(f"Using device for embeddings: {embedding_device}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': embedding_device},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
        )
    except Exception as e:
        logging.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
        exit(1)  # Exit if embeddings can't load

    vector_store = None

    # Check if we need to create or update the vector store
    if FORCE_REINDEX or not os.path.exists(PERSIST_PATH) or not os.listdir(PERSIST_PATH):
        if FORCE_REINDEX:
            logging.info("Force reindexing is enabled. Creating new vector store...")
        else:
            logging.info("No existing vector store found. Creating a new one...")

        # Improved document loading and processing
        documents = load_documents_improved(DATA_PATH, "**/*.*")
        if documents:
            chunks = split_documents(documents)
            if chunks:
                vector_store = get_vector_store(chunks, embeddings, PERSIST_PATH)
            else:
                logging.error("No processable content found after splitting documents.")
        else:
            logging.error(f"Document loading failed. Check '{DATA_PATH}' directory.")
            logging.info("Required libraries for document loading:")
            logging.info(" - pip install 'unstructured[all]' pypdf python-docx")
            logging.info(" - For PDFs: pip install pypdf")
            logging.info(" - For Word: pip install python-docx")
            logging.info(" - For PowerPoint: pip install python-pptx unstructured")

        if not vector_store:
             logging.error("Failed to create the vector store. Exiting.")
             exit(1)
    else:
        # Load existing store
        vector_store = get_vector_store(None, embeddings, PERSIST_PATH)
        if not vector_store:
            logging.error("Failed to load the existing vector store. Exiting.")
            exit(1)

    # Setup RAG Chain with enhanced memory
    rag_chain, retriever, memory = setup_rag_chain_with_memory(vector_store, LLM_MODEL_NAME, RETRIEVER_K, RETRIEVER_SEARCH_TYPE)

    if rag_chain:
        logging.info("\n--- Enhanced RAG System Ready ---")
        print("\n" + "="*50)
        print("📚 Document Assistant 📚")
        print("="*50)
        print("Ask questions about the documents in your 'data' folder.")
        print("The system now handles casual conversations and remembers context.")
        print("Commands:")
        print("  - Type 'exit' or 'quit' to stop")
        print("  - Type 'clear' to reset the conversation memory")
        print("  - Type 'reindex' to force reindexing of all documents")
        print("="*50 + "\n")

        # Enhanced interaction loop
        while True:
            try:
                query = input("\nYour Question: ")
                query = query.strip()

                if not query:
                    print("Please enter a question.")
                    continue

                if query.lower() in ["exit", "quit", "bye", "goodbye"]:
                    logging.info("\nExiting...")
                    print("Goodbye! Have a great day!")
                    break

                if query.lower() == "clear":
                    logging.info("Clearing conversation memory...")
                    # Reset memory but keep the system message
                    system_message = memory.chat_memory.messages[0] if memory.chat_memory.messages else None
                    memory.clear()
                    if system_message:
                        memory.chat_memory.messages.append(system_message)
                    print("Conversation memory has been reset.")
                    continue

                if query.lower() == "reindex":
                    logging.info("Triggering reindexing of documents...")
                    print("Reindexing documents. This may take a while...")
                    
                    # First, close any existing resources
                    if 'vector_store' in locals() and vector_store is not None:
                        try:
                            # Try to explicitly release the collection
                            if hasattr(vector_store, '_collection'):
                                vector_store._collection = None
                            vector_store = None
                        except Exception as e:
                            logging.warning(f"Error releasing vector store: {e}")
                    
                    if 'retriever' in locals() and retriever is not None:
                        retriever = None
                    
                    # Now completely reset the ChromaDB environment
                    db_reset_success = reset_chroma_db(PERSIST_PATH)
                    
                    if not db_reset_success:
                        print("Failed to reset the database environment. Please restart the application.")
                        continue
                    
                    # Now reload and reindex with fresh environment
                    try:
                        # Reload documents
                        documents = load_documents_improved(DATA_PATH, "**/*.*")
                        if not documents:
                            print("No documents found to index. Please check your data directory.")
                            continue
                        
                        # Split into chunks
                        chunks = split_documents(documents)
                        if not chunks:
                            print("Failed to split documents into chunks.")
                            continue
                        
                        # Create new vector store in clean environment
                        vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=PERSIST_PATH
                        )
                        
                        # Recreate the chain
                        rag_chain, retriever, memory = setup_rag_chain_with_memory(
                            vector_store, LLM_MODEL_NAME, RETRIEVER_K, RETRIEVER_SEARCH_TYPE
                        )
                        
                        print("Reindexing completed successfully!")
                    except sqlite3.OperationalError as sqlerr:
                        logging.error(f"SQLite error during reindexing: {sqlerr}")
                        if "readonly database" in str(sqlerr):
                            print("\nERROR: The database is still locked or readonly.")
                            print("Try these steps:")
                            print("1. Exit this application completely")
                            print("2. Delete the vector_store directory manually: rm -rf vector_store")
                            print("3. Restart the application")
                        else:
                            print(f"Database error: {sqlerr}")
                    except Exception as reindex_error:
                        logging.error(f"Error during reindexing: {reindex_error}", exc_info=True)
                        print(f"Reindexing failed: {str(reindex_error)}")
                    
                    continue

                logging.info(f"Processing query: '{query}'")
                start_time = time.time()

                # Add the user's query to memory
                memory.chat_memory.add_user_message(query)

                print("\nThinking...\n")

                try:
                    # Build the input dictionary with question and chat history
                    input_dict = {
                        "question": query,
                        "chat_history": memory.chat_memory.messages  # include all messages
                    }

                    # Get streaming response
                    response = rag_chain(input_dict)
                    print("\nResponse: ", response)

                    # Add the AI's response to memory
                    memory.chat_memory.add_ai_message(response)

                except Exception as stream_error:
                     logging.error(f"\nError during LLM processing: {stream_error}", exc_info=True)
                     print("\n[Error occurred during response generation]")

                end_time = time.time()
                logging.info(f"Query processed in {end_time - start_time:.2f} seconds")

            except KeyboardInterrupt:
                 logging.info("\nExiting...")
                 print("\nGoodbye! Have a great day!")
                 break
            except Exception as e:
                logging.error(f"\nAn error occurred: {e}", exc_info=True)
                print("\n[An unexpected error occurred. Please try again.]")

    else:
        logging.error("\n--- RAG System Initialization Failed ---")
        print("Could not set up the RAG chain. Please check logs for errors.")

    logging.info("\n--- RAG Application Finished ---")