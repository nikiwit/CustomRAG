"""
CustomRAG - A Retrieval Augmented Generation System for Local Document Q&A
--------------------------------------------------------------------------
This application allows users to query their own document collection using 
natural language. It supports various document formats including PDF, DOCX, 
TXT, and PowerPoint files.

Features:
- Document embedding and vector storage using ChromaDB
- Semantic search with similarity matching
- Support for multiple file formats
- Conversation memory for contextual follow-up questions
- GPU acceleration when available (CUDA/MPS)

Author: Nik
License: MIT
"""

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
from pathlib import Path
from chromadb.config import Settings

# EPUB processing imports
import html2text
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

# Document processing imports
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredEPubLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store and embedding imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LLM and prompt imports
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import requests

# Configure logging - production settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("customrag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CustomRAG")

#############################################################################
# CONFIGURATION
#############################################################################

class Config:
    """Configuration settings for the RAG application."""
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.environ.get("CUSTOMRAG_DATA_PATH", os.path.join(SCRIPT_DIR, "data"))
    PERSIST_PATH = os.environ.get("CUSTOMRAG_VECTOR_PATH", os.path.join(SCRIPT_DIR, "vector_store"))
    
    # Embedding and retrieval settings
    EMBEDDING_MODEL_NAME = os.environ.get("CUSTOMRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL_NAME = os.environ.get("CUSTOMRAG_LLM_MODEL", "deepseek-r1:8b")
    CHUNK_SIZE = int(os.environ.get("CUSTOMRAG_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.environ.get("CUSTOMRAG_CHUNK_OVERLAP", "150"))
    RETRIEVER_K = int(os.environ.get("CUSTOMRAG_RETRIEVER_K", "10"))
    RETRIEVER_SEARCH_TYPE = os.environ.get("CUSTOMRAG_SEARCH_TYPE", "mmr")
    
    # Ollama API
    OLLAMA_BASE_URL = os.environ.get("CUSTOMRAG_OLLAMA_URL", "http://localhost:11434")
    
    # Miscellaneous
    FORCE_REINDEX = os.environ.get("CUSTOMRAG_FORCE_REINDEX", "False").lower() in ("true", "1", "t")
    LOG_LEVEL = os.environ.get("CUSTOMRAG_LOG_LEVEL", "INFO")
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc', '.md', '.ppt', '.pptx', '.epub']

    @classmethod
    def setup(cls):
        """Set up the configuration and ensure directories exist."""
        # Set logging level based on configuration
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Ensure data directory exists
        os.makedirs(cls.DATA_PATH, exist_ok=True)
        
        logger.info(f"Data directory: {cls.DATA_PATH}")
        logger.info(f"Vector store directory: {cls.PERSIST_PATH}")
        logger.info(f"Embedding model: {cls.EMBEDDING_MODEL_NAME}")
        logger.info(f"LLM model: {cls.LLM_MODEL_NAME}")

#############################################################################
# CONVERSATION HANDLING
#############################################################################

class ConversationPatterns:
    """Patterns for detecting and responding to conversation elements."""
    
    # Greeting patterns and responses
    GREETING_PATTERNS = [
        r'\b(?:hi|hello|hey|greetings|howdy|good\s*(?:morning|afternoon|evening)|what\'s\s*up)\b',
        r'\bhow\s+are\s+you\b',
    ]

    GREETING_RESPONSES = [
        "Hello! I'm your document assistant. How can I help you with your documents today?",
        "Hi there! I'm ready to help answer questions about your documents. What would you like to know?",
        "Greetings! I'm here to assist with information from your documents. What are you looking for?",
    ]

    # Farewell patterns
    FAREWELL_PATTERNS = [
        r'\b(?:bye|goodbye|see\s*you|farewell|exit|quit)\b',
    ]

    # Acknowledgement patterns and responses
    ACKNOWLEDGEMENT_PATTERNS = [
        r'\b(?:thanks|thank\s*you)\b',
        r'\bappreciate\s*(?:it|that)\b',
        r'\b(?:awesome|great|cool|nice)\b',
        r'\bthat\s*(?:helps|helped)\b',
        r'\bgot\s*it\b',
    ]

    ACKNOWLEDGEMENT_RESPONSES = [
        "You're welcome! Is there anything else you'd like to know about your documents?",
        "Happy to help! Let me know if you have any other questions.",
        "My pleasure! Feel free to ask if you need anything else.",
        "Glad I could assist. Any other questions about your documents?",
    ]

    @classmethod
    def detect_conversation_type(cls, query: str) -> Optional[str]:
        """
        Detects if input is a greeting, acknowledgement, or farewell.
        Returns an appropriate response or None.
        """
        query_lower = query.lower().strip()

        # Check for greetings
        for pattern in cls.GREETING_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                import random
                return random.choice(cls.GREETING_RESPONSES)

        # Check for acknowledgements
        for pattern in cls.ACKNOWLEDGEMENT_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                import random
                return random.choice(cls.ACKNOWLEDGEMENT_RESPONSES)

        # Check for farewells (handled by main loop)
        for pattern in cls.FAREWELL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return None

        # Not a special conversation type
        return None

#############################################################################
# DOCUMENT PROCESSING
#############################################################################

class DocumentProcessor:
    """Handles loading, processing, and splitting documents."""
    
    @staticmethod
    def check_dependencies() -> bool:
        """Verify that required dependencies are installed."""
        missing_deps = []
        
        try:
            import docx2txt
        except ImportError:
            missing_deps.append("docx2txt (for DOCX files)")
        
        try:
            import pypdf
        except ImportError:
            missing_deps.append("pypdf (for PDF files)")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
            logger.warning("Some document types may not load correctly.")
            logger.warning("Install missing dependencies with: pip install " + " ".join([d.split(' ')[0] for d in missing_deps]))
            return False
            
        return True
    
    @staticmethod
    def get_file_loader(file_path: str):
        """Returns appropriate loader based on file extension with error handling."""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)  # Define filename variable

        try:
            if ext == '.pdf':
                return PyPDFLoader(file_path)
            elif ext in ['.docx', '.doc']:
                return Docx2txtLoader(file_path)
            elif ext in ['.ppt', '.pptx']:
                return UnstructuredPowerPointLoader(file_path)
            elif ext == '.epub':
                logger.info(f"Loading EPUB file: {filename}")
                try:
                    return UnstructuredEPubLoader(file_path)
                except Exception as e:
                    logger.warning(f"UnstructuredEPubLoader failed: {e}, trying alternative EPUB loader")
                    # Use DocumentProcessor instead of cls
                    docs = DocumentProcessor.load_epub(file_path)
                    if docs:
                        class CustomEpubLoader(BaseLoader):
                            def __init__(self, documents):
                                self.documents = documents
                            def load(self):
                                return self.documents
                        return CustomEpubLoader(docs)
                    return None
            elif ext in ['.txt', '.md', '.csv']:
                return TextLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {ext} for file {filename}")
                return None
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {str(e)}")
            return None
        
    @classmethod
    def load_documents(cls, path: str, extensions: List[str] = None) -> List:
        """
        Load documents from specified path with specified extensions.
        Returns a list of documents or empty list if none found.
        """
        if extensions is None:
            extensions = Config.SUPPORTED_EXTENSIONS
            
        logger.info(f"Loading documents from: {path}")
        
        try:
            # Find all files with supported extensions
            all_files = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in extensions:
                        all_files.append(file_path)

            if not all_files:
                logger.warning(f"No compatible documents found in {path}")
                return []

            logger.info(f"Found {len(all_files)} compatible files")

            # Load each file with its appropriate loader
            all_documents = []
            for file_path in all_files:
                try:
                    logger.info(f"Loading: {os.path.basename(file_path)}")
                    loader = cls.get_file_loader(file_path)
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
                            logger.info(f"Loaded {len(docs)} sections from {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    continue

            # Filter out empty documents
            valid_documents = [doc for doc in all_documents if doc.page_content and doc.page_content.strip()]
            
            if not valid_documents:
                logger.warning("No document content could be extracted successfully")
                
            logger.info(f"Successfully loaded {len(valid_documents)} total document sections")
            return valid_documents

        except Exception as e:
            logger.error(f"Document loading error: {e}")
            return []
    
    @staticmethod
    def load_epub(file_path: str):
        """
        Custom EPUB loader using ebooklib.
        Returns a list of LangChain Document objects.
        """
        try:
            from ebooklib import epub
            
            filename = os.path.basename(file_path)
            logger.info(f"Loading EPUB with custom loader: {filename}")
            
            # Load the EPUB file
            book = epub.read_epub(file_path)
            
            # Extract and process content
            documents = []
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            
            # Get book title and metadata
            title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "Unknown Title"
            
            # Process each chapter/item
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    # Extract HTML content
                    html_content = item.get_content().decode('utf-8')
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Get plain text content
                    text = h2t.handle(str(soup))
                    
                    if text.strip():
                        # Create a document with metadata
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': file_path,
                                'filename': filename,
                                'title': title,
                                'chapter': item.get_name(),
                            }
                        )
                        documents.append(doc)
            
            logger.info(f"Extracted {len(documents)} sections from EPUB")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading EPUB file {file_path}: {str(e)}")
            return None
    
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

#############################################################################
# VECTOR STORE MANAGEMENT
#############################################################################

class VectorStoreManager:
    """Manages the vector database operations."""
    
    @staticmethod
    def get_embedding_device():
        """Determine the best available device for embeddings."""
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU for embeddings")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon MPS for embeddings")
            return 'mps'
        else:
            logger.info("Using CPU for embeddings")
            return 'cpu'
    
    @staticmethod
    def create_embeddings(model_name=None):
        """Create the embedding model."""
        if model_name is None:
            model_name = Config.EMBEDDING_MODEL_NAME
            
        device = VectorStoreManager.get_embedding_device()
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    @staticmethod
    def reset_chroma_db(persist_directory):
        """Reset the ChromaDB environment - handles file system operations safely."""
        logger.info(f"Resetting vector store at {persist_directory}")
        
        # Release resources via garbage collection
        import gc
        gc.collect()
        time.sleep(0.5)
        
        # Remove existing directory if it exists
        if os.path.exists(persist_directory):
            try:
                # Make directory writable first (for Windows compatibility)
                if sys.platform == 'win32':
                    for root, dirs, files in os.walk(persist_directory):
                        for dir in dirs:
                            os.chmod(os.path.join(root, dir), 0o777)
                        for file in files:
                            os.chmod(os.path.join(root, file), 0o777)
                
                # Try Python's built-in directory removal first
                shutil.rmtree(persist_directory)
            except Exception as e:
                logger.warning(f"Error removing directory with shutil: {e}")
                
                # Fallback to system commands
                try:
                    if sys.platform == 'win32':
                        os.system(f"rd /s /q \"{persist_directory}\"")
                    else:
                        os.system(f"rm -rf \"{persist_directory}\"")
                except Exception as e2:
                    logger.error(f"Failed to remove directory: {e2}")
                    return False
        
        # Create fresh directory structure
        try:
            os.makedirs(persist_directory, exist_ok=True)
            
            # Set appropriate permissions
            if sys.platform != 'win32':
                os.chmod(persist_directory, 0o755)
            
            # Create .chroma subdirectory for ChromaDB to recognize
            os.makedirs(os.path.join(persist_directory, ".chroma"), exist_ok=True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create vector store directory: {e}")
            return False
    
    @classmethod
    def get_or_create_vector_store(cls, chunks=None, embeddings=None, persist_directory=None):
        """
        Get existing vector store or create a new one.
        
        Args:
            chunks: Document chunks to index (if creating new store)
            embeddings: Embedding model to use
            persist_directory: Directory to store the vector database
            
        Returns:
            The vector store or None if creation fails
        """
        if persist_directory is None:
            persist_directory = Config.PERSIST_PATH
            
        # If force reindex flag is set or directory doesn't exist, reset it
        if Config.FORCE_REINDEX or not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            cls.reset_chroma_db(persist_directory)
        
        # Try to load existing store if no chunks provided
        if chunks is None and os.path.exists(persist_directory) and os.listdir(persist_directory):
            try:
                logger.info(f"Loading existing vector store from: {persist_directory}")
                vector_store = Chroma(
                    persist_directory=persist_directory, 
                    embedding_function=embeddings
                )
                return vector_store
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                # If loading fails, reset and prepare for recreation
                cls.reset_chroma_db(persist_directory)
        
        # Create new vector store if chunks are provided
        if chunks:
            try:
                logger.info(f"Creating new vector store with {len(chunks)} chunks")
                
                # Configure ChromaDB properly
                chroma_settings = Settings(
                    anonymized_telemetry=False,
                    persist_directory=persist_directory
                )
                
                # Create the vector store with the chunks
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_metadata={"hnsw:space": "cosine"},
                    client_settings=chroma_settings
                )
                
                # Explicitly persist to ensure it's saved
                if hasattr(vector_store, 'persist'):
                    vector_store.persist()
                    logger.info("Vector store persisted successfully")
                
                return vector_store
                
            except sqlite3.OperationalError as sqlerr:
                logger.error(f"Database error: {sqlerr}")
                return None
                
            except Exception as e:
                logger.error(f"Error creating vector store: {e}")
                return None
        
        logger.error("Cannot create or load vector store - no chunks provided and no existing store")
        return None
    
    @staticmethod
    def print_document_statistics(vector_store):
        """Print statistics about indexed documents."""
        if not vector_store:
            logger.warning("No vector store available for statistics")
            return
            
        try:
            all_docs = vector_store.get()
            if not all_docs or not all_docs.get('documents'):
                logger.warning("Vector store appears to be empty")
                return
                
            all_metadata = all_docs.get('metadatas', [])
            
            # Count documents by filename
            doc_counts = {}
            for metadata in all_metadata:
                if metadata and 'filename' in metadata:
                    filename = metadata['filename']
                    doc_counts[filename] = doc_counts.get(filename, 0) + 1
            
            total_chunks = len(all_docs.get('documents', []))
            unique_files = len(doc_counts)
            
            logger.info(f"Vector store contains {total_chunks} chunks from {unique_files} files")
            
            # Print file statistics
            print(f"\nKnowledge base contains {unique_files} documents ({total_chunks} total chunks):")
            for filename, count in sorted(doc_counts.items()):
                print(f"  - {filename}: {count} chunks")
                
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
    
    @staticmethod
    def verify_document_indexed(vector_store, doc_name):
        """Verify if a specific document is properly indexed."""
        if not vector_store:
            return False
            
        try:
            # Search for the document name in the vector store
            results = vector_store.similarity_search(f"information from {doc_name}", k=3)
            
            # Check if any results match this filename
            for doc in results:
                filename = doc.metadata.get('filename', '')
                if doc_name.lower() in filename.lower():
                    logger.info(f"Document '{doc_name}' is indexed in the vector store")
                    return True
                    
            logger.warning(f"Document '{doc_name}' was not found in the vector store")
            return False
                
        except Exception as e:
            logger.error(f"Error verifying document indexing: {e}")
            return False

#############################################################################
# RAG SYSTEM
#############################################################################

class RAGSystem:
    """Manages the RAG processing pipeline."""
    
    @staticmethod
    def format_docs(docs):
        """Format retrieved documents for inclusion in the prompt."""
        if not docs:
            return "No relevant documents found."
            
        formatted_docs = []
        unique_filenames = set()
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown source')
            filename = doc.metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown file')
            unique_filenames.add(filename)
            
            # Include metadata if available
            page_info = f"page {doc.metadata.get('page', '')}" if doc.metadata.get('page', '') else ""
            chunk_info = f"chunk {i+1}/{len(docs)}"
            
            metadata_line = f"Document {i+1} (from {filename} {page_info} {chunk_info}):\n"
            formatted_text = f"{metadata_line}{doc.page_content}\n\n"
            formatted_docs.append(formatted_text)

        # Add a summary of documents
        summary = f"Retrieved {len(docs)} chunks from {len(unique_filenames)} files: {', '.join(unique_filenames)}\n\n"
        
        return summary + "\n".join(formatted_docs)
    
    @staticmethod
    def stream_ollama_response(prompt, model_name=None, base_url=None):
        """Stream response from Ollama API with token-by-token output."""
        if model_name is None:
            model_name = Config.LLM_MODEL_NAME
            
        if base_url is None:
            base_url = Config.OLLAMA_BASE_URL
            
        url = f"{base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": True
        }

        full_response = ""

        try:
            # Test connection to Ollama API
            test_url = f"{base_url}/api/tags"
            try:
                test_response = requests.get(test_url, timeout=5)
                if test_response.status_code != 200:
                    logger.error(f"Ollama API unavailable: HTTP {test_response.status_code}")
                    return "Error: Could not connect to Ollama API. Make sure Ollama is running."
            except requests.RequestException as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                return "Error: Could not connect to Ollama API. Make sure Ollama is running and accessible."

            # Process the streaming response
            with requests.post(url, headers=headers, json=data, stream=True, timeout=30) as response:
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return f"Error: Failed to generate response (HTTP {response.status_code})"

                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            if 'response' in json_line:
                                token = json_line['response']
                                full_response += token
                                print(token, end='', flush=True)

                            if json_line.get('done', False):
                                break
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON from Ollama API: {line}")
        except Exception as e:
            logger.error(f"Error during Ollama request: {e}")
            return f"Error: {str(e)}"

        print()  # Add newline at end
        return full_response
    
    @classmethod
    def setup_rag_chain(cls, vector_store, llm_model=None, retriever_k=None, search_type=None):
        """Set up the RAG chain with vector store, embeddings, and LLM."""
        if vector_store is None:
            logger.error("Vector store is not initialized")
            return None, None, None
            
        if llm_model is None:
            llm_model = Config.LLM_MODEL_NAME
            
        if retriever_k is None:
            retriever_k = Config.RETRIEVER_K
            
        if search_type is None:
            search_type = Config.RETRIEVER_SEARCH_TYPE

        # Configure retriever
        retriever_kwargs = {'k': retriever_k}
        if search_type == "mmr":
            # For MMR (Maximum Marginal Relevance), fetch more candidates then diversify
            retriever_kwargs['fetch_k'] = retriever_k * 3
            retriever_kwargs['lambda_mult'] = 0.5  # Balance between relevance and diversity

        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=retriever_kwargs
        )

        # Create prompt template
        template = """
        You are a helpful AI assistant answering questions about a collection of documents. You have access to information contained in these documents, and you'll answer questions about their content.

        Guidelines for your responses:
        1. If you find the answer in the documents, respond directly and cite your sources with specific filenames
        2. If the documents contain partial information, use it and make clear where your information comes from
        3. If the question is about general knowledge unrelated to the documents, you can answer it like a normal conversation
        4. If the question is a greeting or casual conversation, respond naturally as a friendly assistant
        5. If you cannot find specific information in the documents, say "I don't have specific information about that in the documents" rather than making up information
        6. Always use specific information from the documents when available instead of giving generic answers
        7. If asked about a specific document or file by name, focus on information from that file and mention clearly whether that file is in your knowledge base

        Important: For this query, I've retrieved multiple document chunks that might contain relevant information. These chunks represent only a small subset of all indexed documents - I have access to many more documents than just these chunks.

        Context from retrieved documents:
        {context}

        Chat History:
        {chat_history}

        Question: {question}

        Helpful Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Initialize LLM
        llm = Ollama(model=llm_model)

        # Create conversation memory
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        # Add a system message to memory
        system_message = SystemMessage(content="I am an AI assistant that helps with answering questions about documents. I can also engage in casual conversation.")
        memory.chat_memory.messages.append(system_message)

        # Create the RAG chain function
        def rag_chain(input_dict, stream=True):
            question = input_dict.get("question", "")

            # Detection of social conversation is handled at a higher level

            # Retrieve relevant document chunks
            try:
                context_docs = retriever.invoke(question)
                context = cls.format_docs(context_docs)
                logger.info(f"Retrieved {len(context_docs)} chunks for query")
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
                context = "Error retrieving relevant documents."

            # Format chat history
            chat_history = input_dict.get("chat_history", [])
            formatted_history = ""
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    formatted_history += f"Human: {message.content}\n"
                elif isinstance(message, AIMessage):
                    formatted_history += f"AI: {message.content}\n"

            # Generate prompt with context and history
            prompt_input = {
                "context": context,
                "chat_history": formatted_history,
                "question": question
            }
            chain_response = prompt.invoke(prompt_input).to_string()

            # Generate response
            if stream:
                response = cls.stream_ollama_response(chain_response, llm_model)
                return response
            else:
                response = llm.invoke(chain_response)
                return response

        return rag_chain, retriever, memory

#############################################################################
# MAIN APPLICATION
#############################################################################

class CustomRAG:
    """Main RAG application class for command line interface."""
    
    def __init__(self):
        """Initialize the RAG application."""
        self.vector_store = None
        self.rag_chain = None
        self.retriever = None
        self.memory = None
        
    def initialize(self):
        """Set up all components of the RAG system."""
        # Setup configuration
        Config.setup()
        
        # Check dependencies
        DocumentProcessor.check_dependencies()
        
        # Create embeddings
        try:
            self.embeddings = VectorStoreManager.create_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False
            
        # Load or create vector store
        if Config.FORCE_REINDEX or not os.path.exists(Config.PERSIST_PATH) or not os.listdir(Config.PERSIST_PATH):
            logger.info("Creating new vector store")
            
            # Load and process documents
            documents = DocumentProcessor.load_documents(Config.DATA_PATH)
            if not documents:
                logger.error("No documents found to index")
                return False
                
            chunks = DocumentProcessor.split_documents(documents)
            if not chunks:
                logger.error("Failed to create document chunks")
                return False
                
            # Create vector store with chunks
            self.vector_store = VectorStoreManager.get_or_create_vector_store(chunks, self.embeddings)
            
            if self.vector_store:
                # Print statistics
                VectorStoreManager.print_document_statistics(self.vector_store)
        else:
            # Load existing vector store
            logger.info("Loading existing vector store")
            self.vector_store = VectorStoreManager.get_or_create_vector_store(None, self.embeddings)
            
            if self.vector_store:
                # Print statistics
                VectorStoreManager.print_document_statistics(self.vector_store)
        
        if not self.vector_store:
            logger.error("Failed to initialize vector store")
            return False
            
        # Set up RAG chain
        self.rag_chain, self.retriever, self.memory = RAGSystem.setup_rag_chain(self.vector_store)
        
        if not self.rag_chain:
            logger.error("Failed to set up RAG chain")
            return False
            
        return True
    
    def reindex_documents(self):
        """Reindex all documents in the data directory."""
        logger.info("Reindexing documents")
        print("Reindexing documents. This may take a while...")
        
        # Close existing resources
        if self.vector_store is not None:
            try:
                # Try to explicitly close Chroma client
                if hasattr(self.vector_store, '_collection') and self.vector_store._collection is not None:
                    if hasattr(self.vector_store._collection, '_client'):
                        if hasattr(self.vector_store._collection._client, 'close'):
                            self.vector_store._collection._client.close()
                
                self.vector_store = None
                self.retriever = None
                
                # Force garbage collection
                import gc
                gc.collect()
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error closing vector store: {e}")
        
        # Reset the vector store directory
        if not VectorStoreManager.reset_chroma_db(Config.PERSIST_PATH):
            logger.error("Failed to reset vector store")
            return False
        
        # Load and process documents
        try:
            documents = DocumentProcessor.load_documents(Config.DATA_PATH)
            if not documents:
                logger.error("No documents found to index")
                return False
                
            chunks = DocumentProcessor.split_documents(documents)
            if not chunks:
                logger.error("Failed to create document chunks")
                return False
                
            # Create fresh embeddings
            self.embeddings = VectorStoreManager.create_embeddings()
                
            # Create vector store with chunks
            self.vector_store = VectorStoreManager.get_or_create_vector_store(chunks, self.embeddings)
            
            if not self.vector_store:
                logger.error("Failed to create vector store")
                return False
                
            # Print statistics
            VectorStoreManager.print_document_statistics(self.vector_store)
            
            # Set up RAG chain
            self.rag_chain, self.retriever, self.memory = RAGSystem.setup_rag_chain(self.vector_store)
            
            if not self.rag_chain:
                logger.error("Failed to set up RAG chain")
                return False
                
            print(f"Reindexing completed successfully! Added {len(chunks)} chunks to the vector store.")
            return True
            
        except Exception as e:
            logger.error(f"Error during reindexing: {e}")
            return False
    
    def run_cli(self):
        """Run the interactive command line interface."""
        if not self.initialize():
            logger.error("Failed to initialize RAG system")
            return
            
        # Print banner and instructions
        print("\n" + "="*60)
        print("📚 CustomRAG - Document Assistant 📚")
        print("="*60)
        print("Ask questions about your documents.")
        print("Commands:")
        print("  - Type 'exit' or 'quit' to stop")
        print("  - Type 'clear' to reset the conversation memory")
        print("  - Type 'reindex' to reindex all documents")
        print("  - Type 'stats' to see document statistics")
        print("="*60 + "\n")
        
        # Main interaction loop
        while True:
            try:
                query = input("\nYour Question: ").strip()
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                # Handle commands
                if query.lower() in ["exit", "quit", "bye", "goodbye"]:
                    print("Goodbye! Have a great day!")
                    break
                    
                if query.lower() == "clear":
                    # Reset memory but keep system message
                    system_message = self.memory.chat_memory.messages[0] if self.memory.chat_memory.messages else None
                    self.memory.clear()
                    if system_message:
                        self.memory.chat_memory.messages.append(system_message)
                    print("Conversation memory has been reset.")
                    continue
                    
                if query.lower() == "stats":
                    VectorStoreManager.print_document_statistics(self.vector_store)
                    continue
                    
                if query.lower() == "reindex":
                    self.reindex_documents()
                    continue
                
                # Process query
                start_time = time.time()
                
                # Add query to memory
                self.memory.chat_memory.add_user_message(query)
                
                print("\nThinking...\n")
                
                try:
                    # Check for social responses first (greetings, thanks, etc.)
                    social_response = ConversationPatterns.detect_conversation_type(query)
                    if social_response:
                        # Handle social messages directly
                        print(f"Response: {social_response}")
                        self.memory.chat_memory.add_ai_message(social_response)
                    else:
                        # Process with RAG pipeline
                        input_dict = {
                            "question": query,
                            "chat_history": self.memory.chat_memory.messages
                        }
                        
                        # Get response
                        response = self.rag_chain(input_dict)
                        print(f"Response: {response}")
                        
                        # Add to memory
                        self.memory.chat_memory.add_ai_message(response)
                        
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    print("\n[Error occurred during response generation]")
                
                end_time = time.time()
                logger.info(f"Query processed in {end_time - start_time:.2f} seconds")
                
            except KeyboardInterrupt:
                print("\nGoodbye! Have a great day!")
                break
            except Exception as e:
                logger.error(f"Error in CLI: {e}")
                print("\n[An unexpected error occurred. Please try again.]")

def main():
    """Entry point for the CustomRAG application."""
    try:
        app = CustomRAG()
        app.run_cli()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())